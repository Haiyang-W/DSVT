import torch
import torch.nn as nn
# import spconv
from ...utils.spconv_utils import replace_feature, spconv


class Sparse2DBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        # encoder_layer_name: [c_in, c_out, cur_stride]
        # fpn_layer_name: encoder_layer_name
        self.map_layer_name_to_info = {}

        # total_stride: layer_name
        self.map_stride_to_layer = {}

        layer_cfgs = self.model_cfg.LAYER_CONFIG
        self.total_layers = len(layer_cfgs)
        self.conv_layer_list = []

        c_in = input_channels
        cur_stride = (1, 1)
        for layer_idx, cur_config in enumerate(layer_cfgs):
            cur_layer_name = f'conv{layer_idx + 1}'
            self.map_layer_name_to_info[cur_layer_name] = [c_in, cur_config['out_channels'], cur_config['stride']]

            c_in, cur_layer = self.make_layer(
                in_channels=c_in, out_channels=cur_config['out_channels'],
                num_blocks=cur_config['num_blocks'], stride=tuple(cur_config['stride']),
                ref_key=cur_layer_name, kernel_size=cur_config.get('kernel_size', 3),
            )
            self.__setattr__(cur_layer_name, cur_layer)

            cur_stride = (cur_stride[0] * cur_config['stride'][0], cur_stride[1] * cur_config['stride'][1])
            self.map_stride_to_layer[cur_stride] = cur_layer_name
            self.conv_layer_list.append(cur_layer_name)

        self.use_fpn = self.model_cfg.get('USE_FPN', False)
        self.up_layer_list = []
        if self.use_fpn:
            fpn_cfgs = self.model_cfg.FPN_CONFIG
            for fpn_layer_idx, cur_config in enumerate(fpn_cfgs):
                cur_layer_name = f'upsample_layer{fpn_layer_idx + 1}'
                up_lateral_layer_name = self.map_stride_to_layer[cur_stride]
                cur_stride = (cur_stride[0] // cur_config['stride'][0], cur_stride[1] // cur_config['stride'][1])
                lateral_layer_name = self.map_stride_to_layer[cur_stride]

                _, lateral_c_in, _ = self.map_layer_name_to_info[lateral_layer_name]

                self.map_layer_name_to_info[cur_layer_name] = lateral_layer_name

                upsample_layer, conv_layers = self.make_upsample_layer(
                    in_channels=c_in, out_channels=cur_config['out_channels'], lateral_channels=lateral_c_in,
                    stride=tuple(cur_config.get('stride', None)), num_convs=cur_config['num_convs'],
                    up_ref_key=up_lateral_layer_name, ref_key='_up' + lateral_layer_name,
                    kernel_size=cur_config.get('kernel_size', 3),
                )
                self.__setattr__(cur_layer_name, upsample_layer)
                self.__setattr__(cur_layer_name + '_postconv', conv_layers)

                c_in = cur_config['out_channels']
                self.up_layer_list.append(cur_layer_name)

        self.output_strides = cur_stride
        self.num_bev_features = c_in
        self.use_dense_output = self.model_cfg.get('USE_DENSE_OUTPUT', False)

    @staticmethod
    def make_layer(in_channels, out_channels, num_blocks, stride, ref_key=None, kernel_size=3, **kwargs):
        assert isinstance(stride, tuple), 'Please provide the full stride with tuple format'
        layers = []
        for k in range(num_blocks):
            if k == 0 and max(stride[0], stride[1]) > 1:
                layers.extend([
                    spconv.SparseConv2d(
                        in_channels, out_channels, kernel_size=3, stride=stride,
                        padding=1, bias=False, indice_key=ref_key + '_stride'
                    ),
                    nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            else:
                layers.extend([
                    spconv.SubMConv2d(
                        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                        indice_key=ref_key
                    ),
                    nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            in_channels = out_channels
        return out_channels, spconv.SparseSequential(*layers)

    def make_upsample_layer(self, in_channels, out_channels, lateral_channels, stride, num_convs,
                            up_ref_key=None, ref_key=None, kernel_size=3, radius=None, **kwargs):
        assert isinstance(stride, tuple), 'Please provide the full stride with tuple format'

        if max(stride[0], stride[1]) > 1:
            upsample_layer = spconv.SparseSequential(
                spconv.SparseInverseConv2d(
                    in_channels, out_channels, kernel_size=kernel_size, indice_key=up_ref_key + '_stride'
                ),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        else:
            upsample_layer = spconv.SparseSequential(
                spconv.SubMConv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                    indice_key=ref_key
                ),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )

        c_in = out_channels + lateral_channels
        conv_layers = []
        for k in range(num_convs):
            conv_layers.extend([
                spconv.SubMConv2d(
                    c_in, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                    indice_key=ref_key
                ),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ])

            c_in = out_channels
        conv_layers = spconv.SparseSequential(*conv_layers)
        return upsample_layer, conv_layers

    def forward(self, batch_dict):
        x = spconv.SparseConvTensor(
            features=batch_dict['pixel_features'],
            indices=batch_dict['pixel_coords'].int(),
            spatial_shape=batch_dict['spatial_shape'],
            batch_size=batch_dict['batch_size']
        )

        encoder_output_dict = {}
        for layer_name in self.conv_layer_list:
            x = self.__getattr__(layer_name)(x)
            encoder_output_dict[layer_name] = x

        if self.use_fpn:
            for layer_name in self.up_layer_list:
                lateral_layer_name = self.map_layer_name_to_info[layer_name]
                x_lateral = encoder_output_dict[lateral_layer_name]
                x_pre = self.__getattr__(layer_name)(x)
                # x_pre.features = torch.cat((x_pre.features, x_lateral.features), dim=1)
                x_pre = replace_feature(x_pre, torch.cat((x_pre.features, x_lateral.features), dim=1))
                x_cat = x_pre

                x = self.__getattr__(layer_name + '_postconv')(x_cat)

        if self.use_dense_output:
            batch_dict['spatial_features_2d'] = x.dense()  # (B, C, H, W)
        else:
            batch_dict['pixel_features'] = x.features
            batch_dict['pixel_coords'] = x.indices.long()
            batch_dict['spatial_shape_2d'] = x.spatial_shape

        return batch_dict
