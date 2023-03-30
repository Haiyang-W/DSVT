import numpy as np
import torch
import torch.nn as nn


class BasicStackConvLayers(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        
        conv_kwargs = self.model_cfg.conv_kwargs
        c_in = input_channels
        
        conv_list = []
        num_layers = len(conv_kwargs)
        for i in range(num_layers):
            c_out = conv_kwargs[i]['c_out']
            convnormrelu = nn.Sequential(
                nn.Conv2d(
                    c_in, c_out,
                    kernel_size=conv_kwargs[i]['kernel_size'],
                    dilation=conv_kwargs[i]['dilation'],
                    padding=conv_kwargs[i]['padding'],
                    stride=conv_kwargs[i]['stride'], bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True)
            )
            conv_list.append(convnormrelu)
            c_in = c_out

        self.conv_layer = nn.Sequential(*conv_list)
        self.num_bev_features = c_in
        
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']    
        ret_features = self.conv_layer(spatial_features)
        
        data_dict['spatial_features_2d'] = ret_features

        return data_dict
