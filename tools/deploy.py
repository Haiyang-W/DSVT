from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
import os
import numpy as np
import torch
import onnx
import onnxruntime as ort
import torch.nn as nn

from typing import Sequence, NamedTuple


####### load model #######
cfg_file = "./cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml"
cfg_from_yaml_file(cfg_file, cfg)
if os.path.exists('./deploy_files')==False:
    os.mkdir('./deploy_files')
log_file = './deploy_files/log_trt.log'
logger = common_utils.create_logger(log_file, rank=0)
test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False, workers=8, logger=logger, training=False
)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
ckpt = "path to dsvt piller version ckpt"
model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False, pre_trained_path=None)
model.eval()
model.cuda()
####### load model #######

####### read input #######
batch_dict = torch.load("path to batch_dict.pth", map_location="cuda")
inputs = batch_dict
####### read input #######

####### DSVT #######
class AllDSVTBlocksTRT(nn.Module):
    def __init__(self, dsvtblocks_list, layer_norms_list):
        super().__init__()
        self.layer_norms_list = layer_norms_list
        self.dsvtblocks_list = dsvtblocks_list
    def forward(
        self,
        pillar_features, 
        set_voxel_inds_tensor_shift_0,
        set_voxel_inds_tensor_shift_1,
        set_voxel_masks_tensor_shift_0, 
        set_voxel_masks_tensor_shift_1,
        pos_embed_tensor,
    ):
        outputs = pillar_features

        residual = outputs
        blc_id = 0
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        
        outputs = self.layer_norms_list[blc_id](residual + outputs)

        residual = outputs
        blc_id = 1
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        
        outputs = self.layer_norms_list[blc_id](residual + outputs)

        residual = outputs
        blc_id = 2
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        
        outputs = self.layer_norms_list[blc_id](residual + outputs)

        residual = outputs
        blc_id = 3
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.dsvtblocks_list[blc_id].encoder_list[set_id](*inputs)
        
        outputs = self.layer_norms_list[blc_id](residual + outputs)

        return outputs
####### DSVT #######

####### torch to onnx #######
with torch.no_grad():
    DSVT_Backbone = model.backbone_3d
    dsvtblocks_list = DSVT_Backbone.stage_0
    layer_norms_list = DSVT_Backbone.residual_norm_stage_0
    inputs = model.vfe(inputs)
    voxel_info = DSVT_Backbone.input_layer(inputs)
    set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}'] for i in range(2)] for s in range(1)]
    set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}'] for i in range(2)] for s in range(1)]
    pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}'] for i in range(2)] for b in range(4)] for s in range(1)]

    pillar_features = inputs['voxel_features']
    alldsvtblockstrt_inputs = (
        pillar_features,
        set_voxel_inds_list[0][0],
        set_voxel_inds_list[0][1],
        set_voxel_masks_list[0][0],
        set_voxel_masks_list[0][1],
        torch.stack([torch.stack(v, dim=0) for v in pos_embed_list[0]], dim=0),
    )

    jit_mode = "trace"
    input_names = [
        'src',
        'set_voxel_inds_tensor_shift_0', 
        'set_voxel_inds_tensor_shift_1', 
        'set_voxel_masks_tensor_shift_0', 
        'set_voxel_masks_tensor_shift_1',
        'pos_embed_tensor'
    ]
    output_names = ["output",]
    input_shapes = {
        "src": {
            "min_shape": [24629, 192],
            "opt_shape": [24629, 192],
            "max_shape": [24629, 192],
        },
        "set_voxel_inds_tensor_shift_0": {
            "min_shape": [2, 1156, 36],
            "opt_shape": [2, 1156, 36],
            "max_shape": [2, 1156, 36],
        },
        "set_voxel_inds_tensor_shift_1": {
            "min_shape": [2, 834, 36],
            "opt_shape": [2, 834, 36],
            "max_shape": [2, 834, 36],
        },
        "set_voxel_masks_tensor_shift_0": {
            "min_shape": [2, 1156, 36],
            "opt_shape": [2, 1156, 36],
            "max_shape": [2, 1156, 36],
        },
        "set_voxel_masks_tensor_shift_1": {
            "min_shape": [2, 834, 36],
            "opt_shape": [2, 834, 36],
            "max_shape": [2, 834, 36],
        },
        "pos_embed_tensor": {
            "min_shape": [4, 2, 24629, 192],
            "opt_shape": [4, 2, 24629, 192],
            "max_shape": [4, 2, 24629, 192],
        },
    }


    dynamic_axes = {
        "src": {
            0: "voxel_number",
        },
        "set_voxel_inds_tensor_shift_0": {
            1: "set_number_shift_0",
        },
        "set_voxel_inds_tensor_shift_1": {
            1: "set_number_shift_1",
        },
        "set_voxel_masks_tensor_shift_0": {
            1: "set_number_shift_0",
        },
        "set_voxel_masks_tensor_shift_1": {
            1: "set_number_shift_1",
        },
        "pos_embed_tensor": {
            2: "voxel_number",
        },
        "output": {
            0: "voxel_number",
        }
    }

    base_name = "./deploy_files/dsvt"
    ts_path = f"{base_name}.ts"
    onnx_path = f"{base_name}.onnx"

    allptransblocktrt = AllDSVTBlocksTRT(dsvtblocks_list, layer_norms_list).eval().cuda()
    torch.onnx.export(
        allptransblocktrt,
        alldsvtblockstrt_inputs,
        onnx_path, input_names=input_names,
        output_names=output_names, dynamic_axes=dynamic_axes,
        opset_version=14,
    )
    # test onnx
    ort_session = ort.InferenceSession(onnx_path)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(pillar_features),
                  ort_session.get_inputs()[1].name: to_numpy(set_voxel_inds_list[0][0]),
                  ort_session.get_inputs()[2].name: to_numpy(set_voxel_inds_list[0][1]),
                  ort_session.get_inputs()[3].name: to_numpy(set_voxel_masks_list[0][0]),
                  ort_session.get_inputs()[4].name: to_numpy(set_voxel_masks_list[0][1]),
                  ort_session.get_inputs()[5].name: to_numpy(torch.stack([torch.stack(v, dim=0) for v in pos_embed_list[0]], dim=0)),}
    ort_outs = ort_session.run(None, ort_inputs) 
####### torch to onnx #######


####### torch to trt engine #######
# trtexec --onnx={path to onnx} --saveEngine={path to save trtengine} \
# --memPoolSize=workspace:4096 --verbose --buildOnly --device=1 --fp16 \
# --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=src:3000x192,set_voxel_inds_tensor_shift_0:2x170x36,set_voxel_inds_tensor_shift_1:2x100x36,set_voxel_masks_tensor_shift_0:2x170x36,set_voxel_masks_tensor_shift_1:2x100x36,pos_embed_tensor:4x2x3000x192 \
# --optShapes=src:20000x192,set_voxel_inds_tensor_shift_0:2x1000x36,set_voxel_inds_tensor_shift_1:2x700x36,set_voxel_masks_tensor_shift_0:2x1000x36,set_voxel_masks_tensor_shift_1:2x700x36,pos_embed_tensor:4x2x20000x192 \
# --maxShapes=src:35000x192,set_voxel_inds_tensor_shift_0:2x1500x36,set_voxel_inds_tensor_shift_1:2x1200x36,set_voxel_masks_tensor_shift_0:2x1500x36,set_voxel_masks_tensor_shift_1:2x1200x36,pos_embed_tensor:4x2x35000x192 \
####### torch to trt engine #######
