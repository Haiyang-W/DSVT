[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dsvt-dynamic-sparse-voxel-transformer-with/3d-object-detection-on-waymo-vehicle)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle?p=dsvt-dynamic-sparse-voxel-transformer-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dsvt-dynamic-sparse-voxel-transformer-with/3d-object-detection-on-waymo-pedestrian)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-pedestrian?p=dsvt-dynamic-sparse-voxel-transformer-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dsvt-dynamic-sparse-voxel-transformer-with/3d-object-detection-on-waymo-cyclist)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-cyclist?p=dsvt-dynamic-sparse-voxel-transformer-with)


[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2301.06051) [![GitHub Stars](https://img.shields.io/github/stars/Haiyang-W/DSVT?style=social)](https://github.com/Haiyang-W/DSVT) ![visitors](https://visitor-badge.glitch.me/badge?page_id=Haiyang-W/DSVT) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7a0a7f47c1614b813e35e15a2c0c0a488ee5e0aa%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/DSVT%3A-Dynamic-Sparse-Voxel-Transformer-with-Rotated-Wang-Shi/7a0a7f47c1614b813e35e15a2c0c0a488ee5e0aa)


# DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets
	
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-pedestrian)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-pedestrian?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-cyclist)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-cyclist?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-vehicle)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle?p=embracing-single-stride-3d-object-detector) -->

This repo is the official implementation of: [DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets](https://arxiv.org/abs/2301.06051) as well as the follow-ups. Our DSVT achieves state-of-the-art performance on large-scale Waymo Open Dataset with real-time inference speed (27Hz).

> DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets
>
> [Haiyang Wang*](https://scholar.google.com/citations?user=R3Av3IkAAAAJ&hl=en&oi=ao), Chen Shi*, Shaoshuai Shi $^\dagger$, Meng Lei, Sen Wang, Di He, Bernt Schiele, Liwei Wang $^\dagger$
> - Primary contact: Haiyang Wang ( wanghaiyang6@stu.pku.edu.cn )

<div align="center">
  <img src="assets/Figure2.png" width="500"/>
</div>

## News
- [23-01-15] DSVT is released on [arXiv](https://arxiv.org/abs/2301.06051).
- [23-02-28] 🔥 DSVT is accepted at CVPR 2023.
- [23-03-30] Code of Waymo is released.

## TODO

- [x] Release the [arXiv](https://arxiv.org/abs/2301.06051) version.
- [x] SOTA performance of 3D object detection (Waymo & Nuscenes) and BEV Map Segmentation (Nuscenes).
- [x] Clean up and release the code of Waymo.
- [ ] Release code of NuScenes.
- [ ] Merge DSVT to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Introduction
Dynamic Sparse Voxel Transformer is an efficient yet deployment-friendly 3D transformer backbone for outdoor 3D object detection. It partitions a series of local regions in each window according to its sparsity and then computes the features of all regions in a fully parallel manner. Moreover, to allow the cross-set connection, it designs a rotated set partitioning strategy that alternates between two partitioning configurations in consecutive self-attention layers.

DSVT achieves state-of-the-art performance on large-scale Waymo one-sweeps 3D object detection (`78.2 mAPH L1` and `72.1 mAPH L2` on one-stage setting) and (`78.9 mAPH L1` and `72.8 mAPH L2` on two-stage setting), surpassing previous models by a large margin. Moreover, as for multiple sweeps setting ( `2`, `3`, `4` sweeps settings), our model reaches `74.6 mAPH L2`, `75.0 mAPH L2` and `75.6 mAPH L2` in terms of one-stage framework and `75.1 mAPH L2`, `75.5 mAPH L2` and `76.2 mAPH L2` on two-stage framework, which outperforms the previous best multi-frame methods with a large margin. Note that our model is not specifically designed for multi-frame detection, and only takes concatenated point clouds as input.

![Pipeline](assets/Figure3_sc.png)

## Main results
**We provide the pillar and voxel 3D version of one-stage DSVT. The two-stage versions with [CT3D](https://github.com/hlsheng1/CT3D) are also listed below.**
### 3D Object Detection (on Waymo validation)
We run training for 3 times and report average metrics across all results.
#### One-Sweeps Setting
|  Model  |  #Sweeps | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Log |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  [DSVT(Pillar)](tools/cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml) | 1       |  79.5/77.1  | 73.2/71.0  | 79.3/78.8 | 70.9/70.5 | 82.8/77.0 | 75.2/69.8 | 76.4/75.4 | 73.6/72.7 | [Log](https://drive.google.com/file/d/1DlEMIb-ZUFilJBDd8fuyb8nuRnSFPzWy/view?usp=share_link) |
|  [DSVT(Voxel)](tools/cfgs/dsvt_models/dsvt_3D_1f_onestage.yaml) | 1        |  80.3/78.2  |  74.0/72.1  | 79.7/79.3 | 71.4/71.0 | 83.7/78.9 | 76.1/71.5 | 77.5/76.5 | 74.6/73.7 | [Log](https://drive.google.com/file/d/19Z8Q6Mp945XJaLuccb5rtYejGQdl7xjG/view?usp=share_link) | 
|  DSVT(Pillar-TS) | 1    |  80.6/78.2  |  74.3/72.1  | 80.2/79.7 | 72.0/71.6 | 83.7/78.0 | 76.1/70.7 | 77.8/76.8 | 74.9/73.9 | [Log](https://drive.google.com/file/d/1YeVpF7An79yBZApCkagfnwGgFniZWRZl/view?usp=share_link) | 
|  DSVT(Voxel-TS) | 1     |  81.1/78.9  |  74.8/72.8  | 80.4/79.9 | 72.2/71.8 | 84.2/79.3 | 76.5/71.8 | 78.6/77.6 | 75.7/74.7 | [Log](https://drive.google.com/file/d/1hwrQ2iEiuIBKlXn1UhOGYhwdfogF9PpY/view?usp=share_link) | 

#### Multi-Sweeps Setting
##### 2-Sweeps
|  Model  |  #Sweeps | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Log |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  DSVT(Pillar) | 2        |  81.4/79.8  |  75.4/73.9  | 80.8/80.3 | 72.7/72.3 | 84.5/81.3 | 77.2/74.1 | 78.8/77.9 | 76.3/75.4 | [Log](https://drive.google.com/file/d/13uFqSxxlQuywLvU5CZhK9R7JoGqVXAt4/view?usp=share_link) |
|  DSVT(Voxel) | 2       |  81.9/80.4  |  76.0/74.6  | 81.1/80.6 | 73.0/72.6 | 84.9/81.7 | 77.8/74.8 | 79.8/78.9 | 77.3/76.4 | [Log](https://drive.google.com/file/d/1Ketd-x5MvJXBbJiFAeVtFDQyyACGekL-/view?usp=share_link) |
|  DSVT(Pillar-TS) | 2       |  81.9/80.4  |  76.0/74.5  | 81.3/80.9 | 73.4/73.0 | 85.2/81.9 | 77.9/74.7 | 79.2/78.3 | 76.7/75.9 | [Log](https://drive.google.com/file/d/1tTyg3mkTTuYrbk3O_F2C5R8Yi0JYXfkb/view?usp=share_link) | 
|  DSVT(Voxel-TS) | 2       |  82.3/80.8  |  76.6/75.1  | 81.4/81.0 | 73.5/73.1 | 85.4/82.2 | 78.4/75.3 | 80.2/79.3 | 77.8/76.9 | [Log](https://drive.google.com/file/d/15BH2FwQIZQ3IVkPmexilrVd4BO7shIcK/view?usp=share_link) |

##### 3-Sweeps
|  Model  |  #Sweeps | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Log |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  DSVT(Pillar) | 3       |  81.9/80.5  |  76.2/74.8  | 81.2/80.8 | 73.3/72.9 | 85.0/82.0 | 78.0/75.0 | 79.6/78.8 | 77.2/76.4 | [Log](https://drive.google.com/file/d/1C7Koel3xVVixpkQenJyx7koIMQ2x-pbU/view?usp=share_link) | 
|  DSVT(Voxel) | 3       |  82.1/80.8  |  76.3/75.0  |  81.5/81.1 | 73.6/73.2 | 85.3/82.4 | 78.2/75.4 | 79.6/78.8 | 77.2/76.4 | [Log](https://drive.google.com/file/d/1BOiCTcN8Izz6hY3sQeifTjxSHnMkz-S5/view?usp=share_link) | 
|  DSVT(Pillar-TS) | 3       |  82.5/81.0  |  76.7/75.4  | 81.8/81.3 | 74.0/73.6 | 85.6/82.6 | 78.5/75.6 | 80.1/79.2 | 77.7/76.9 | [Log](https://drive.google.com/file/d/1ylMBirihLJLIVXEllLHxw5C7JQoYXZpL/view?usp=share_link) | 
|  DSVT(Voxel-TS) | 3       |  82.6/81.2  |  76.8/75.5  | 81.8/81.4 | 74.0/73.6 | 85.8/82.9 | 78.8/75.9 | 80.1/79.2 | 77.7/76.9 | [Log](https://drive.google.com/file/d/1eJChG0DulNXxcC5tysAIzFw6lyALFQEk/view?usp=share_link) | 

##### 4-Sweeps
|  Model  |  #Sweeps | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Log |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  DSVT(Pillar) | 4       |  82.5/81.0  |  76.7/75.3  |  81.7/81.2 | 73.8/73.4 | 85.4/82.3 | 78.5/75.5 | 80.3/79.4 | 77.9/77.1 | [Log](https://drive.google.com/file/d/1vziBYjNrACuf3T2dqq2rIPucJMKf827y/view?usp=share_link) | 
|  DSVT(Voxel) | 4       |  82.6/81.3  |  76.9/75.6  | 81.8/81.4 | 74.1/73.6 | 85.6/82.8 | 78.6/75.9 | 80.4/79.6 | 78.1/77.3 | [Log](https://drive.google.com/file/d/1e-2h03w19bIFcj99pvbpQ9Y682oq17tN/view?usp=share_link) | 
|  DSVT(Pillar-TS) | 4       |  82.9/81.5  |  77.3/75.9  | 82.1/81.6 | 74.4/74.0 | 85.8/82.8 | 79.0/76.1 | 80.9/80.0 | 78.6/77.7 | [Log](https://drive.google.com/file/d/1f9FzlPpk5qlqBXTU47O-QW7dTvVqr77x/view?usp=share_link) | 
|  DSVT(Voxel-TS) | 4       |  83.1/81.7  |  77.5/76.2  | 82.1/81.6 | 74.5/74.1 | 86.0/83.2 | 79.1/76.4 | 81.1/80.3 | 78.8/78.0 | [Log](https://drive.google.com/file/d/13ZnoOqdkwnjgLRSCM9VFHL3kqrw0N4oB/view?usp=share_link) | 


### 3D Object Detection (on NuScenes validation)
|  Model  | mAP | NDS | mATE | mASE | mAOE | mAVE| mAAE |
|---------|---------|--------|---------|---------|--------|---------|--------|
|  DSVT(Pillar) | 66.4 | 71.1 | 27.0 | 24.8 | 27.2 | 22.6 | 18.9|


### 3D Object Detection (on NuScenes test)
|  Model  | mAP | NDS | mATE | mASE | mAOE | mAVE| mAAE | results |
|---------|---------|--------|--------|---------|---------|--------|---------|--------|
|  DSVT(Pillar) | 68.4 | 72.7 | 24.8 | 23.0 | 29.6 | 24.6 | 13.6 | [result.json](https://drive.google.com/file/d/1lfi06sRroNrlrmkgocSiInJigeSYskLi/view?usp=share_link) |

### Bev Map Segmentation (on NuScenes validation)
|  Model  | Drivable |Ped.Cross.| Walkway |  StopLine  | Carpark |  Divider  |  mIoU  |
|---------|----------|--------|--------|--------|--------|---------|--------|
|  DSVT(Pillar) |   87.6   |   67.2   |   72.7   |   59.7   |   62.7  |   58.2   |   68.0   |

#### Inference Speed
We present a comparison with other state-of-the-art methods on both inference speed and performance accuracy. **After being deployed by NVIDIA TensorRT, our model can achieve a real-time running speed (27Hz).** 

![Speed](assets/Figure1_arxiv.png)


|  Model  |  Latency |  mAP_L2  | mAPH_L2 | 
|---------|---------|---------|--------|
|  Centerpoint-Pillar | 35ms       |  66.0   |  62.2  |
|  Centerpoint-Voxel | 40ms       |  68.2   |  65.8  |
|  PV-RCNN++(center) | 113ms       |  71.7   |  69.5  |
|  DSVT(Pillar) | 67ms       |  73.2   |  71.0  |  
|  DSVT(Voxel) | 97ms       |  74.0   |  72.1  |
|  DSVT(Pillar+TensorRt) | 37ms       |  73.2   |  71.0  |  


## Usage
### Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for installation.

### Dataset Preparation
Please follow the instructions from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md).

### Training
```
# multi-gpu training
cd tools
bash scripts/dist_train.sh 8 --cfg_file <CONFIG_FILE> --sync_bn [other optional arguments]
```
You can train the model with fp16 setting to save cuda memory, which may occasionally report gradient NaN error.
```
# fp16 training
cd tools
bash scripts/dist_train.sh 8 --cfg_file <CONFIG_FILE> --sync_bn --fp16 [other optional arguments]
```

### Testing
```
# multi-gpu testing
cd tools
bash scripts/dist_test.sh 8 --cfg_file <CONFIG_FILE> --ckpt <CHECKPOINT_FILE>
```


### Quick Start
- To cater to users with limited resources who require quick experimentation, we also provide results trained with a single frame of 20% data for 12 epoch on 8 RTX 3090 GPUs.
  
| Performance@(20% Data for 12 epoch)  |  Batch Size | Training time | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Log |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  [DSVT(Pillar)](tools/cfgs/dsvt_models/dsvt_plain_D512e.yaml) | 1   | ~5.5h  |  75.3/72.4  |  69.3/66.4 | 75.3/74.8 | 66.9/66.4 | 79.4/71.7 | 71.7/64.6 | 71.9/70.8 | 69.2/68.1 | [Log](https://drive.google.com/file/d/1XoLwwzDUGRRUv0hNeRNBoGwxaibH5KRG/view?usp=share_link) |
|  [DSVT(Voxel)](tools/cfgs/dsvt_models/dsvt_3D_D512e.yaml) | 1    | ~6.5h  |  76.2/73.6  | 69.9/67.4  | 75.7/75.2 | 67.2/66.8 | 80.1/73.7 | 72.5/66.4 | 72.8/71.8 | 70.1/69.1 | [Log](https://drive.google.com/file/d/14iZpyinw-_2HjI4oR1JpCvSMK9AI8wDf/view?usp=share_link) | 

- To reproduce the resutls in main paper, please refer the following configs.
  
| Performance@(100% Data for 24 epoch)  |  Batch Size | Training time | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Log |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  [DSVT(Pillar)](tools/cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml) | 3 |  ~22.5h      |  79.5/77.1  | 73.2/71.0  | 79.3/78.8 | 70.9/70.5 | 82.8/77.0 | 75.2/69.8 | 76.4/75.4 | 73.6/72.7 | [Log](https://drive.google.com/file/d/1DlEMIb-ZUFilJBDd8fuyb8nuRnSFPzWy/view?usp=share_link) |
|  [DSVT(Voxel)](tools/cfgs/dsvt_models/dsvt_3D_1f_onestage.yaml) | 3 | ~27.5h |  80.3/78.2  |  74.0/72.1  | 79.7/79.3 | 71.4/71.0 | 83.7/78.9 | 76.1/71.5 | 77.5/76.5 | 74.6/73.7 | [Log](https://drive.google.com/file/d/19Z8Q6Mp945XJaLuccb5rtYejGQdl7xjG/view?usp=share_link) | 


## Citation
Please consider citing our work as follows if it is helpful.
```
@inproceedings{wang2023dsvt,
    title={DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets},
    author={Haiyang Wang, Chen Shi, Shaoshuai Shi, Meng Lei, Sen Wang, Di He, Bernt Schiele and Liwei Wang},
    booktitle={CVPR},
    year={2023}
}
```

## Acknowledgments
This project is based on the following codebases.
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [SST](https://github.com/tusen-ai/SST)