# DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets
	
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-pedestrian)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-pedestrian?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-cyclist)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-cyclist?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-vehicle)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle?p=embracing-single-stride-3d-object-detector) -->

This repo is the official implementation of: [DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets](www.google.com) as well as the follow-ups. Our DSVT achieves state-of-the-art performance on large-scale Waymo Open Dataset with real-time inference speed (27Hz).

<div align="center">
  <img src="assets/Figure2.png" width="500"/>
</div>

## News
- [23-01-17] DSVT is released on [arXiv](www.google.com).

## Introduction
Dynamic Sparse Voxel Transformer is initially described in [arXiv](www.google.com), which is an efficient yet deployment-friendly 3D transformer backbone for outdoor 3D object detection. It partitions a series of local regions in each window according to its sparsity and then computes the features of all regions in a fully parallel manner. Moreover, to allow the cross-set connection, it designs a rotated set partitioning strategy that alternates between two partitioning configurations in consecutive self-attention layers.

DSVT achieves state-of-the-art performance on large-scale Waymo one-sweeps 3D object detection (`78.2 mAPH L1` and `72.1 mAPH L2` on one-stage setting) and (`78.9 mAPH L1` and `72.8 mAPH L2` on two-stage setting), surpassing previous models by a large margin. Moreover, as for multiple sweeps setting ( `2`, `3`, `4` sweeps settings), our model reaches `74.6 mAPH L2`, `75.0 mAPH L2` and `75.6 mAPH L2` in terms of one-stage framework and `75.1 mAPH L2`, `75.5 mAPH L2` and `76.2 mAPH L2` on two-stage framework, which outperforms the previous best multi-frame methods with a large margin. Note that our model is not specifically designed for multi-frame detection, and only takes concatenated point clouds as input.

![Pipeline](assets/Figure3_sc.png)

## Main results
**We provide the pillar and voxel 3D version of one-stage DSVT. The two-stage versions with [CT3D](https://github.com/hlsheng1/CT3D) are also listed below.**
### 3D Object Detection (on Waymo validation)
#### One-Sweeps Setting
|  Model  |  #Sweeps | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|
|  DSVT(Pillar) | 1       |  79.5/77.1  | 73.2/71.0  | 79.3/78.8 | 70.9/70.5 | 82.8/77.0 | 75.2/69.8 | 76.4/75.4 | 73.6/72.7 |
|  DSVT(Voxel) | 1        |  80.3/78.2  |  74.0/72.1  | 79.7/79.3 | 71.4/71.0 | 83.7/78.9 | 76.1/71.5 | 77.5/76.5 | 74.6/73.7 | 
|  DSVT(Pillar-TS) | 1    |  80.6/78.2  |  74.3/72.1  | 80.2/79.7 | 72.0/71.6 | 83.7/78.0 | 76.1/70.7 | 77.8/76.8 | 74.9/73.9 | 
|  DSVT(Voxel-TS) | 1     |  81.1/78.9  |  74.8/72.8  | 80.4/79.9 | 72.2/71.8 | 84.2/79.3 | 76.5/71.8 | 78.6/77.6 | 75.7/74.7 | 

#### Multi-Sweeps Setting
##### 2-Sweeps
|  Model  |  #Sweeps | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|
|  DSVT(Pillar) | 2        |  81.4/79.8  |  75.4/73.9  | 80.8/80.3 | 72.7/72.3 | 84.5/81.3 | 77.2/74.1 | 78.8/77.9 | 76.3/75.4 |
|  DSVT(Voxel) | 2       |  81.9/80.4  |  76.0/74.6  | 81.1/80.6 | 73.0/72.6 | 84.9/81.7 | 77.8/74.8 | 79.8/78.9 | 77.3/76.4 |
|  DSVT(Pillar-TS) | 2       |  81.9/80.4  |  76.0/74.5  | 81.3/80.9 | 73.4/73.0 | 85.2/81.9 | 77.9/74.7 | 79.2/78.3 | 76.7/75.9 | 
|  DSVT(Voxel-TS) | 2       |  82.3/80.8  |  76.6/75.1  | 81.4/81.0 | 73.5/73.1 | 85.4/82.2 | 78.4/75.3 | 80.2/79.3 | 77.8/76.9 |

##### 3-Sweeps
|  Model  |  #Sweeps | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|
|  DSVT(Pillar) | 3       |  81.9/80.5  |  76.2/74.8  | 81.2/80.8 | 73.3/72.9 | 85.0/82.0 | 78.0/75.0 | 79.6/78.8 | 77.2/76.4 |
|  DSVT(Voxel) | 3       |  82.1/80.8  |  76.3/75.0  |  81.5/81.1 | 73.6/73.2 | 85.3/82.4 | 78.2/75.4 | 79.6/78.8 | 77.2/76.4 |
|  DSVT(Pillar-TS) | 3       |  82.5/81.0  |  76.7/75.4  | 81.8/81.3 | 74.0/73.6 | 85.6/82.6 | 78.5/75.6 | 80.1/79.2 | 77.7/76.9 |
|  DSVT(Voxel-TS) | 3       |  82.6/81.2  |  76.8/75.5  | 81.8/81.4 | 74.0/73.6 | 85.8/82.9 | 78.8/75.9 | 80.1/79.2 | 77.7/76.9 |

##### 4-Sweeps
|  Model  |  #Sweeps | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|
|  DSVT(Pillar) | 4       |  82.5/81.0  |  76.7/75.3  |  81.7/81.2 | 73.8/73.4 | 85.4/82.3 | 78.5/75.5 | 80.3/79.4 | 77.9/77.1 |
|  DSVT(Voxel) | 4       |  82.6/81.3  |  76.9/75.6  | 81.8/81.4 | 74.1/73.6 | 85.6/82.8 | 78.6/75.9 | 80.4/79.6 | 78.1/77.3 |
|  DSVT(Pillar-TS) | 4       |  82.9/81.5  |  77.3/75.9  | 82.1/81.6 | 74.4/74.0 | 85.8/82.8 | 79.0/76.1 | 80.9/80.0 | 78.6/77.7 |
|  DSVT(Voxel-TS) | 4       |  83.1/81.7  |  77.5/76.2  | 82.1/81.6 | 74.5/74.1 | 86.0/83.2 | 79.1/76.4 | 81.1/80.3 | 78.8/78.0 |


#### Inference Speed
We present a comparison with other state-of-the-art methods on both inference speed and performance accuracy. **After being deployed by NVIDIA TensorRT, our model can achieve a real-time running speed (27Hz).** 

![Speed](assets/Figure1_arxiv.png)

|  Model  |  Latency |  mAP_L2  | mAPH_L2 | 
|---------|---------|---------|--------|
|  Centerpoint-Pillar | 35ms       |  74.3   |  72.1  |
|  Centerpoint-Voxel | 40ms       |  76.0   |  74.5  |
|  PV-RCNN++(center) | 113ms       |  76.7   |  75.4  |
|  DSVT(Pillar) | 67ms       |  77.3   |  75.9  |  
|  DSVT(Voxel) | 97ms       |  74.8   |  72.8  |
|  DSVT(Pillar+TensorRt) | 37ms       |  76.6   |  75.1  |  



## Citation
Please consider citing our work as follows if it is helpful.


## Acknowledgments
This project is based on the following codebases.
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [SST](https://github.com/tusen-ai/SST)