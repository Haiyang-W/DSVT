# DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets
	
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-pedestrian)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-pedestrian?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-cyclist)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-cyclist?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-vehicle)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle?p=embracing-single-stride-3d-object-detector) -->

This is the official implementation of: [DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets](www.google.com). Our DSVT achieves state-of-the-art performance on large-scale Waymo Open Dataset with real-time inference speed (27Hz).

![Pipeline](assets/Figure3_sc.png)

## NEWS
- [23-01-17] DSVT is released on [arXiv](www.google.com).


## Main results
### 3D Object Detection (on Waymo validation)
#### One Stage Model
|  Model  |  #Sweeps | mAP_L1 | mAPH_L1 | mAP_L2  | mAPH_L2 | 
|---------|---------|--------|--------|---------|--------|
|  DSVT(Pillar) | 1       |  79.5  |  77.1  |  73.2   |  71.0  |
|  DSVT(Pillar) | 2       |  81.4  |  79.8  |  75.4   |  73.9  |
|  DSVT(Pillar) | 3       |  81.9  |  80.5  |  76.2   |  74.8  |
|  DSVT(Pillar) | 4       |  82.5  |  81.0  |  76.7   |  75.3  |  
|  DSVT(Voxel) | 1       |  80.3  |  78.2  |  74.0   |  72.1  |
|  DSVT(Voxel) | 2       |  81.9  |  80.4  |  76.0   |  74.6  |  
|  DSVT(Voxel) | 3       |  82.1  |  80.8  |  76.3   |  75.0  |  
|  DSVT(Voxel) | 4       |  82.6  |  81.3  |  76.9   |  75.6  |

#### Two Stage Model
**We provide a two-stage version of DSVT with [CT3D](https://github.com/hlsheng1/CT3D).**
|  Model  |  #Sweeps | mAP_L1 | mAPH_L1 | mAP_L2  | mAPH_L2 | 
|---------|---------|--------|--------|---------|--------|
|  DSVT(Pillar-TS) | 1       |  80.6  |  78.2  |  74.3   |  72.1  |
|  DSVT(Pillar-TS) | 2       |  81.9  |  80.4  |  76.0   |  74.5  |
|  DSVT(Pillar-TS) | 3       |  82.5  |  81.0  |  76.7   |  75.4  |
|  DSVT(Pillar-TS) | 4       |  82.9  |  81.5  |  77.3   |  75.9  |  
|  DSVT(Voxel-TS) | 1       |  81.1  |  78.9  |  74.8   |  72.8  |
|  DSVT(Voxel-TS) | 2       |  82.3  |  80.8  |  76.6   |  75.1  |  
|  DSVT(Voxel-TS) | 3       |  82.6  |  81.2  |  76.8   |  75.5  |  
|  DSVT(Voxel-TS) | 4       |  83.1  |  81.7  |  77.5   |  76.2  |

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