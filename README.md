# Building Detection from High Resolution Satellite Images

<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/results/sample.png" width="400"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/results/sample_result.png" width="400">
</p>

Implementation of Fully Convolutional Network, U-Net, Deep Residual U-Net, Pyramid Scene Parsing Network and Deep Structured Active Contour.

## Papers  
Fully Convolutional Network Paper: [Link](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)  
SegNet Paper: [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7803544)  
U-Net Paper: [Link](https://arxiv.org/pdf/1505.04597.pdf)  
Deep UNet or Residual UNet Paper: [Link](https://arxiv.org/pdf/1711.10684.pdf)  
PSPNet Paper: [Link](https://arxiv.org/pdf/1612.01105.pdf)  

## Architectures
<p align="center">
<img src="https://github.com/arghadeep25/Segmentation/blob/master/architecture/fcn_archi.png" width="300">   <img src="https://github.com/arghadeep25/Segmentation/blob/master/architecture/segnet_archi_mod.png" width="300">   <img src="https://github.com/arghadeep25/Segmentation/blob/master/architecture/unet_archi_mod.png" width="300"> 
<img src="https://github.com/arghadeep25/Segmentation/blob/master/architecture/deepunet_archi_mod.png" width="300">   <img src="https://github.com/arghadeep25/Segmentation/blob/master/architecture/pspnet_archi_mod.png" width="300">
</p>

# Dataset  
The datasets used in this project can be downloaded from the following links 
### [Inria Building Dataset](https://project.inria.fr/aerialimagelabeling/):  
<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/images/austin3_45.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/gt/austin3_45.png" width="120">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/images/chicago1_21.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/gt/chicago1_21.png" width="120">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/images/vienna2_63.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/gt/vienna2_63.png" width="120"> 
</p>  
  

### [ISPRS Dataset](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html):  
<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/images/top_potsdam_3_11_RGB.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/masks/top_potsdam_3_11_RGB.png" width="120">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/gt/top_potsdam_3_11_RGB.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/images/top_potsdam_7_7_RGB.png" width="120">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/masks/top_potsdam_7_7_RGB.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/gt/top_potsdam_7_7_RGB.png" width="120"> 
</p>   

### [SpaceNet Challenge Dataset](https://spacenetchallenge.github.io/):  
<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/spacenet/RGB-PanSharpen_AOI_3_Paris_img418.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/spacenet/RGB-PanSharpen_AOI_3_Paris_img418_gt.png" width="120">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/spacenet/RGB-PanSharpen_AOI_4_Shanghai_img945.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/spacenet/RGB-PanSharpen_AOI_4_Shanghai_img945_gt.png" width="120">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/spacenet/RGB-PanSharpen_AOI_5_Khartoum_img5.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/spacenet/RGB-PanSharpen_AOI_5_Khartoum_img5_gt.png" width="120"> 
</p>   

  
### [CrowdAI Mapping Challenge Dataset](https://www.crowdai.org/challenges/mapping-challenge):  
<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/crowdai/image/1_img.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/crowdai/anno/ann_1.png" width="120">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/crowdai/gt/11.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/crowdai/image/5_img.png" width="120">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/crowdai/anno/ann_5.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/crowdai/gt/55.png" width="120"> 
</p> 

### [WHU Building Dataset](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html):  
#### WHU East Asia Dataset:  
<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/whu_east_asia/whu_east_asia.png" width="800">
</p>  

<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/whu_east_asia/whu_east_asia_patches.png" width="800">
</p>  

#### WHU New Zealand:  
<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/nzlnd/whu_nz.png" width="800">
</p>  

#### WHU Global Cities:  
<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/whu_sample/whu_global_cities.png" width="800">
</p>  

# Requirements  
> [OpenCV-Python](https://pypi.org/project/opencv-python/)  
> [Matplotlib](https://pypi.org/project/matplotlib/)  
> [Numpy](https://pypi.org/project/numpy/)  
> [Scikit Images](https://scikit-image.org/docs/dev/install.html)  
> [Keras](https://pypi.org/project/Keras/)  
> [Tensorflow](https://www.tensorflow.org/install/pip)  
> [GDAL](https://pypi.org/project/GDAL/)  
> [PyCOCOTools](https://pypi.org/project/pycocotools/)  

# Usage  
## Help  
```console  
python3 main.py --help  
```  

## Split Dataset  
```console  
python3 main.py -sd True -imp ../../Dataset/Inria_Dataset/train/ -od ../../Dataset/Inria_Patches/
```  

## Train a Model  
```console  
python3 main.py -pp ../../Dataset/Inria_Patches/inria_dataset_256/train/ -t True -m unet
```  
### Training Plots
<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/training_plots/fcn_training.png" title="FCN" width="400"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/training_plots/segnet_training.png" title="SegNet" width="400"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/training_plots/unet_training.png" title="U-Net" width="400"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/training_plots/deepunet_training.png" title="Deep U-Net" width="400"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/training_plots/pspnet_training.png" title="PSPNet" width="400">
</p>
