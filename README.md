# Building Detection from High Resolution Satellite Images using CNN

<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/results/sample.png" width="400"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/results/sample_result.png" width="400">
</p>

Implementation of Fully Convolutional Network, U-Net, Deep Residual U-Net, Pyramid Scene Parsing Network and Deep Structured Active Contour.

## Papers  
Fully Convolutional Network Paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf  
SegNet Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7803544  
U-Net Paper: https://arxiv.org/pdf/1505.04597.pdf  
Deep UNet or Residual UNet Paper: https://arxiv.org/pdf/1711.10684.pdf  
PSPNet Paper: https://arxiv.org/pdf/1612.01105.pdf  

## Architectures
<p align="center">
<img src="https://github.com/arghadeep25/Segmentation/blob/master/architecture/fcn_archi.png" width="300">   <img src="https://github.com/arghadeep25/Segmentation/blob/master/architecture/segnet_archi_mod.png" width="300">   <img src="https://github.com/arghadeep25/Segmentation/blob/master/architecture/unet_archi_mod.png" width="300"> 
<img src="https://github.com/arghadeep25/Segmentation/blob/master/architecture/deepunet_archi_mod.png" width="300">   <img src="https://github.com/arghadeep25/Segmentation/blob/master/architecture/pspnet_archi_mod.png" width="300">
</p>

# Dataset  
The datasets used in this project can be downloaded from the following links 
### [Inria Building Dataset](https://project.inria.fr/aerialimagelabeling/):  
<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/images/austin3_45.png" width="100"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/gt/austin3_45.png" width="100">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/images/chicago1_21.png" width="100"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/gt/chicago1_21.png" width="100">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/images/vienna2_63.png" width="100"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/gt/vienna2_63.png" width="100">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/images/tyrol-w2_12.png" width="100"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/inria/gt/tyrol-w2_12.png" width="100">
</p>  
  

### [ISPRS Dataset](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html):  
<p align="center">
  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/images/top_potsdam_3_11_RGB.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/masks/top_potsdam_3_11_RGB.png" width="120">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/gt/top_potsdam_3_11_RGB.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/images/top_potsdam_7_7_RGB.png" width="120">  <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/masks/top_potsdam_7_7_RGB.png" width="120"> <img src="https://github.com/arghadeep25/Segmentation/blob/master/datasets/isprs/gt/top_potsdam_7_7_RGB.png" width="120"> 
</p>   

### [SpaceNet Challenge Dataset](https://spacenetchallenge.github.io/):  
  
### [CrowdAI Mapping Challenge Dataset](https://www.crowdai.org/challenges/mapping-challenge):  

### [WHU Building Dataset](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html):  



# Requirements  
> OpenCV  
> Matplotlib  
> Scikit Images  
> Keras  
> Tensorflow  

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


# To Do
- [x] Split images into patches 250x250
- [x] Split images into patches 500x500
- [x] Implement FCN
- [x] Implement UNet
- [x] Implement Deep UNet
- [ ] Implement PSPNet
- [ ] Implement DSAC
- [x] Train on small dataset
- [ ] Train on large dataset
- [ ] Test all the trained models
- [ ] Plot graphs on Tensorboard
- [ ] Plot IoU graphs
- [ ] Figure out the limitations
