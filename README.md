# Segmentation
Implementation of Fully Convolutional Network, U-Net, Deep Residual U-Net, Pyramid Scene Parsing Network and Deep Structured Active Contour.
## Papers  
Fully Convolutional Network Paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf  
U-Net Paper: https://arxiv.org/pdf/1505.04597.pdf  
Deep UNet or Residual UNet Paper: https://arxiv.org/pdf/1711.10684.pdf  
PSPNet Paper: https://arxiv.org/pdf/1612.01105.pdf  
DSAC Paper: https://arxiv.org/pdf/1803.06329.pdf  

# Dataset  
Inria Building Dataset: https://project.inria.fr/aerialimagelabeling/  
ISPRS Dataset: http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html  
SpaceNet Challenge Dataset: https://spacenetchallenge.github.io/  

# Requirements  
> OpenCV  
> Matplotlib  
> Scikit Images  
> Keras  
> Tensorflow  

# Usage  
## Help
> python3 main.py --help  

## Split Dataset  
>python3 main.py -sd True -imp ../../Dataset/Inria_Dataset/train/ -od ../../Dataset/Inria_Patches/

## Train a Model  
>python3 main.py -pp ../../Dataset/Inria_Patches/inria_dataset_256/train/ -t True -m unet



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
