# Segmentation

# Requirements

# Usage  
## Help
> python3 main.py --help  

This wil show the usage details as shown below  

>usage: main.py [-h] [-sd {True,False}] [-imp IMAGE_MASK_PATH]  
>               [-od OUTPUT_DIRECTORY] [-pp PATCHES_PATH] [-t {True,False}]  
>               [-m {fcn,unet,deep_unet,pspnet,dsac}]  
  
>Building Segmentation  
  
>optional arguments:  
>  -h, --help            show this help message and exit   
>  -sd {True,False}, --split_dataset {True,False}  
>                        Dataset Preparation  
>  -imp IMAGE_MASK_PATH, --image_mask_path IMAGE_MASK_PATH  
>                        Original Image and Mask Path  
>  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY  
>                        Path for daving all the patches  
>  -pp PATCHES_PATH, --patches_path PATCHES_PATH  
>                        Image and Mask patches path for training  
>  -t {True,False}, --train {True,False}  
>                        Train a model  
>  -m {fcn,unet,deep_unet,pspnet,dsac}, --model {fcn,unet,deep_unet,pspnet,dsac}  
>                        Select a model for training  
## Split Dataset  
>python3 main.py -sd True -imp ../../Dataset/Inria_Dataset/train/ -od ../../Dataset/Inria_Patches/

## Train a Model  
>python3 main.py -pp ../../Dataset/Inria_Patches/inria_dataset_256/train/ -t True -m unet

# Papers
Dataset Link: https://project.inria.fr/aerialimagelabeling/
Fully Convolutional Network Paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
U-Net Paper: https://arxiv.org/pdf/1505.04597.pdf
Deep UNet or Residual UNet Paper: https://arxiv.org/pdf/1711.10684.pdf
PSPNet Paper: https://arxiv.org/pdf/1612.01105.pdf
DSAC Paper: https://arxiv.org/pdf/1803.06329.pdf

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
