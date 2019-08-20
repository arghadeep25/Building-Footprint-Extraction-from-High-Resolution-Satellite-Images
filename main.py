"""
Name: Main
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. Prepare Dataset
                - Inria Dataset
                - ISPRS Dataset
                - CrowdAI Building Dataset
                - SpaceNet Challenge Dataset
             2. Train
                - FCN
                - UNet
                - Deep UNet
                - PSPNet
                - DSAC

             3. Test
"""
import argparse
from train.trainUNet import TrainUNet
from train.trainFCN import TrainFCN
from train.trainPSPNet import TrainPSPNet
from train.trainDeepUNet import TrainDeepUNet
from train.trainSegNet import TrainSegNet
from utils.generate import InriaDataGenerator


def cmd_line_parser():
    parser = argparse.ArgumentParser(description="Building Segmentation")

    # Argument for converting
    # Image of (5000 x 5000) --> 400 patches of (250 x 250)
    parser.add_argument('-sd', '--split_dataset',
                        help='Dataset Preparation',
                        choices=[True,
                                 False],
                        type=bool,
                        default=False)

    # Argument for loading image and mask from the folder
    # Note: Check datastructure for training
    parser.add_argument('-imp', '--image_mask_path',
                        help='Original Image and Mask Path',
                        type=str,
                        default=None)

    # Argument for saving the patches path
    parser.add_argument('-od', '--output_directory',
                        help='Path for daving all the patches',
                        type=str,
                        default='../../Datasets/updated_dataset/')

    # Argument for loading patches for training
    parser.add_argument('-pp', '--patches_path',
                        help='Image and Mask patches path for training',
                        type=str,
                        default=None)

    # Argument for training
    parser.add_argument('-t', '--train',
                        help='Train a model',
                        choices=[True,
                                 False],
                        type=bool,
                        default=False)

    # Argument for selecting a model for training
    parser.add_argument('-m', '--model',
                        help='Select a model for training',
                        choices=['fcn',
                                 'unet',
                                 'deep_unet',
                                 'segnet',
                                 'pspnet',
                                 'dsac'],
                        type=str,
                        default=None)

    return parser.parse_args()


def main():
    args = cmd_line_parser()

    # Dataset Preparation (from image to patch)
    # Note: To prepare the dataset pass the arguments
    #  1. -sd True
    #  2. -imp [ORIGINAL_MASK_IMAGE_PATH]
    if args.split_dataset is True:
        if args.image_mask_path is None:
            print('Please specify the path...')
        data_generator = InriaDataGenerator(data_path=args.image_mask_path,
                                            output_path=args.output_directory)
        data_generator.split_all_images()

    # Training
    # Note: For training pass the arguments
    #  1. -t True
    #  2. -m [MODEL_NAME]
    #  3. -pp [PATCHES_PATH]
    if args.train is True:
        if args.model == 'fcn':
            train_fcn = TrainFCN(train_path=args.patches_path)
            train_fcn.train()
        elif args.model == 'unet':
            train_unet = TrainUNet(train_path=args.patches_path)
            train_unet.train()
        elif args.model == 'deep_unet':
            train_deep_unet = TrainDeepUNet(train_path=args.patches_path)
            train_deep_unet.train()
        elif args.model == 'segnet':
            train_segnet = TrainSegNet(train_path=args.patches_path)
            train_segnet.train()
        elif args.model == 'pspnet':
            train_pspnet = TrainPSPNet(train_path=args.patches_path)
            train_pspnet.train()
        elif args.model == 'dsac':
            return
        else:
            print('Please select a valid model to continue...')


if __name__ == '__main__':
    main()
