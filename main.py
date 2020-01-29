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
                - SegNet
                - UNet
                - Deep UNet
                - PSPNet

             3. Predict
"""
import argparse
from train.train import Train
from prediction.predict import Predict
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

    # Argument for converting test images into patches
    # Image of (5000x5000) -> patches of (256x256)
    parser.add_argument('-std', '--split_test_dataset',
                         help = 'Test Data Preparation',
                         choices=[True,
                                  False],
                         type=bool,
                         default=False)
    # Argument for loading test images
    parser.add_argument('-tedp', '--test_data_path',
                         help = 'Test Data Path',
                         type=str,
                         default=None)
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
                        default=None)

    # Argument for train data path
    parser.add_argument('-trdp', '--train_data_path',
                        help='Image and Mask patches path for training',
                        type=str,
                        default=None)
    # Argument for validation data path
    parser.add_argument('-vdp', '--val_data_path',
                        help='Image and Mask patches path for validation',
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
                                 'pspnet'],
                        type=str,
                        default=None)
    # Prediction
    parser.add_argument('-p', '--predict',
                        help='Prediction',
                        choices=[True,
                                 False],
                        type=bool,
                        default=False)

    # Results path
    parser.add_argument('-op', '--output_path',
                        help='Path for saving results',
                        type=str,
                        default=None)

    return parser.parse_args()


def main():
    args = cmd_line_parser()

    # Train Dataset Preparation (from image to patch)
    if args.split_dataset is True:
        if args.image_mask_path is None:
            print('Please specify the path...')
        data_generator = InriaDataGenerator(data_path=args.image_mask_path,
                                            output_path=args.output_directory,
                                            patch_size = 256)
        data_generator.split_all_images()

    # Test Dataset Preparation (from image to patch) [without ground truth]
    if args.split_test_dataset is True:
        if args.test_data_path is None:
            print('Please specify the path...')
        data_generator = InriaTestDataGenerator(data_path=args.test_data_path,
                                                output_path=args.output_directory,
                                                patch_size=256)
        data_generator.split_all_images()
    # Training
    if args.train is True:
        train_model = Train(train_path=args.train_data_path,
                            validation_path=args.val_data_path,
                            model_name=args.model,
                            patch_size=256,
                            activate_aug=True,
                            pre_trained=False,
                            epochs=1,
                            rotation=90,
                            sigma=0,
                            zoom_range=1,
                            vertical_flip=True,
                            horizontal_flip=True,
                            shear=0.2,
                            brightness=True,
                            add_noise=True,
                            hist_eq=True,)
        train_model.train()

    if args.predict is True:
        predict_model = Predict(model_name=args.model,
                                data_path=args.test_data_path,
                                output_path=args.output_path)
        predict_model.eval()


if __name__ == '__main__':
    main()
