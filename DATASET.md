AgroCare Dataset
Dataset Source

AgroCare uses the PlantVillage Dataset, specifically the raw/color image set, for crop disease recognition model development.

The dataset was obtained from the original PlantVillage Dataset repository and prepared locally for use with the AgroCare training pipeline.

Dataset Structure

The color dataset is organized into class-specific folders:

Each folder represents a crop-disease or healthy-crop class.
Images are stored directly inside their corresponding class folder.
The dataset contains 38 classes.
Supported image formats include .jpg, .jpeg, .png, .bmp, and .webp.
Dataset Verification

The collected dataset was checked before use in the AgroCare project.

| Verification | Result |
| Classes | 38 |
| Total images | 54,305 |
| Unreadable images | 0 |
| OpenCV image-read check | Passed |

All 54,305 images were successfully read using OpenCV's image loading functionality.

Training Preparation

The AgroCare training pipeline reads the dataset by class folder and supports a maximum number of images per class using the --max-per-class option.

For the current training configuration:

--max-per-class 300

This limits the number of images loaded from larger classes while retaining the class-folder structure.

Local Dataset Location

The dataset is stored locally and is not included in the GitHub repository because of its large size.

Example local path:

F:\AgroCare_Dataset

The training script accepts the dataset location through the --dataset command-line argument, so the local path can be changed when running the project on another machine.

Purpose

The dataset provides labeled crop images for training and evaluating the machine-learning models used in AgroCare's crop disease diagnosis system.

Verification Status

Dataset collection and verification completed.
