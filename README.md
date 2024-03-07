# AI Image Data Augmentation

This project provides tools for modifying image data for AI training purposes. It includes various transformations and augmentations to increase the diversity of the dataset and prevent overfitting.

## Features

- Translation: Move images in the XY plane.
- Rotation: Rotate images by a specified angle.
- Warping: Apply a wavy distortion to images.
- Gaussian Blur: Add Gaussian blur to images.
- Brightness Adjustment: Adjust the brightness of images.
- Contrast Adjustment: Adjust the contrast of images.

### Installation

To use this project, you'll need Python installed on your system. You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Usage

To use the project, follow these steps:

1.  Place the images you want to modify in the "Train" folder, alongside the `main.py` file.
2.  Run the `main.py` script. It will generate 5 variants of each image and save two versions: one original variant and one mirrored. This effectively increases the training set size by 1100%.
3.  The number of variants, the amount of augmentation, etc can be tuned in the `__main__` section at the bottom of the `main.py` script.

### Word of Caution

The data augmentation process generates distorted variants of images, which can help prevent overfitting on small training sets. While it doesn't guarantee high-quality images, it expands the training set artificially to improve training results when real-world data is limited.