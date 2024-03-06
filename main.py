'''
Author: PlumpDolphin
Date: March 7, 2024

Description: 
    Provides functions for augmenting AI image training sets with the purpose of preventing over-fitting
    in domains where sample sizes are inherently small, such as medical imaging data.

License:
    The code in this file is licensed under the
    Revised 3-Clause BSD License.
    For details, see https://opensource.org/licenses/BSD-3-Clause
'''



import os, os.path
import random

import numpy as np
from PIL import Image, ImageFilter





class AugmentedImage():
    ''' Wraps the PIL image class providing simple, builder-style transformations '''
    def __init__(self, image):
        self.img = image

    @classmethod
    def open(cls, path):
        ''' Creates an Augmented Image from a file path '''
        return cls(Image.open(path))


    def warp(self, amplitude, frequency):
        ''' Applies sinusoidal distortions in both the X and Y dimensions '''
        # Convert image to numpy array
        img_array = np.array(self.img)

        # Get image dimensions
        width, height = self.img.size

        # Create mesh grid for coordinates
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Apply sine wave distortion to x-coordinates
        distorted_x_coords = x_coords + amplitude * np.sin(2 * np.pi * frequency * y_coords / height)

        # Apply sine wave distortion to y-coordinates
        distorted_y_coords = y_coords + amplitude * np.sin(2 * np.pi * frequency * x_coords / width)

        # Clip distorted coordinates to image boundaries
        distorted_x_coords = np.clip(distorted_x_coords, 0, width - 1).astype(int)
        distorted_y_coords = np.clip(distorted_y_coords, 0, height - 1).astype(int)

        # Create distorted image
        distorted_image = np.zeros_like(img_array)
        for y in range(height):
            for x in range(width):
                distorted_image[y, x] = img_array[distorted_y_coords[y, x], distorted_x_coords[y, x]]

        # Convert distorted image array back to PIL Image
        distorted_image_pil = Image.fromarray(distorted_image)

        return AugmentedImage(distorted_image_pil)

    def mirror(self):
        ''' Flips the image horizontally '''
        return AugmentedImage(self.img.transpose(Image.FLIP_LEFT_RIGHT))

    def flip(self):
        ''' Flips the image vertically '''
        return AugmentedImage(self.img.transpose(Image.FLIP_TOP_BOTTOM))

    def translate(self, x, y):
        ''' Moves the image in the X and Y axes '''
        transform_matrix = (1, 0, x, 0, 1, y)
        return AugmentedImage(self.img.transform(self.img.size, Image.AFFINE, transform_matrix))

    def rotate(self, angle):
        ''' Rotates the image '''
        return AugmentedImage(self.img.rotate(angle, resample=Image.BICUBIC, expand=1))

    def blur(self, pixels):
        ''' Blurs the image using a Box blur filter '''
        return AugmentedImage(self.img.filter(ImageFilter.BoxBlur(pixels)))
    
    def gaussian(self, pixels):
        ''' Blurs the image using a Gaussian blur filter '''
        return AugmentedImage(self.img.filter(ImageFilter.GaussianBlur(pixels)))

    def contrast(self, factor):
        ''' Modifies the image contrast by a given factor (0 to 2) '''
        def f(c):
            return 128 + factor * (c - 128)
        return AugmentedImage(self.img.point(f))
    
    def brightness(self, factor):
        ''' Modifies the image brightness by a given factor (0 to inf) '''
        def f(c):
            return factor * c
        return AugmentedImage(self.img.point(f))

    def fireflies(self, factor, brightness):
        ''' Converts random pixel to a gray a given brightness, the amount of pixels is a factor of the image (0 to 1) '''
        def f(c):
            return brightness if random.random() < factor else c
        return AugmentedImage(self.img.point(f))

    def show(self):
        ''' Shows the image in an external window '''
        # This function only exists to prevent the need of typing `<AugmentedImage>.img.show()`
        self.img.show()

    def save(self, path):
        ''' Saves the image to a file path '''
        # This function only exists to prevent the need of typing `<AugmentedImage>.img.save()`
        self.img.save( path )





def generate_variants(aug_img, count):
    ''' Generates randomized variants from a given image, the number of variants returned is equal to count '''

    def rand(r):
        diff = abs(r[0] - r[1])
        return (random.random() * diff) + min(r)

    # Define minimum and maximum values for randomization
    ranges = {
        'blur':      (0, .5), # Pixels
        'rotate':    (-4, 4), # Degrees
        'translate': (-2, 2), # Pixels

        'contrast':   (1, 1.2),   # Factor
        'brightness': (0.9, 1.1), # Factor

        'warp_amp':  (.5, 1), # Pixels
        'warp_freq': (1, 5),  # Pixels
    }

    # Memoization to prevent 
    variants = {}
    
    for i in range(count):
        values = {k: rand(v) for k, v in ranges.items()}

        # Generate unique key (tuple) from random values
        key = tuple(values.values())

        if key not in variants:
            # Generate variant
            variants[key] = aug_img.translate(values['translate'], values['translate']) \
                                   .rotate(values['rotate']) \
                                   .warp(values['warp_amp'], values['warp_freq']) \
                                   .gaussian(values['blur']) \
                                   .brightness(values['brightness']) \
                                   .contrast(values['contrast'])
        else:
            # Variant already used, retry iteration
            i -= 1
            continue

    return variants.values()





if __name__ == '__main__':
    # Specify image extensions
    img_ext = ['png', 'jpg']

    # Source folder
    source_dir = 'Train'
    export_dir = 'Export'

    # Variant count
    variant_count = 5



    # Open up dataset image
    for root, _, files in os.walk(source_dir):
        for file in files:

            # Skip if extension is not a permitted image extension
            if file.split('.')[-1] not in img_ext:
                continue

            # Get source path for image
            path = (os.path.join(root, file))
            path_rel = os.path.relpath(path, source_dir)

            # Load image
            im = AugmentedImage.open(path)

            # Generate variant images
            variants = generate_variants(im, variant_count)

            # Create files from variants
            for i, v in enumerate(variants):
                # Generate path to export to
                path_export = os.path.join(export_dir, path_rel)
                
                # Make sure folders exist
                os.makedirs(os.path.dirname(path_export), exist_ok = True)

                # Parse name elements
                args = path_export.split('.')
                name = '.'.join(args[:-1])
                extension = args[-1]

                # Save variant to file
                v.save( f'{name}_{i}.{extension}' )

                # Save mirrored variant to file
                v.mirror().save( f'{name}_{i}m.{extension}' )
    