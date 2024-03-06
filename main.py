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
    # Open up dataset image
    im = AugmentedImage.open('Train/Moderate_Demented/moderate_17.jpg')
    im.show()
    input()
    im.fireflies(0.1, 128).blur(0.5).contrast(1.2).show()
    exit()

    variants = generate_variants(im, 5)

    for v in variants:
        v.show()
        v.mirror().show()