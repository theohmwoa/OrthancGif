import numpy as np
from PIL import Image
from tools import set_contrast


class MosaicGenerator:

    numpy_array :np.ndarray
    cols :int
    nb_images :int
    final_width :int
    final_height :int

    def __init__(self, numpy_array: np.ndarray, cols: int, nb_images: int, final_width :int, final_height :int):
        self.numpy_array = numpy_array
        self.cols = cols
        self.nb_images = nb_images
        self.final_width = final_width
        self.final_height = final_height

    def combine_images(self, columns, images: np.array, space=0) -> Image:
        rows = len(images) // columns
        if len(images) % columns:
            rows += 1
        first_image = Image.fromarray(images[0])
        width_max = first_image.width
        height_max = first_image.height
        background_width = width_max*columns + (space*columns)-space
        background_height = height_max*rows + (space*rows)-space
        concatenated_image = Image.new(
            'L', (background_width, background_height))
        x = 0
        y = 0
        for i, image in enumerate(images):
            img = Image.fromarray(image.astype(np.uint16))
            x_offset = int((width_max-img.width)/2)
            y_offset = int((height_max-img.height)/2)
            concatenated_image.paste(img, (x+x_offset, y+y_offset))
            x += width_max + space
            if (i+1) % columns == 0:
                y += height_max + space
                x = 0
        return concatenated_image

    def select_images(self) -> np.ndarray:
        slices = []
        for i in range(self.numpy_array.shape[0]):
            if i % self.nb_images == 0:
                slices.append(self.numpy_array[i])
        return np.stack(slices)

    def createImage(self, output: bytes) -> None :
        images = self.select_images()
        images = set_contrast(images)
        image = self.combine_images(self.cols, images)
        image.thumbnail((self.final_width, self.final_height), Image.BILINEAR)
        image.save(output, format='PNG')