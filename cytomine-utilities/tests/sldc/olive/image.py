import numpy as np
from sldc import Image, Tile, TileBuilder


class NumpyImage(Image):
    def __init__(self, np_image):
        self._np_image = np_image

    @property
    def height(self):
        return self._np_image.shape[0]

    @property
    def channels(self):
        return self._np_image.shape[2] if len(self._np_image.shape) > 2 else 1

    @property
    def width(self):
        return self._np_image.shape[1]

    @property
    def np_image(self):
        return np.copy(self._np_image)


class NumpyTile(Tile):
    def get_numpy_repr(self):
        image = self._parent.np_image
        row_1, row_n = self.offset_y, self.offset_y + self.height
        col_1, col_n = self.offset_x, self.offset_x + self.width
        return image[row_1:row_n, col_1:col_n]


class NumpyTileBuilder(TileBuilder):
    def build(self, image, offset, width, height):
        return NumpyTile(image, offset, width, height)


