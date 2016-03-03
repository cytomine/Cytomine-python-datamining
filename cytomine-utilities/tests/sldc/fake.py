import numpy as np
from sldc import Image, Tile, TileBuilder, ImageWindow


class FakeImage(Image):
    """
    Fake image for testing
    """
    def __init__(self, w, h, c):
        Image.__init__(self)
        self._w = w
        self._h = h
        self._c = c

    @property
    def width(self):
        return self._w

    @property
    def height(self):
        return self._h

    @property
    def channels(self):
        return self._c

    @property
    def np_image(self):
        return np.zeros((self._w,self._h,self._c), "uint8")

    def window(self, offset, max_width, max_height):
        offset_x = offset[0]
        offset_y = offset[1]
        final_offset = (offset_x, offset_y)
        # width are bound to the current window size, not the parent one
        width = min(max_width, self.width - offset[0])
        height = min(max_height, self.height - offset[1])
        return ImageWindow(self, final_offset, width, height)


class FakeTile(Tile):
    """
    Fake tile for testing
    """
    def __init__(self, parent, offset, width, height):
        Tile.__init__(self, parent, offset, width, height)

    def get_numpy_repr(self):
        return np.zeros((self.width, self.height, self.channels))


class FakeTileBuilder(TileBuilder):
    """
    Fake tile builder for testing
    """
    def build(self, image, offset, width, height):
        return FakeTile(image, offset, width, height)
