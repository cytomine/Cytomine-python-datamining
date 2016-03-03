import cv2
import numpy as np
from PIL import ImageDraw
from PIL.Image import fromarray

from sldc import Segmenter, DispatchingRule, PolygonClassifier, DispatcherClassifier, SLDCWorkflow
from image import NumpyTileBuilder


def toTriple(n):
    base = 256
    return n if n == 1 else (n % base, (n // base) % base, (n // (base ** 2)) % base)


def dominant_color(image, polygon):
    w, h = image.shape[0:2]
    # create image in base 255
    base256_image = np.zeros((w, h), "int32")
    base256_image += image[:, :, 0]
    base256_image += 256 * image[:, :, 1]
    base256_image += 256 * 256 * image[:, :, 2]
    # draw the polygon mask

    pil_image = fromarray(np.zeros((w, h), "uint8"))
    drawer = ImageDraw.ImageDraw(pil_image)
    drawer.polygon(polygon.boundary.coords, fill=255)
    poly_image = np.asarray(pil_image)

    # erase values not in the polygon
    base256_image[poly_image < 255] = -1
    base256_image = np.reshape(base256_image, (w * h,))
    values, counts = np.unique(base256_image, return_counts=True)
    max_idx = np.argmax(counts)
    curr_value = values[max_idx]
    if curr_value == -1 and len(values) > 1:
        counts[max_idx] = 0
        max_idx = np.argmax(counts)
        curr_value = values[max_idx]
    return toTriple(curr_value)


class PizzaSegmenter(Segmenter):
    def segment(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grey[grey == 255] = 0
        grey[grey > 0] = 255
        return grey


class SmallPizzaRule(DispatchingRule):
    def evaluate(self, polygon):
        return 125000 <= polygon.area < 200000


class BigPizzaRule(DispatchingRule):
    def evaluate(self, polygon):
        return polygon.area >= 200000


class ObjectType(object):
    SMALL_YELLOW = 1
    SMALL_RED = 2
    BIG_YELLOW = 3
    BIG_RED = 4
    GREEN = 5
    BLUE = 6

    @staticmethod
    def str(code):
        if code == ObjectType.SMALL_YELLOW:
            return "SMALL YELLOW"
        elif code == ObjectType.SMALL_RED:
            return "SMALL RED"
        elif code == ObjectType.BIG_YELLOW:
            return "BIG YELLOW"
        elif code == ObjectType.GREEN:
            return "GREEN"
        elif code == ObjectType.BLUE:
            return "BLUE"
        else:
            return "BIG RED"


class BigPizzaClassifier(PolygonClassifier):
    def predict(self, image, polygon):
        np_image = image.np_image
        dominant = dominant_color(np_image, polygon)

        if dominant == (255, 255, 0):
            return ObjectType.BIG_YELLOW
        elif dominant == (255, 0, 0):
            return ObjectType.BIG_RED


class SmallPizzaClassifier(PolygonClassifier):
    def predict(self, image, polygon):
        np_image = image.np_image
        dominant = dominant_color(np_image, polygon)

        if dominant == (255, 255, 0):
            return ObjectType.SMALL_YELLOW
        elif dominant == (255, 0, 0):
            return ObjectType.SMALL_RED


class PizzaDispatcherClassifier(DispatcherClassifier):
    def __init__(self):
        DispatcherClassifier.__init__(self, [SmallPizzaRule(), BigPizzaRule()], [SmallPizzaClassifier(), BigPizzaClassifier()])


class PizzaWorkflow(SLDCWorkflow):
    def __init__(self):
        SLDCWorkflow.__init__(self, PizzaSegmenter(), PizzaDispatcherClassifier(), NumpyTileBuilder(),
                              tile_max_width=512, tile_max_height=512)
