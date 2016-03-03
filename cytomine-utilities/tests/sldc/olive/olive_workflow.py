import cv2
import numpy as np
from PIL import ImageDraw
from PIL.Image import fromarray

from shapely.geometry import box
from sldc import Segmenter, DispatchingRule, PolygonClassifier, DispatcherClassifier, SLDCWorkflow
from image import NumpyTileBuilder
from pizza_workflow import ObjectType


def toTriple(n):
    base = 256
    return n if n == 1 else (n % base, (n // base) % base, (n // (base ** 2)) % base)


def base256(t):
    return t[0] + 256 * t[1] + 256 * 256 * t[2]


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


class OliveSegmenter(Segmenter):
    def segment(self, image):
        w, h, _ = image.shape
        encoding_mat = np.zeros((w, h), "int32")
        encoding_mat += image[:, :, 0]
        encoding_mat += 256 * image[:, :, 1]
        encoding_mat += 256 * 256 * image[:, :, 2]

        rs_encoding_mat = np.reshape(encoding_mat, (w * h,))
        values, counts = np.unique(rs_encoding_mat, return_counts=True)
        sorted_counts = np.argsort(counts)
        segmented = rs_encoding_mat
        segmented[segmented == values[sorted_counts[-1]]] = 0
        segmented[segmented == values[sorted_counts[-2]]] = 0
        segmented[segmented != 0] = 255
        segmented = segmented.astype("uint8")
        return segmented.reshape((h, w))


class OliveRule(DispatchingRule):
    def evaluate(self, polygon):
        return 1250 <= polygon.area < 3750


class OliveClassifier(PolygonClassifier):
    def predict(self, image, polygon):
        np_image = image.np_image
        dominant = dominant_color(np_image, polygon)

        if dominant == (0, 0, 255):
            return ObjectType.BLUE
        elif dominant == (0, 255, 0):
            return ObjectType.GREEN


class OliveDispatcherClassifier(DispatcherClassifier):
    def __init__(self):
        DispatcherClassifier.__init__(self, [OliveRule()], [OliveClassifier()])


class OliveWorkflow(SLDCWorkflow):
    def __init__(self):
        SLDCWorkflow.__init__(self, OliveSegmenter(), OliveDispatcherClassifier(), NumpyTileBuilder())


