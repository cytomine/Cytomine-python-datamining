import numpy as np
from PIL.ImageDraw import ImageDraw
from PIL.Image import fromarray


def mk_gray_img(w,h,level=0):
    return np.ones((w, h)).astype("uint8") * level


def draw_poly(image, poly, fill=255, edge=255):
    pil = fromarray(image)
    drawer = ImageDraw(pil)
    drawer.polygon(poly.boundary.coords, fill=fill, outline=edge)
    return np.asarray(pil)


def draw_circle(image, center, radius, fill=255, edge=255):
    from shapely.geometry import Point
    polygon = Point(center[0], center[1])
    polygon = polygon.buffer(radius)
    return draw_poly(image, polygon, fill=fill, edge=edge), polygon
