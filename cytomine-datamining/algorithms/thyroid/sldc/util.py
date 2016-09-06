# -*- coding: utf-8 -*-
import Image
import ImageDraw
import numpy as np
from shapely.geometry.base import BaseMultipartGeometry

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def emplace(src, dest, mapping):
    """Place the values of src into dest at the indexes indicated by the mapping

    Parameters
    ----------
    src: iterable (size: n)
        Elements to emplace into the dest list
    dest: list (size: m)
        The list in which the elements of src must be placed
    mapping: iterable (subtype: int, size: n)
        The indexes of dest where the elements of src must be placed
    """
    for index, value in zip(mapping, src):
        dest[index] = value


def take(src, idx):
    """Generate a list containing the elements of src of which the index is contained in idx

    Parameters
    ----------
    src: list (size: n)
        Source iterable from which elements must be taken
    idx: iterable (subtype: int, range: [0, n[, size: m)
        Indexes iterable

    Returns
    -------
    list: list
        The list of taken elements
    """
    return [src[i] for i in idx]


def batch_split(n_batches, items):
    """Partition the items into a given number of batches of similar sizes (if the number of batches is greater than
    the number of items N in the topology, N batches are returned).

    Parameters
    ----------
    n_batches: int
        The number of batches
    items: iterable
        The elements to split into batches

    Returns
    -------
    batches: iterable (subtype: iterable (subtype: Tile), size: min(n_batches, N))
        The batches of tiles
    """
    item_count = len(items)
    if n_batches >= item_count:
        return [[item] for item in items]
    batches = [[] for _ in range(0, n_batches)]
    current_batch = 0
    bigger_batch_count = item_count % n_batches
    smaller_batch_size = (item_count / n_batches)
    bigger_batch_size = (item_count / n_batches) + 1
    for item in items:
        batches[current_batch].append(item)
        if (current_batch < bigger_batch_count and len(batches[current_batch]) >= bigger_batch_size) \
                or (current_batch >= bigger_batch_count and len(batches[current_batch]) >= smaller_batch_size):
            # check whether the current batch is full and should be changed
            current_batch += 1
    return batches


def has_alpha_channel(image):
    """Check whether the image has an alpha channel

    Parameters
    ----------
    image: ndarray
        The numpy representation of the image

    Returns
    -------
    has_alpha: boolean
        True if the image has an alpha channel, false otherwise
    """
    chan = image.shape
    return len(chan) == 3 and (chan[2] == 2 or chan[2] == 4)


def alpha_rasterize(image, polygon):
    """
    Rasterize the given polygon as an alpha mask of the given image. The
    polygon is assumed to be referenced to the top left pixel of the image.
    If the image has already an alpha mask it is replaced by the polygon mask

    Parameters
    ----------
    image: ndarray
        The numpy representation of the image
    polygon : Polygon
        The polygon to rasterize

    Return
    ------
    rasterized : ndarray
        The image (in numpy format) of the rasterization of the polygon.
        The image should have the same dimension as the bounding box of
        the polygon.
    """
    # destination image
    source = np.asarray(image)

    # extract width, height and number of channels
    chan = source.shape
    if len(chan) == 3:
        height, width, depth = source.shape
    else:
        height, width = source.shape
        depth = 1
        source = source.reshape((height, width, depth))

    # if there is already an alpha mask, replace it
    if has_alpha_channel(image):
        source = source[:, :, 0:depth-1]
    else:
        depth += 1

    # create rasterization mask
    alpha = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(alpha)
    boundary = polygon.boundary
    if isinstance(boundary, BaseMultipartGeometry):  # handle multi-part geometries
        for sub_boundary in boundary.geoms:
            seq_pts = sub_boundary.coords
            draw.polygon(seq_pts, outline=0, fill=255)
    else:
        seq_pts = polygon.boundary.coords
        draw.polygon(seq_pts, outline=0, fill=255)

    # merge mask with images
    rasterized = np.zeros((height, width, depth), dtype=source.dtype)
    rasterized[:, :, 0:depth-1] = source
    rasterized[:, :, depth-1] = alpha
    return rasterized
