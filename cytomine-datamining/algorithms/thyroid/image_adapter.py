# -*- coding: utf-8 -*-
import cStringIO
import os

import math
import numpy as np
from PIL.Image import fromarray
from shapely.affinity import translate

from sldc.image import Image, Tile, TileBuilder
from PIL import Image as PILImage
from helpers.utilities.datatype.polygon import bounds
from shapely.geometry import Polygon, MultiPolygon, box
from helpers.datamining.rasterizer import alpha_rasterize

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def _get_crop(cytomine, image_inst, geometry):
    """
    Download the crop corresponding to bounds on the given image instance
    from cytomine

    Parameters
    ----------
    cytomine : :class:`Cytomine`
        The cilent holding the communication
    image_inst : :class:`ImageInstance` or image instance id (int)
        The image on which to extract crop
    geometry : :class:`shapely.Polygon` or :class:`Bounds`
        The geometry of the crop. /!\ the geometries are assumed to be
        in image coordinate (at zoom 0)
    """
    if isinstance(geometry, Polygon) or isinstance(geometry, MultiPolygon):
        bounds_ = bounds(geometry)
    else:
        bounds_ = geometry
    url = image_inst.get_crop_url(bounds_)
    #TODO change in the client
    url = cytomine._Cytomine__protocol + cytomine._Cytomine__host + cytomine._Cytomine__base_path + url
    resp, content = cytomine.fetch_url(url)
    if resp.status != 200:
        raise IOError("Couldn't fetch the crop for image {} and bounds {} from server (status : {}).".format(image_inst.id, bounds, resp.status))
    tmp = cStringIO.StringIO(content)
    return PILImage.open(tmp)


class CytomineSlide(Image):
    """
    A slide from a cytomine project
    """

    def __init__(self, cytomine, id_img_instance):
        """Construct CytomineSlide objects

        Parameters
        ----------
        cytomine: cytomine.Cytomine
            The cytomine client
        id_img_instance: int
            The id of the image instance
        """
        self._cytomine = cytomine
        self._img_instance = self._cytomine.get_image_instance(id_img_instance)

    @property
    def image_instance(self):
        return self._img_instance

    @property
    def cytomine(self):
        return self._cytomine

    @property
    def np_image(self):
        raise NotImplementedError("Disabled due to the too heavey size of the images")

    @property
    def width(self):
        return self._img_instance.width

    @property
    def height(self):
        return self._img_instance.height

    @property
    def channels(self):
        return 3

    def __str__(self):
        return "CytomineSlide (#{}) ({} x {})".format(self._img_instance.id, self.width, self.height)


class CytomineTile(Tile):
    """
    A tile from a cytomine slide
    """
    def __init__(self, cytomine, parent, offset, width, height, tile_identifier=None):
        """Constructor for CytomineTile objects

        Parameters
        ----------
        cytomine: cytomine.Cytomine
            An initialized instance of the cytomine client
        parent: Image
            The image from which is extracted the tile
        offset: (int, int)
            The x and y coordinates of the pixel at the origin point of the slide in the parent image.
            Coordinates order is the following : (x, y).
        width: int
            The width of the tile
        height: int
            The height of the tile
        tile_identifier: int, optional (default: None)
            A integer identifier that identifies uniquely the tile among a set of tiles

        Notes
        -----
        The coordinates origin is the leftmost pixel at the top of the slide
        """
        Tile.__init__(self, parent, offset, width, height, tile_identifier=tile_identifier)
        self._cytomine = cytomine
        self._np_image = np.asarray(_get_crop(self._cytomine, self._parent.image_instance, self._tile_box()))

    @property
    def np_image(self):
        return self._np_image

    def _tile_box(self):
        return box(self.offset_x, self.offset_y, self.offset_x + self.width, self.offset_y + self.height)


class CytomineTileBuilder(TileBuilder):
    """
    A builder for CytomineTile objects
    """

    def __init__(self, cytomine):
        """Construct CytomineTileBuilder objects

        Parameters
        ----------
        cytomine: cytomine.Cytomine
            The initialized cytomine client
        """
        self._cytomine = cytomine

    def build(self, image, offset, width, height):
        return CytomineTile(self._cytomine, image, offset, width, height)


class TileCache(object):
    """A class for fetching crops of polygons as Tile objects and caching the fetched image for later retrieval
    """
    def __init__(self, tile_builder):
        self._cache = dict()
        self._tile_builder = tile_builder

    def get_tile(self, image, polygon):
        """Get a tile cropping the given polygon

        Parameters
        ----------
        image: Image
            The image from which the crop must be extracted
        polygon: Polygon
            The polygon that should be cropped

        Returns
        -------
        tile: Tile
            The tile cropping the polygon either fetched from cache or from the server on cache miss.
        """
        minx, miny, maxx, maxy = polygon.bounds
        fminx, fminy = int(math.floor(minx)), int(math.floor(miny))
        cmaxx, cmaxy = int(math.ceil(maxx)), int(math.ceil(maxy))
        offset = (fminx, fminy)
        width = cmaxx - fminx
        height = cmaxy - fminy
        key = TileCache._cache_key(image, offset[0], offset[1], width, height)
        if key in self._cache:
            return self._cache[key]
        else:
            tile = self._tile_builder.build(image, offset, width, height)
            self._cache[key] = tile
            return tile

    def save_tile(self, image, polygon, base_path, alpha=False):
        """Fetch and save in the filesystem a tile cropping the given polygon

        Parameters
        ----------
        image: Image
            The image from which the crop must be extracted
        polygon: Polygon
            The polygon that should be cropped
        base_path: string
            The path of the folder in which the image file should be stored
        alpha: bool (optional, default: False)
            True for applying an alpha mask on the image before saving it, false for storing the image as such
            The alpha mask is shaped like the polygon.
        Returns
        -------
        tile: Tile
            The tile object
        path: string
            The full path where the file was stored
        """
        tile = self.get_tile(image, polygon)
        path = TileCache._tile_path(image, tile, base_path)
        if alpha:
            # translate polygon into tile coordinate system
            minx, miny, _, _ = polygon.bounds
            translated_polygon = translate(polygon, -minx, -miny)
            np_image = alpha_rasterize(tile.np_image, translated_polygon)
        else:
            np_image = tile.np_image
        fromarray(np_image.astype('uint8')).save(path)
        return tile, path

    @staticmethod
    def _tile_path(image, tile, base_path):
        """Return the path where to store the tile

        Parameters
        ----------
        image: Image
            The image object from which the tile was extracted
        tile: Tile
            The tile object containing the image to store
        base_path: string
            The path in which the crop image file should be stored

        Returns
        -------
        path: string
            The path in which to store the image
        """
        filename = "{}_{}_{}_{}_{}.png".format(image.image_instance.id, tile.offset_x,
                                               tile.offset_y, tile.width, tile.height)
        return os.path.join(base_path, filename)

    @staticmethod
    def _cache_key(image, offset_x, offset_y, width, height):
        """Given tile extraction parameters, create a unique identifier for the tile to use a key of the cache)

        Parameters
        ----------
        image: Image
        offset_x: int
        offset_y: int
        width: int
        height: int

        Return
        ------
        key: tuple
            A unique tuple identifying the tile
        """
        return image.image_instance.id, offset_x, offset_y, width, height
