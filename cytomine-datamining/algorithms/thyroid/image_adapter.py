# -*- coding: utf-8 -*-
import cStringIO
import os

import numpy as np
from PIL.Image import fromarray
from shapely.affinity import translate

from sldc.image import Image, Tile, TileBuilder, ImageWindow
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
    # TODO change in the client
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
        raise NotImplementedError("Disabled due to the too heavy size of the images")

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
        self._np_image = np.asarray(_get_crop(self._cytomine, self.base_image.image_instance, self._tile_box()))

    @property
    def np_image(self):
        return self._np_image

    def _tile_box(self):
        offset_x, offset_y = self.abs_offset
        return box(offset_x, offset_y, offset_x + self.width, offset_y + self.height)


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
        offset, width, height = Image.polygon_box(polygon)
        key = TileCache._cache_key(image, offset[0], offset[1], width, height)
        if key in self._cache:
            return self._cache[key]
        else:
            tile = self._tile_builder.build(image, offset, width, height)
            # TODO re-enable caching (disabled because of memory consumption)
            # self._cache[key] = tile
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
        if isinstance(image, ImageWindow):
            image = image.base_image
        filename = "{}_{}_{}_{}_{}.png".format(image.image_instance.id, tile.offset_x,
                                               tile.offset_y, tile.width, tile.height)
        return os.path.join(base_path, filename)

    @staticmethod
    def _cache_key(image, offset_x, offset_y, width, height):
        """Given tile extraction parameters, create a unique identifier for the tile to use a key of the cache)
        The offset part of the key is relative to the base image, not the tile parent image
        Parameters
        ----------
        image: Image
        offset_x: int
            Offset x of the tile in its parent image
        offset_y: int
            Offset y of the tile in its parent image
        width: int
        height: int

        Return
        ------
        key: tuple
            A unique tuple identifying the tile
        """
        if isinstance(image, ImageWindow):  # use the base image as reference image for coordinates
            offset_x, offset_y = image.abs_offset_x + offset_x, image.abs_offset_y + offset_y
            image = image.base_image
        return image.image_instance.id, offset_x, offset_y, width, height


class CytomineMaskedTile(CytomineTile):
    """
    A tile from a cytomine slide on which is applied a mask represented by a polygon
    """
    def __init__(self, cytomine, parent, offset, width, height, polygon, tile_identifier=None):
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
        polygon: Polygon
            The polygon representing the mask over the image. The polygon coordinate should be relative to the parent
            image top-left pixel
        tile_identifier: int, optional (default: None)
            A integer identifier that identifies uniquely the tile among a set of tiles

        Notes
        -----
        The coordinates origin is the leftmost pixel at the top of the slide
        """
        CytomineTile.__init__(self, cytomine, parent, offset, width, height, tile_identifier=tile_identifier)
        self._np_image = alpha_rasterize(self._np_image, translate(polygon, -offset[0], -offset[1]))


class CytomineMaskedWindow(ImageWindow):
    """
    A cytomine slide on which is applied an alpha mask with a polygon
    """

    def __init__(self, parent, offset, width, height, polygon):
        """Construct CytomineMaskedSlide objects

        Parameters
        ----------
        parent: CytomineSlide
            The parent slide
        offset: (int, int)
            The x and y coordinates of the pixel at the origin point of the slide in the parent image.
            Coordinates order is the following : (x, y).
        width: int
            The width of the tile
        height: int
            The height of the tile
        polygon: Polygon
            The polygon representing the mask in the parent image coordinate system
        """
        ImageWindow.__init__(self, parent, offset, width, height)
        self._polygon = polygon

    @property
    def polygon_mask(self):
        return self._polygon

    @property
    def np_image(self):
        raise NotImplementedError("Disabled due to the too heavy size of the images.")

    @property
    def channels(self):
        if self.parent.channels == 1 or self.parent.channels == 3:
            return self.parent.channels + 1
        else:
            return self.parent.channels

    @staticmethod
    def from_window(window, polygon):
        """Build a CytomineMaskedWindow from another window and a polygon
        Parameters
        ----------
        window: ImageWindow
            The other window
        polygon: Polygon
            The polygon representing the mask in the parent image coordinate system

        Returns
        -------
        masked_window: CytomineMaskedWindow
            The cytomine masked window
        """
        return CytomineMaskedWindow(window.parent, window.offset, window.width, window.height, polygon)


class CytomineMaskedTileBuilder(TileBuilder):
    def __init__(self, cytomine):
        """Construct CytomineMaskedTileBuilder objects

        Parameters
        ----------
        cytomine: cytomine.Cytomine
            The initialized cytomine client
        """
        self._cytomine = cytomine

    def build(self, image, offset, width, height):
        """ Build method
        Parameters
        ----------
        image: CytomineMaskedWindow
            The parent image from which is constructed the tile
        offset: tuple (int, int)
            Offset of the tile in the parent image
        width: int
            Width of the tile
        height: int
            Height of the tile
        """
        return CytomineMaskedTile(self._cytomine, image, offset, width, height, image.polygon_mask)
