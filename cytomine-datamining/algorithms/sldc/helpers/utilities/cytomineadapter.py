# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of Liège, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of Liège, Belgium"
__version__ = '0.1'

try:
    import Image
except:
    from PIL import Image
import cStringIO

from shapely.geometry.polygon import Polygon

from cytomine.models import ImageInstance

from .source import TileStream, TileStreamBuilder, ImageLoader
from .datatype import Tile, Bounds, bounds, PILConverter
from .reader import CytomineReader

#import urllib, cStringIO


#TODO manage connexion errors
#TODO make something for annotations and their tiles
#TODO allow for storage on disk (maybe not for the reader class)
class CytomineTileStream(TileStream):
    """
    ==================
    CytomineTileStream
    ==================
    Adaptator to the :class:`TileStream` interface for the Cytomine client

    The seed must an instance of :class:`Reader`
    """

    def __init__(self, reader):
        self.reader = reader
        self.has_next = True

    def next(self):
        if not self.has_next:
            raise StopIteration

        self.reader.read()
        image = self.reader.data
        x, y, width, height = self.reader.window_position

        self.has_next = self.reader.next()

        return Tile(image, x, y)

    def get_image_id(self):
        return self.reader.image.image.id



class CytomineTileStreamBuilder(TileStreamBuilder):
    """
    =========================
    CytomineTileStreamBuilder
    =========================
    Adaptator to the :class:`TileStreamBuilder` interface for the Cytomine
    client

    Note
    ----
    1. The seeds must :class:`WholeSlide`.
    2. This do not incur dataflows

    Constructor parameters
    ----------------------
    cytomine_client : :class:`Cytomine`
        The cilent holding the communication
    origin : tupe(x,y) (default : (0,0))
        The origin of the sliding window
    width : int > 0 (default : 4096)
        The width of sliding window
    height : int > 0 (default 4096)
        The height of the sliding window
    zoom : int >= 0 (default : 0)
        The zoom level
    overlap : int >= 0 (default : 0)
        The overlap
    """
    def __init__(self, cytomine_client,
                 origin=(0, 0),
                 width=1024, height=1024,
                 zoom=0,
                 overlap=0):
        self._cytomine_client = cytomine_client
        self._bounds = Bounds(origin[0], origin[1], width, height)
        self._zoom = zoom
        self._overlap = overlap

    def build(self, seed):
        #Creating a reader for img_id
        slide = seed
        reader = CytomineReader(self._cytomine_client,
                                slide,
                                window_position=self._bounds,
                                zoom=self._zoom,
                                overlap=self._overlap)
        return CytomineTileStream(reader)



def _get_crop(cytomine, image_inst, geometry):
    """
    Download the crop corresponding to bounds on the given image instance
    from cytomine

    Parameters
    ----------
    cytomine : :class:`Cytomine`
        The cilent holding the communication
    image_instance : :class:`ImageInstance` or image instance id (int)
        The image on which to extract crop
    geometries : :class:`shapely.Polygon` or :class:`Bounds`
        The geometry of the crop. /!\ the geometries are assumed to be
        in image coordinate (at zoom 0)
    """
    if isinstance(geometry, Polygon):
        bounds_ = bounds(geometry)
    else:
        bounds_ = geometry
    #TODO check bounds
    url = image_inst.get_crop_url(bounds_)
    #TODO change in the client
    url = cytomine._Cytomine__protocol + cytomine._Cytomine__host + cytomine._Cytomine__base_path + url
    #TODO check response
    resp, content = cytomine.fetch_url(url)
    tmp = cStringIO.StringIO(content)
    return Image.open(tmp)


class CytomineCropStream(TileStream):
    """
    ==================
    CytomineCropStream
    ==================
    A :class:`TileStream` which streams crops of polygons
    from ONE image.

    Constructor parameters
    ----------------------
    cytomine_client : :class:`Cytomine`
        The cilent holding the communication
    image_instance : :class:`ImageInstance` or image instance id (int)
        The image on which to extract crop
    polygons : a sequence of :class:`shapely.Polygon`
        The geometry of the crop./!\ the geometries are assumed to be
        in image coordinate (at zoom 0)
    rasterizer : a rasterizer
        if None : only the crop is downloaded
        else : The ROI polygons are rasterized and appended to the tile as
        an extra dimension
    """
    def __init__(self, cytomine_client, image_instance,
                 polygons, rasterizer=None):
        self._cytomine = cytomine_client
        self._img = image_instance
        if not isinstance(image_instance, ImageInstance):
            self._img = self._cytomine.get_image_instance(image_instance)
        self._polygons = polygons
        self._rasterizer = rasterizer
        self._index = 0

    def next(self):
        try:
            bounds_ = bounds(self._polygons[self._index])
            x = bounds_["x"]
            y = bounds_["y"]
            image = _get_crop(self._cytomine, self._img, bounds_)
            if self._rasterizer is None:
                return Tile(image, x, y)
            # Rasterize the polygon
            masked_image = self._rasterizer.alpha_rasterize(image, self._polygons[self._index])
            return Tile(masked_image, x, y)
        except IndexError:
            raise StopIteration
        finally:
            self._index += 1

    def get_image_id(self):
        return self._img.id



class CytomineCropStreamBuilder(TileStreamBuilder):
    """
    =========================
    CytomineCropStreamBuilder
    =========================
    A factory for :class:`CytomineCropStream`.

    Seeds
    -----
    The seed must  be a pair (img, geometries)
    where img is either
    image_instance : :class:`ImageInstance` or image id (int)
        The image on which to extract the crops
    geometries : :class:`shapely.Polygon` or :class:`Bounds`
        The geometry of the crop. /!\ the geometries are assumed to be
        in image coordinate (at zoom 0)

    Constructor parameters
    ----------------------
    cytomine_client : :class:`Cytomine`
        The cilent holding the communication
    rasterizer : concrete instance of :class:`AbstractRasterizer`
    (default : None)
        The rasterizer to use, if needed
    """

    def __init__(self, cytomine_client, rasterizer=None):
        self._cytomine = cytomine_client
        self._rasterizer = rasterizer

    def build(self, seed):
        img_inst, geometries = seed
        return CytomineCropStream(self._cytomine, img_inst, geometries, self._rasterizer)



class CropLoader(ImageLoader):
    """
    ==========
    CropLoader
    ==========
    An :class:`ImageLoader` which extracts crop from Cytomine.
    See also :`ImageBuffer`.

    Seed
    ----
    The seeds are pairs of
    image_instance : :class:`ImageInstance`
        The image on which to extract the crop
    crop_bounds :  :class:`shapely.Polygon` or :class:`Bounds`
        The geometries to download. /!\ the geometries are assumed to be
        in image coordinate (at zoom 0)

    Constructor parameters
    ----------------------
    image_converter : ImageConverter
        A converter to get the appropriate format
    """

    def __init__(self, cytomine_client, image_converter=PILConverter()):
        ImageLoader.__init__(self, image_converter)
        self._cytomine = cytomine_client

    def _load(self, seed):
        image_instance, crop_bounds = seed
        return _get_crop(self._cytomine, image_instance, crop_bounds)
