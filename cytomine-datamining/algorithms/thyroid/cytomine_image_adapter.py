# -*- coding: utf-8 -*-
import cStringIO

from shapely.geometry import Polygon, MultiPolygon, box

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

from sldc.image import Image, Tile
from PIL import Image as PILImage


def to_lower_left(polygon)




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


class CytomineTile(Tile):

    def get_numpy_repr(self):
        _get_crop(self._parent.cytomine, self._parent.image_instance, self._tile_box())

    def _tile_box(self):
        return box(self.offset_x, self.offset_y, self.offset_x + self.width, self.offset_y + self.height)

