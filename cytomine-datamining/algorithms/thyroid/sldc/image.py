# -*- coding: utf-8 -*-

import math
from abc import ABCMeta, abstractmethod, abstractproperty

from errors import TileExtractionException

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version__ = "0.1"


class Image(object):
    """
    Abstract representation of an (possibly huge) image.
    To actually explore the image and its representation, it is recommended to iterate over tiles extracted
    using a TileTopology.

    Notes
    -----
    To avoid memory leaks the representation of the image should not be stored into memory as a class attribute.
    Alternatively, the loading should be deferred to the call of np_image.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def height(self):
        """Return the height of the image
        Returns
        -------
        height: int
            Height of the image
        """
        pass

    @abstractproperty
    def width(self):
        """Return the width of the image
        Returns
        -------
        width: int
            Width of the image
        """
        pass

    @abstractproperty
    def channels(self):
        """Return the number of channels in the image
        Returns
        -------
        width: int
            Width of the image
        """
        pass

    @abstractproperty
    def np_image(self):
        """Return a numpy representation of the image

        Returns
        -------
        np_image: ndarray
            A numpy representation of the image

        Raises
        ------
        ImageExtractionException: when the image cannot be extracted (and so is its representation)

        Notes
        -----
        When working with very big images, this method may be "disabled" (i.e. raise NotImplementedError) to prevent
        user of the class to fully load them into memory. Tiles should be used instead to explore large images.
        """
        pass

    def window(self, offset, max_width, max_height):
        """Build an image object representing a window of the image

        Parameters
        ----------
        offset: (int, int)
            The (x, y) coordinates of the pixel at the origin point of the window in the parent image
        max_width: int
            The maximum width of the window
        max_height: int
            The maximum height of the window

        Returns
        -------
        window: ImageWindow
            The resulting image window
        """
        # width are bound to the current window size, not the parent one
        width = min(max_width, self.width - offset[0])
        height = min(max_height, self.height - offset[1])
        return ImageWindow(self, offset, width, height)

    def window_from_polygon(self, polygon):
        """Build and return a window being the minimum bounding box around the passed polygon.
        At least a part of the polygon should fit the image

        Parameters
        ----------
        polygon: shapely.geometry.Polygon
            The polygon of which the minimum bounding window should be returned

        Returns
        -------
        window: ImageWindow
            The resulting image window

        Raises
        ------
        IndexError: if the polygon box offset is not inside the image
        """
        offset, width, height = Image.polygon_box(polygon)
        if not self._check_tile_offset(offset):
            raise IndexError("Offset {} is out of the image.".format(offset))
        return self.window(offset, width, height)

    def tile(self, tile_builder, offset, max_width, max_height):
        """Extract a tile from the image

        Parameters
        ----------
        tile_builder: TileBuilder
            A tile builder for constructing the Tile object
        offset: (int, int)
            The (x, y) coordinates of the pixel at the origin point of the tile in the parent image
        max_width: int
            The maximum width of the tile
        max_height: int
            The maximum height of the tile

        Returns
        -------
        tile: Tile
            The extracted tile

        Raises
        ------
        IndexError: if the offset is not inside the image
        TileExtractionException: if the tile cannot be extracted
        """
        if not self._check_tile_offset(offset):
            raise IndexError("Offset {} is out of the image.".format(offset))
        width = min(max_width, self.width - offset[0])
        height = min(max_height, self.height - offset[1])
        return tile_builder.build(self, offset, width, height)

    def tile_from_polygon(self, tile_builder, polygon):
        """Build a tile being the minimum bounding box around the passed polygon

        Parameters
        ----------
        tile_builder: TileBuilder
            The builder for effectively building the tile
        polygon: shapely.geometry.Polygon
            The polygon of which the bounding tile should be returned

        Returns
        -------
        tile: Tile
            The bounding tile

        Raises
        ------
        IndexError: if the offset is not inside the image
        TileExtractionException: if the tile cannot be extracted
        """
        offset, width, height = Image.polygon_box(polygon)
        return self.tile(tile_builder, offset, width, height)

    def tile_iterator(self, builder, max_width=1024, max_height=1024, overlap=0):
        """Build and return a tile iterator that iterates over the image

        Parameters
        ----------
        builder: TileBuilder
            The builder to user for actually constructing the tiles while iterating over the image
        max_width: int (optional, default: 1024)
            The maximum width of the tiles to build
        max_height: int (optional, default: 1024)
            The maximum height of the tiles to build
        overlap: int (optional, default: 0)
            The overlapping between tiles

        Returns
        -------
        iterator: TileTopologyIterator
            An iterator that iterates over a tile topology of the image
        """
        topology = TileTopology(self, max_width=max_width, max_height=max_height, overlap=overlap)
        return TileTopologyIterator(builder, topology)

    def tile_topology(self, max_width=1024, max_height=1024, overlap=0):
        """Builds a tile topology for this the image

        Parameters
        ----------
        max_width: int (optional, default: 1024)
            The maximum width of the tiles to build
        max_height: int (optional, default: 1024)
            The maximum height of the tiles to build
        overlap: int (optional, default: 0)
            The overlapping between tiles

        Returns
        -------
        topology: TileTopology
            The image tile topology
        """
        return TileTopology(self, max_width=max_width, max_height=max_height, overlap=overlap)

    def _check_tile_offset(self, offset):
        """Check whether the given tile offset belongs to the image

        Parameters
        ----------
        offset: (int, int)
            The (x, y) coordinates of the pixel at the origin point of the tile in the parent image

        Returns
        -------
        valid: bool
            True if the offset is valid, False otherwise
        """
        return 0 <= offset[0] < self.width and 0 <= offset[1] < self.height

    @staticmethod
    def polygon_box(polygon):
        """From a shapely polygon, return the information about the polygon bounding box.
        These information are offset (x, y), width and height.

        Parameters
        ----------
        polygon: shapely.geometry.Polygon
            The polygon of which the bounding box should be computed

        Returns
        -------
        offset: tuple (int, int)
            The offset of the polygon bounding box
        width: int
            The bounding box width
        height
            The bounding box heigth
        """
        minx, miny, maxx, maxy = polygon.bounds
        fminx, fminy = int(math.floor(minx)), int(math.floor(miny))
        cmaxx, cmaxy = int(math.ceil(maxx)), int(math.ceil(maxy))
        offset = (fminx, fminy)
        width = cmaxx - fminx
        height = cmaxy - fminy
        return offset, width, height


class ImageWindow(Image):
    """An image window represents a window in another image
    """
    def __init__(self, parent, offset, width, height):
        """Constructor for ImageWindow objects

        Parameters
        ----------
        parent: Image
            The image from which is extracted the window
        offset: (int, int)
            The x and y coordinates of the pixel at the origin point of the slide in the parent image.
            Coordinates order is the following : (x, y).
        width: int
            The width of the image
        height: int
            The height of the image

        Notes
        -----
        The coordinates origin is the leftmost pixel at the top of the slide
        """
        self._parent = parent
        self._offset = offset
        self._width = width
        self._height = height

    @property
    def offset_x(self):
        """Return the x offset of the tile
        Returns
        -------
        offset_x: int
            X offset of the tile
        """
        return self._offset[0]

    @property
    def offset_y(self):
        """Return the y offset of the tile
        Returns
        -------
        offset_y: int
            Y offset of the tile
        """
        return self._offset[1]

    @property
    def offset(self):
        """Return the offset of the tile
        Returns
        -------
        offset: (int, int)
            The (x, y) offset of the tile
        """
        return self._offset

    @property
    def abs_offset_x(self):
        """Return the x offset of the window relatively to the base image.
        Returns
        -------
        abs_offset_x: int
            The absolute x offset of the window
        """
        return self.offset_x + self.parent.abs_offset_x if isinstance(self.parent, ImageWindow) else self.offset_x

    @property
    def abs_offset_y(self):
        """Return the y offset of the window relatively to the base image.
        Returns
        -------
        abs_offset_y: int
            The absolute y offset of the window
        """
        return self.offset_y + self.parent.abs_offset_y if isinstance(self.parent, ImageWindow) else self.offset_y

    @property
    def abs_offset(self):
        """Return the offset of the window relatively to the base image.

        Returns
        -------
        abs_offset: tuple (int, int)
            The absolute offset of the window
        """
        return self.abs_offset_x, self.abs_offset_y

    @property
    def channels(self):
        return self._parent.channels

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def base_image(self):
        """Return the base Image object from which the window was extracted. If the parent image is a Window then, the
        base image is fetched recursively from it.
        Returns
        -------
        base_image: Image
            The base image
        """
        return self._parent.base_image if isinstance(self._parent, ImageWindow) else self._parent

    @property
    def parent(self):
        return self._parent

    @property
    def np_image(self):
        """Return a numpy representation of the image

        Returns
        -------
        np_image: ndarray
            A numpy representation of the image

        Raises
        ------
        ImageExtractionException: when the image cannot be extracted (and so is its representation)

        Notes
        -----
        Actually accesses to the parent image np_image property !!
        """
        minx = self.offset_x
        miny = self.offset_y
        maxx = self.offset_x + self.width
        maxy = self.offset_y + self.height
        return self._parent.np_image[miny:maxy, minx:maxx]


class Tile(ImageWindow):
    """Abstract representation of an image's tile. A tile is a "small" window of an image.
    This class should be used along with a TileTopology to explore huge images.

    Notes
    -----
    To avoid memory leaks the representation of the tile should not be stored into memory as a class attribute.
    Alternatively, the loading should be deferred to the call of np_image.
    """
    __metaclass__ = ABCMeta

    def __init__(self, parent, offset, width, height, tile_identifier=None):
        """Constructor for Tile objects

        Parameters
        ----------
        parent: Image
            The image from which is extracted the tile
        offset: (int, int)
            The x and y coordinates of the pixel at the origin point of the slide in the parent image.
            Coordinates order is the following : (x, y).
        width: int
            The width of the tile
        height: int
            The height of the tile
        tile_identifier: int (optional, default: None)
            A integer identifier that identifies uniquely the tile among a set of tiles

        Notes
        -----
        The coordinates origin is the leftmost pixel at the top of the slide
        """
        ImageWindow.__init__(self, parent, offset, width, height)
        self._identifier = tile_identifier

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        self._identifier = value

    @property
    def np_image(self):
        """Return a numpy representation of the tile

        Returns
        -------
        np_image: ndarray
            A numpy representation of the tile

        Raises
        ------
        TileExtractionException: when the tile cannot be extracted

        Notes
        -----
        This property should be used to actually load the tile representation into memory.
        Default implementation uses parent image np_image property!!!
        """
        try:
            return super(Tile, self).np_image
        except Exception as e:
            raise TileExtractionException("Cannot extract the tile : {}".format(e.message))

    def __str__(self):
        return "({},{},{},{})".format(self.width, self.height, self.offset_x, self.offset_y)


class TileBuilder(object):
    """An interface to be implemented by any class which builds specific tiles
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self, image, offset, width, height):
        """Build and return a tile object

        Parameters
        ----------
        image: Image
            The image from which the tile is extracted
        offset: (int, int)
            The (x, y) coordinates of the pixel at the origin point of the slide in the parent image
        width: int
            The exact width of the tile
            /!\ The resulting tile must not overflow the image
        height: int
            The exact height of the tile
            /!\ The resulting tile must not overflow the image

        Returns
        -------
        tile: Tile
            The built tile object

        Errors
        ------
        TypeError: when the reference image is not set
        TileExtractionException: when the tile cannot be extracted

        Notes
        -----
        The coordinates origin is the leftmost pixel at the top of the slide
        """
        pass

    def build_from_polygon(self, image, polygon):
        """Build a tile boxing the given polygon in the image

        Parameters
        ----------
        image: Image
            The image from which the tile should be extracted
        polygon: Polygon
            The polygon of which the bounding tile should be returned

        Returns
        -------
        tile: Tile
            The bounding tile

        Errors
        ------
        TypeError: when the reference image is not set
        TileExtractionImage: when the tile cannot be extracted
        """
        return image.tile_from_polygon(self, polygon)


class DefaultTileBuilder(TileBuilder):
    """A tile builder which builds Tile objects"""
    def build(self, image, offset, width, height):
        return Tile(image, offset, width, height)


class TileTopologyIterator(object):
    """An object to iterate over an image tile per tile as defined by a TileTopology
    """

    def __init__(self, builder, tile_topology, silent_fail=False):
        """Constructor for TilesIterator objects

        Parameters
        ----------
        builder: TileBuilder
            The builder to user for actually constructing the tiles while iterating over the image
        tile_topology: TileTopology
            The topology on which must iterate the iterator
        silent_fail: bool (optional, default: False)
            True for silently skipping tiles that cannot be constructed, otherwise, an error is raised

        Notes
        -----
        Some tiles might actually be smaller than (max_width, max_height) on the edges of the image
        """
        self._builder = builder
        self._topology = tile_topology
        self._silent_fail = silent_fail

    def __iter__(self):
        for tile_identifier in range(1, self._topology.tile_count + 1):
            try:
                yield self._topology.tile(tile_identifier, self._builder)
            except TileExtractionException, e:
                if not self._silent_fail:
                    raise e


class TileTopology(object):
    """A tile topology is an object storing information about a set of tiles of given dimensions and overlapping that
    fully covers an image. These parameters (image width and height, tile width and height and overlap) fully define
    the topology.

    The tile topology defines a bijection between the tiles and the integers. The tile generated by the topology are
    therefore associated identifiers matching this bijection. Given 'v', the number of vertical tiles and 'h', the
    number of horizontal tiles :
        - tile 1 is the top-left tile
        - tile h is the top-right tile
        - tile (h * (v - 1) + 1) is the bottom-left tile
        - tile (v * h) is the bottom-right tile

    The implementation is aimed at being memory efficient by computing everything on-the-fly for avoiding storing heavy
    data structures. Also, methods that compute topology properties (such as finding the neighbour tiles, finding the
    total number of tiles...) are implemented as efficiently as possible (in general, O(1)).
    """

    def __init__(self, image, max_width=1024, max_height=1024, overlap=0):
        """Constructor for TileTopology objects

        Parameters
        ----------
        image: Image
            The image for which the topology must be built
        max_width: int (optional, default: 1024)
            The maximum width of the tiles
        max_height: int (optional, default: 1024)
            The maximum height of the tiles
        overlap: int (optional, default: 0)
            The number of pixels of overlapping between neighbouring tiles

        Notes
        -----
        Some tiles might actually be smaller than (max_width, max_height) on the edges of the image.
        The same goes if the image's dimensions are smaller than (max_width, max_height).
        """
        self._image = image
        self._max_width = max_width
        self._max_height = max_height
        self._overlap = overlap

    def tile(self, identifier, tile_builder):
        """Extract and build the tile corresponding to the given identifier.

        Parameters
        ----------
        identifier: int
            A tile identifier
        tile_builder : TileBuilder
            A builder for building a Tile object from the extracted tile

        Returns
        -------
        tile: Tile
            The tile object
        """
        self._check_identifier(identifier)
        offset = self.tile_offset(identifier)
        tile = self._image.tile(tile_builder, offset, self._max_width, self._max_height)
        tile.identifier = identifier
        return tile

    def tile_offset(self, identifier):
        """Return the offset of the given tile

        Parameters
        ----------
        identifier: int
            A tile identifier

        Returns
        -------
        offset: (int, int)
            The (x, y) coordinates of the pixel at the origin point of the tile in the parent image
        """
        self._check_identifier(identifier)
        row, col = self._tile_coord(identifier)
        offset_x = col * (self._max_width - self._overlap)
        offset_y = row * (self._max_height - self._overlap)
        return offset_x, offset_y

    def tile_neighbours(self, identifier):
        """Return the identifiers of the tiles round a given tile

        Parameters
        ----------
        identifier: int
            The tile identifier

        Returns
        -------
        neighbours: tuple
            A four-element tuple containing the identifiers of the neighbours tiles. If a neighbour tile doesn't exist
            None is returned instead of the identifier. The tuple is structured as follows (top, bottom, left, right).
        """
        self._check_identifier(identifier)
        tile_count = self.tile_count
        h_tile_count = self.tile_horizontal_count
        tile_row = self._tile_coord(identifier)[0]
        top = (identifier - h_tile_count) if (identifier - h_tile_count) >= 1 else None
        bottom = (identifier + h_tile_count) if (identifier + h_tile_count) <= tile_count else None
        left = identifier - 1 if identifier > 1 else None
        # check whether the tile is on an edge. In this case no left tile.
        if left is not None and self._tile_coord(left)[0] != tile_row:
            left = None
        right = identifier + 1 if identifier < tile_count else None
        # check whether the tile is on an edge. In this case no left tile.
        if right is not None and self._tile_coord(right)[0] != tile_row:
            right = None
        return top, bottom, left, right

    def _check_identifier(self, identifier):
        """Check whether the identifiers is valid for the given topology.

        Parameters
        ----------
        identifier: int
            A tile identifier

        Raises
        ------
        ValueError: if the identifier is out of range
        """
        tile_count = self.tile_count
        if identifier > tile_count:
            raise ValueError("The value {} is an invalid tile identifier. Maximum identifier is {}.".format(identifier,
                                                                                                            tile_count))

    def _tile_coord(self, identifier):
        """Compute the row and column of the tile in the tile grid

        Parameters
        ----------
        identifier: int
            The tile identifier (starting at 1)

        Returns
        -------
        tile_coord: (int, int)
            Coordinates of the tile in the tile grid/topology. Coordinates tuple is structured as (row, col).

        Notes
        -----
        Rows and columns identifiers start at 0
        """
        id_start_at_0 = identifier - 1
        return (id_start_at_0 // self.tile_horizontal_count), (id_start_at_0 % self.tile_horizontal_count)

    @property
    def tile_count(self):
        """Compute the total number of tiles in the given topology.

        Returns
        -------
        tile_count: int
            The number of tiles
        """
        return self.tile_vertical_count * self.tile_horizontal_count

    @property
    def tile_vertical_count(self):
        """Compute the number of tiles that fits on the vertical dimension of the image
        Returns
        -------
        tile_count: int
            The number of tiles that fits vertically on the image
        """
        return TileTopology.tile_count_1d(self._image.height, self._max_height, self._overlap)

    @property
    def tile_horizontal_count(self):
        """Compute the number of tiles that fits on the horizontal dimension of the image
        Returns
        -------
        tile_count: int
            The number of tiles that fits horizontally on the image
        """
        return TileTopology.tile_count_1d(self._image.width, self._max_width, self._overlap)

    @staticmethod
    def tile_count_1d(length, tile_length, overlap=0):
        """Compute the number of tiles of length 'tile_length' that can be generated over one dimension of the an image
        of length 'length' and with an overlap of 'overlap'.

        Parameters
        ----------
        length: int
            The number of pixels of one dimension of the image
        tile_length: int
            The number of pixels of a tile for the same dimension
        overlap: int
            The number of pixels that overlap between the tiles

        Returns
        -------
        tile_count: int
            The number of tile that fits in the image dimension given the tile_width and overlap constraints
        """
        return 1 if length < tile_length else int(math.ceil(float(length - overlap) / (tile_length - overlap)))
