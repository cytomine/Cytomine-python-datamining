# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of LiÃ¨ge, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.

Note
----
Polygons are assumed to be represented in (x, y) coordinate system
with the origin in the upper left corner. x is the column of a pixel and
y is the row (increasing downwards)
North-South are on row axis. With North being row < 0 (relative to the tile)
West-East are on the column axis. With West being column < 0 (relative
    to the tile)
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of LiÃ¨ge, Belgium"
__version__ = '0.1'

from shapely.ops import cascaded_union
from shapely.geometry import JOIN_STYLE

from ..utilities.iterator2d import Iterable2D, RowOrderIterator

class Graph(object):
    def __init__(self):
        self.nodes = []
        self.edges = {}

    def add_node(self, value):
        self.nodes.append(value)
        return len(self.nodes) - 1

    def add_edge(self, source, destination):
        ls = self.edges.get(source, [])
        if len(ls) == 0:
            self.edges[source] = ls
        ls.append(destination)

    def connex_components(self):
        visited = [False]*len(self.nodes)
        components = []
        stack = []
        current_node = 0
        for node in range(len(self.nodes)):
            current_comp = []

            stack.append(node)
            while len(stack) > 0:
                current_node = stack.pop()
                if visited[current_node]:
                    continue
                visited[current_node] = True
                current_comp.append(current_node)
                map(stack.append, self.edges.get(current_node, []))
            if len(current_comp) > 0:
                components.append(current_comp)
        return components


    def __getitem__(self, node_index):
        return self.nodes[node_index]



def add_edge_on_dist(graph, border_gids, neighbor_gids, dist):

    # Iterate through border geometries
    for b_geom in border_gids:
        for nb_geom in neighbor_gids:
            # Iterate through neighbor border geometries
            if  graph[b_geom].distance(graph[nb_geom]) <= dist:
                graph.add_edge(b_geom, nb_geom)



def merge_geom_in_graph(graph, dilation_dist, discretized=True):
    result = []
    join = JOIN_STYLE.mitre if discretized else JOIN_STYLE.round
    components = graph.connex_components()
    for component in components:
        if len(component) == 1:
            result.append(graph[component[0]])
        else:
            polygons = [graph[x].buffer(dilation_dist, join_style=join)
                        for x in component]
            result.append(cascaded_union(polygons).buffer(-dilation_dist,
                                                          join_style=join))
    return result




class TileFrame:

    def __init__(self, row_offset, col_offset, width, height,
                 boundary_thickness=0, discretized=True):
        self.row_offset = row_offset
        self.col_offset = col_offset
        self.width = width
        self.height = height
        self.boundary_thickness = boundary_thickness
        self.discretized = discretized
        self._north_tile_frame = None
        self._south_tile_frame = None
        self._west_tile_frame = None
        self._east_tile_frame = None
        self._northeast_tile_frame = None
        self._southwest_tile_frame = None
        self._northwest_tile_frame = None
        self._southeast_tile_frame = None
        self._north_overlap = 0
        self._south_overlap = 0
        self._east_overlap = 0
        self._west_overlap = 0
        self.entries = {}
        self.north = []
        self.northeast = []
        self.east = []
        self.southeast = []
        self.south = []
        self.southwest = []
        self.west = []
        self.northwest = []

    def __repr__(self):
        return str((self.row_offset, self.col_offset, self.height, self.width))

    def __str__(self):
        return str((self.row_offset, self.col_offset, self.height, self.width))

    def set_north_tile_frame(self, tile_frame):
        self._north_tile_frame = tile_frame
        overlap = (tile_frame.row_offset + tile_frame.height) - self.row_offset
        self._north_overlap = overlap
        tile_frame._south_tile_frame = self
        tile_frame._south_overlap = overlap

    def set_east_tile_frame(self, tile_frame):
        self._east_tile_frame = tile_frame
        overlap = (self.col_offset + self.width) - tile_frame.col_offset
        self._east_overlap = overlap
        tile_frame._west_tile_frame = self
        tile_frame._west_overlap = overlap

    def set_northeast_tile_frame(self, tile_frame):
        self._northeast_tile_frame = tile_frame
        tile_frame._southwest_tile_frame = self

    def set_northwest_tile_frame(self, tile_frame):
        self._northwest_tile_frame = tile_frame
        tile_frame._southeast_tile_frame = self

    def get_east_tile_frame(self):
        return self._east_tile_frame

    def add_geometry(self, geometry, gid):
        self.entries[gid] = geometry

    def pack(self):
        """
        Must be called to compute which polygons touch
        which border
        """
        for gid, geometry in self.entries.iteritems():
            mincol, minrow, maxcol, maxrow = geometry.bounds
            touches_north = False
            touches_east = False
            touches_west = False
            touches_south = False


            if self.discretized:
                maxrow += 1
                maxcol += 1

            if minrow <= self.row_offset + self.boundary_thickness + self._north_overlap:
                touches_north = True
            if mincol <= self.col_offset + self.boundary_thickness + self._west_overlap:
                touches_west = True
            if maxrow >= (self.row_offset + self.height) - (self.boundary_thickness + self._south_overlap):
                touches_south = True
            if maxcol >= (self.col_offset + self.width) - (self.boundary_thickness + self._east_overlap):
                touches_east = True


            if touches_north:
                self.north.append(gid)
                if touches_east:
                    self.northeast.append(gid)
                if touches_west:
                    self.northwest.append(gid)
            if touches_east:
                self.east.append(gid)
            if touches_south:
                self.south.append(gid)
                if touches_east:
                    self.southeast.append(gid)
                if touches_west:
                    self.southwest.append(gid)
            if touches_west:
                self.west.append(gid)





#TODO :  geometric simplification ?
class Merger(object):
    """
    ======
    Merger
    ======
    A :class:`Merger` merges polygons from neighboring tiles based on the
    instance policy

    Constructor parameters
    ----------------------
    boundary_thickness : float (default : 0)
        Distance from the actual boundary at which an object is considered as
        touching the boundary
    """

    def __init__(self, boundary_thickness=0):
        self.__boundary_thickness = boundary_thickness

    def get_boundary_thickness(self):
        return self.__boundary_thickness

    def store(self, tile, polygons):
        """
        Store the polygons of the given tile for later merging with
        neighboring tiles

        Paremeters
        ----------
        tile : :class:`Tile`
            The tile where the polygons belongs
        polygons : sequence of :class:`shapely.Polygon`
            The polygons to store
        """
        pass

    def merge(self):
        """
        Merge the polygons of neighboring tiles according to the actual
        instance policy

        Return
        ------
        polygons : a sequence of :class:`shapely.Polygon`
            All the polygons
        """
        pass


class GeneralMerger(Merger):
    """
    =============
    GeneralMerger
    =============
    A :class:`GeneralMerger` is a :class:`Merger` which can accept tiles in
    any order and recover the topology afterwards.
    """
    def __init__(self):
        raise NotImplementedError("Not yet implemented")


class RowOrderMerger(Merger, Iterable2D):
    """
    ==============
    RowOrderMerger
    ==============
    :class:`RowOrderMerger` accepts tiles in row order for efficiency reasons.
    Be sure to give the tiles in that fashion or the merging phase will not
    work properly

    Constructor parameters
    ----------------------
    boundary_thickness : float (default : 0)
        Distance from the actual boundary at which an object is considered as
        touching the boundary
    """

    def __init__(self, boundary_thickness=0, discretized=True):
        Merger.__init__(self, boundary_thickness)
        self._discretized = discretized
        self.tiles = []
        self._row = -1
        self._col = -1
        self.graph = Graph()

    def store(self, tile, polygons):
        if tile.col_offset == 0:
            self.tiles.append([])
            self._row += 1
            self._col = -1

        thickness = self.get_boundary_thickness()
        col_offset = tile.col_offset
        row_offset = tile.row_offset
        tile_height = tile.get_height()
        tile_width = tile.get_width()
        tile_frame = TileFrame(row_offset, col_offset,
                               tile_width, tile_height,
                               thickness, self._discretized)
        self.tiles[self._row].append(tile_frame)
        self._col += 1



        for polygon in polygons:
            gid = self.graph.add_node(polygon)
            tile_frame.add_geometry(polygon, gid)

    def pack(self):
        """
        Builds the topology
        """
        # Adding the topology
        nb_rows = len(self.tiles)
        nb_columns = 1
        for row in xrange(nb_rows):
            nb_columns = len(self.tiles[row])
            for col in xrange(nb_columns):
                if row > 0:
                    self.tiles[row][col].set_north_tile_frame(self.tiles[row-1][col])
                    if col > 0:
                        self.tiles[row][col].set_northwest_tile_frame(self.tiles[row-1][col-1])
                    if col < nb_columns - 1:
                        self.tiles[row][col].set_northeast_tile_frame(self.tiles[row-1][col+1])
                if col < nb_columns - 1:
                    self.tiles[row][col].set_east_tile_frame(self.tiles[row][col+1])
        self.nb_rows = nb_rows
        self.nb_cols = nb_columns
        # pack the tiles
        for tile in self:
            tile.pack()

    def _do_merge(self, buffer_size):
        """
        Perform the actual merging with the current topology
        """
        for tile in self:
            graph = self.graph
            #East
            border_geometries = tile.east
            neighbor_tile = tile._east_tile_frame
            if neighbor_tile:
                add_edge_on_dist(graph, border_geometries,
                                 neighbor_tile.west, buffer_size)

            #South East
            border_geometries = tile.southeast
            neighbor_tile = tile._southeast_tile_frame
            if neighbor_tile:
                add_edge_on_dist(graph, border_geometries,
                                 neighbor_tile.northwest, buffer_size)

            #South
            border_geometries = tile.south
            neighbor_tile = tile._south_tile_frame
            if neighbor_tile:
                add_edge_on_dist(graph, border_geometries,
                                 neighbor_tile.north, buffer_size)
            #South West
            border_geometries = tile.southwest
            neighbor_tile = tile._southwest_tile_frame
            if neighbor_tile:
                add_edge_on_dist(graph, border_geometries,
                                 neighbor_tile.northeast, buffer_size)

    def merge(self):
        #Building the topology
        self.pack()
        #Merging polygons
        buffer_size = self.get_boundary_thickness()
        if self._discretized:
            # +1,45 accounts for distance between both boundaries (in diagonal)
            buffer_size += 1.45
        self._do_merge(buffer_size)
        #Yielding the merged polygons
        return merge_geom_in_graph(self.graph, buffer_size, self._discretized)

    #-----------------------Iterable2D-------------------#
    def __iter__(self):
        return RowOrderIterator(self)

    def get_element(self, row, column):
        return self.tiles[row][column]

    def get_height(self):
        return self.nb_rows

    def get_width(self):
        return self.nb_cols


class DoNothingMerger(Merger):

    def __init__(self, boundary_thickness=0):
        Merger.__init__(self, boundary_thickness)
        self.polygons = []


    def get_boundary_thickness(self):
        return self.__boundary_thickness

    def store(self, tile, polygons):
        self.polygons += polygons

    def merge(self):
        return self.polygons



class MergerAbstractFactory:
    """
    =====================
    MergerAbstractFactory
    =====================
    Abstract base class for :class:`Merger` factory

    Constructor parameters
    ----------------------
    boundary_thickness : int (default : 0)
        The boundary thickness. That is, the distance at which an object
        is considered as touching the boundary. See :class:`Merger` for
        more information
    """

    def __init__(self, boundary_thickness=0):
        self.__boundary_thickness = boundary_thickness

    def get_boundary_thickness(self):
        """
        Return
        ------
        boundary_thickness : int
            The boundary thickness
        """
        return self.__boundary_thickness

    def create_merger(self):
        """
        Abstract factory method

        Return
        ------
        merger : :class:`Merger`
            A merger with appropriate boundary thickness
        """
        pass


class MergerFactory(MergerAbstractFactory):
    """
    =============
    MergerFactory
    =============
    A simple factory which creates :class:`Merger` from a given class

    Constructor parameters
    ----------------------
    boundary_thickness : int (default : 0)
        The boundary thickness. That is, the distance at which an object
        is considered as touching the boundary. See :class:`Merger` for
        more information
    class_to_build : class
        The class of object to build

    C1(boundary_thickness) <=> MergerFactory(boundary_thickness, C1).create_merger()
    """

    def __init__(self, boundary_thickness, class_to_build):
        MergerAbstractFactory.__init__(self, boundary_thickness)
        self._class_to_build = class_to_build

    def create_merger(self):
        return self._class_to_build(self.get_boundary_thickness())
