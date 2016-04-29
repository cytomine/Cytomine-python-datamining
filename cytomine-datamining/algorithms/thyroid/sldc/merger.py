# -*- coding: utf-8 -*-

from shapely.geometry import JOIN_STYLE
from shapely.ops import cascaded_union

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"
__contributor__ = "Begon Jean-Michel <jm.begon@gmail.com>"


class Graph(object):
    """
    A class for representing a graph
    """
    def __init__(self):
        self.nodes = []
        self.node2idx = {}
        self.edges = {}

    def add_node(self, value):
        """Add a node to the graph
        :param value:
        :return:
        """
        self.nodes.append(value)
        self.node2idx[value] = len(self.nodes) - 1
        return len(self.nodes) - 1

    def add_edge(self, source, destination):
        """Add an edge to the graph
        :param source:
        :param destination:
        :return:
        """
        ls = self.edges.get(source, [])
        if len(ls) == 0:
            self.edges[source] = ls
        ls.append(destination)

    def connex_components(self):
        """Find the connex components of the graph
        :return:
        """
        visited = [False]*len(self.nodes)
        components = []
        stack = []  # store index of reachable nodes
        for node in self.node2idx.keys():
            current_comp = []
            stack.append(node)
            while len(stack) > 0:
                current_node = stack.pop()
                curr_idx = self.node2idx[current_node]
                if visited[curr_idx]:
                    continue
                visited[curr_idx] = True
                current_comp.append(current_node)
                map(stack.append, self.edges.get(current_node, []))
            if len(current_comp) > 0:
                components.append(current_comp)
        return components

    def __getitem__(self, node_index):
        return self.nodes[node_index]


class Merger(object):
    """
    A class for merging polygons from neighbouring tiles of an image
    """
    def __init__(self, boundary_thickness):
        """Constructor for Merger objects

        Parameters:
        -----------
        boundary_thickness: int
            Distance from the actual boundary at which an object is considered as
            touching the boundary
        """
        self._boundary_thickness = boundary_thickness

    def merge(self, polygons_tiles, tile_topology):
        """Merge the polygons passed in a per-tile fashion according to the tile topology

        Parameters
        ----------
        polygons_tiles: iterable
            An array of tuples. Each tuple contains a tile and its polygons in an array.
            The polygons are of type shapely.geometry.Polygon.
        tile_topology: TileTopology
            The tile topology used to generate the tile passed in polygons_tiles

        Returns
        -------
        polygons: list
            An array of polygons objects containing the merged polygons
        """
        tiles_dict, polygons_dict = Merger._build_dicts(polygons_tiles)
        # no polygons
        if len(polygons_dict) <= 0:
            return []
        # stores the polygons indexes as nodes
        geom_graph = Graph()
        # add polygons
        for index in polygons_dict.keys():
            geom_graph.add_node(index)
        # add edges between polygons that should be merged
        for tile_identifier in tiles_dict.keys():
            # check whether polygons in neighbour tiles must be merged
            neighbour_tiles = tile_topology.tile_neighbours(tile_identifier)
            for neighbour in neighbour_tiles:
                if neighbour is not None:
                    self._register_merge(tiles_dict[tile_identifier], tiles_dict[neighbour], polygons_dict, geom_graph)
        return self._do_merge(geom_graph, polygons_dict)

    def _register_merge(self, polygons1, polygons2, polygons_dict, geom_graph):
        """Compare 2-by-2 the polygons in the two arrays. If they are very close (using thickness_boundary as distance
        threshold), they are registered as polygons to be merged in the geometry graph (the registration is simply
        an edge between the nodes corresponding to the polygons).

        Parameters
        ----------
        polygons1: iterable
            Array of integers containing polygons indexes
        polygons2: iterable
            Array of integers containing polygons indexes
        polygons_dict: dict
            Dictionary mapping polygon identifiers with actual shapely polygons objects
        geom_graph: Graph
            The graph in which must be registered the polygons to be merged
        """
        for poly1 in polygons1:
            for poly2 in polygons2:
                if polygons_dict[poly1].distance(polygons_dict[poly2]) < self._boundary_thickness:
                    geom_graph.add_edge(poly1, poly2)

    def _do_merge(self, geom_graph, polygons_dict):
        """Effectively merges the polygons that were registered to be merged in the geom_graph Graph and return the
        resulting polygons in a list.

        Parameters
        ----------
        polygons_dict: dict
            Dictionary mapping polygon identifiers with actual shapely polygons objects
        geom_graph: Graph
            The graph in which were registered the polygons to be merged

        Returns
        -------
        polygons: list
            An array of polygons objects containing the merged polygons
        """
        components = geom_graph.connex_components()
        dilation_dist = self._boundary_thickness
        join = JOIN_STYLE.mitre
        results = []
        for component in components:
            if len(component) == 1:
                to_append = polygons_dict[component[0]]
            else:
                polygons = [polygons_dict[poly_id].buffer(dilation_dist, join_style=join) for poly_id in component]
                to_append = cascaded_union(polygons).buffer(-dilation_dist, join_style=join)
            results.append(to_append)
        return results

    @classmethod
    def _build_dicts(cls, polygons_tiles):
        """Given a array of tuples (polygons, tile), return dictionaries for executing the merging:

        Parameters
        ----------
        polygons_tiles: iterable
            ...

        Returns
        -------
        polygons_dict: dictionary
            Maps a unique integer identifier with a polygon. All the polygons passed to the functions are given an
            identifier and are stored in this dictionary
        tiles_dict:
            Maps a tile identifier with the an array containing the ids of the polygons located in this tile.
        """
        tiles_dict = dict()
        polygons_dict = dict()
        polygon_cnt = 1
        for tile, polygons in polygons_tiles:
            polygons_ids = []

            for polygon in polygons:
                polygons_dict[polygon_cnt] = polygon
                polygons_ids.append(polygon_cnt)
                polygon_cnt += 1

            tiles_dict[tile.identifier] = polygons_ids

        return tiles_dict, polygons_dict
