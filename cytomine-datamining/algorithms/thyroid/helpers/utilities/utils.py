# -*- coding: utf-8 -*-

"""
Copyright 2010-2013 University of Liège, Belgium.

This software is provided 'as-is', without any express or implied warranty. 
In no event will the authors be held liable for any damages arising from the use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

__author__          = "Stévens Benjamin <b.stevens@ulg.ac.be>" 
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>", "Rollus Loïc <lrollus@ulg.ac.be"]                
__copyright__       = "Copyright 2010-2013 University of Liège, Belgium"
__version__         = '0.1'

from shapely.geometry.polygon import Polygon
from shapely.wkt import dumps

class Utils_(object):

    def get_geometries(self, components, min_area = None, max_area = None):
        locations = []
        for component in components:
            p = Polygon(component)
            if min_area and max_area:
                #print p.area
                if p.area > min_area and p.area < max_area:
                    locations.append(p.wkt)
            else:
                locations.append(p.wkt)

        return locations


class Utils(object):

    def get_geometries(self, components, min_area = None, max_area = None):
        locations = []
        for component in components:
            p = Polygon(component[0], component[1])
            if min_area and max_area:
                #print p.area
                if p.area > min_area and p.area < max_area:
                    locations.append(p.wkt)
            else:
                locations.append(p.wkt)

        return locations
