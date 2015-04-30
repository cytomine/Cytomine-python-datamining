#!/bin/bash

#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


#__author__          = "Marée Raphael <raphael.maree@ulg.ac.be>"
#__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"



#Example to run an object finder over a whole gigapixel image

#0. Edit the add_software.py file to add the software to Cytomine Core (once) and project (once)

#1. Edit following XXX and 0 values with your cytomine identifiers

#2. Replace XXX values by your settings
cytomine_host="XXX"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
cytomine_id_project=XXX
cytomine_id_image=XXX
cytomine_id_software=XXX
cytomine_predict_term=XXX #id of term to associate to objects detected by object finder (0 if undefined)
cytomine_working_path=/bigdata/tmp/cytomine/
cytomine_zoom_level=3 #zoom level
cytomine_tile_size=512 #size of the tile
cytomine_filter="otsu" #filter applied to the tile (adaptive,binary,otsu)
cytomine_min_area=100 #minimum area of geometries to keep
cytomine_max_area=100000 #maximum area of geometries to keep
cytomine_tile_overlap=0 #overlap between successive tile

#3. Adapt union parameters (to merge geometries over the whole slide by merging local geometries detected in each tile)
cytomine_union_minlength=10 # we consider merging polygons that have at least 20 pixels in common
cytomine_union_bufferoverlap=5 # for each polygon, we look in a surrounding region of 5 pixels for considering neighboor polygons
cytomine_union_area=5000
cytomine_union_min_point_for_simplify=1000  #if an annotation has more than x points after union, it will be simplified (default 10000)
cytomine_union_min_point=500 #minimum number of points for simplified annotation
cytomine_union_max_point=1000 #maximum number of points for simplified annotation
cytomine_union_nb_zones_width=5 #an image is divided into this number of horizontal grid cells to perform lookup
cytomine_union_nb_zones_height=5 #an image is divided into this number of vertical grid cells to perform lookup



#Note: The script will go through a whole image at given zoom_level, apply filtering at each tile,
# and upload detected connected components (eventually with term id) to Cytomine server. It then
# perform a union to merge connected components that intersect accross tiles (e.g. cell clusters)
# Warning: the script saves each tile locally to an existing directory
# $working_path/image-id_image-tile*** (see code)

python ../image_wholeslide_objectfinder.py --cytomine_host $cytomine_host --cytomine_public_key $cytomine_public_key --cytomine_private_key $cytomine_private_key --cytomine_base_path /api/ --cytomine_working_path $cytomine_working_path --cytomine_id_software $cytomine_id_software --cytomine_id_project $cytomine_id_project  --cytomine_id_image $cytomine_id_image --cytomine_tile_size $cytomine_tile_size --cytomine_zoom_level $cytomine_zoom_level --cytomine_tile_overlap $cytomine_tile_overlap --cytomine_filter $cytomine_filter --cytomine_union_min_length $cytomine_union_minlength --cytomine_union_bufferoverlap $cytomine_union_bufferoverlap --cytomine_union_area $cytomine_union_area --cytomine_union_min_point_for_simplify $cytomine_union_min_point_for_simplify  --cytomine_union_min_point $cytomine_union_min_point --cytomine_union_max_point $cytomine_union_max_point --cytomine_union_nb_zones_width $cytomine_union_nb_zones_width --cytomine_union_nb_zones_height $cytomine_union_nb_zones_height --cytomine_predict_term $cytomine_predict_term #--cytomine_min_area $cytomine_min_area #--cytomine_max_area $cytomine_max_area


