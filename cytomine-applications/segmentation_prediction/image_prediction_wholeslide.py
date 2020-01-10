# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
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


__author__ = "Marée Raphael <raphael.maree@ulg.ac.be>"
__contributors__ = ["Gilles Louppe <g.louppe@gmail.com>", "Stévens Benjamin <b.stevens@ulg.ac.be>", "Olivier Caubo",
                    "Elodie Burtin <elodie.burtin@cytomine.coop>"]
__copyright__ = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"
__version__ = "2.0.0"

# exemple usage, see test-predict.sh
# This is a whole workflow (code to be redesigned in a future release): It can work within ROIs (e.g. tissue section),
# apply a segmentation model (pixel classifier) tile per tile, detect connected components, perform union of detected
# geometries in the whole image, apply post-processing based on min/max are, apply a final classifier on geometries,
# and finally output statistics (counts/area).

import os
import sys
import time
import socket
import logging
import shutil
import pickle

from shapely.geometry.polygon import Polygon
from sklearn.externals.joblib import Parallel, delayed
from shapely.wkt import loads
import shapely.wkt
import shapely.ops
import numpy as np
import scipy.ndimage
import cv2

try:
    import Image
    import ImageStat
except ImportError:
    from PIL import Image, ImageStat

from pyxit.data import build_from_dir
from pyxit.estimator import _get_image_data, _partition_images
import cytomine
from cytomine.utilities import WholeSlide, ObjectFinder, Bounds, CytomineReader
from cytomine.models import ImageInstanceCollection, AnnotationCollection, Annotation, AlgoAnnotationTerm


# -----------------------------------------------------------------------------------------------------------
# Functions

def _parallel_crop_boxes(y_roi, x_roi, image_filename, half_width, half_height, pyxit_colorspace):
    """
    For parallel extraction of subwindows in current tile

    Parameters
    ----------
    y_roi: list[int]
        the y positions to use as centers of subwindows
    x_roi: list[int]
        The x positions to use as centers of subwindows
    image_filename: str
        The path to the image
    half_width: int
        Half the width of the subwindows
    half_height: int
        Half the height of the subwindows
    pyxit_colorspace: int
     The colorspace to use for pyxit feature extraction (see possible enum values in pyxit.estimator)

    Returns
    -------
    tuple(list, list)
        first list contains the coordinates (as tuple(minx, miny, maxx, maxy)) of each subwindow
        second list contains the extracted pyxit features for each subwindow
    """
    _X = []
    boxes = np.empty((len(x_roi) * len(y_roi), 4), dtype=np.int)
    i = 0
    image = Image.open(image_filename)
    for y in y_roi:
        for x in x_roi:
            min_x = int(x - half_width)
            min_y = int(y - half_height)
            max_x = int(x + half_width)
            max_y = int(y + half_height)
            boxes[i] = min_x, min_y, max_x, max_y
            sub_window = image.crop(boxes[i])
            sub_window_data = _get_image_data(sub_window, pyxit_colorspace)
            _X.append(sub_window_data)
            i += 1
    return boxes, _X


def _parallel_confidence_map(pixels, y, boxes, tile_width, tile_height, n_classes, subwindow_width, subwindow_height):
    """
    For parallel construction of confidence map in current tile

    Parameters
    ----------
    pixels: list[int]
        the indexes of the pixels to consider in the subwindows
    y: list[list]
        the list of predictions (in all subwindows) for each of the considered pixels
    boxes: list[tuple(int)]
        the coordinates (minx, miny, maxx, maxy) of each subwindow used by pyxit
    tile_width: int
        the width of the tile
    tile_height: int
        the height of the tile
    n_classes: int
        number of classes handled by the classifier
    subwindow_width: int
        the width of the subwindows
    subwindow_height: int
        the height of the subwindows

    Returns
    -------
    ndarray
        array providing for each pixel of the tile the score of each class
    """
    votes_class = np.zeros((tile_width, tile_height, n_classes))

    for i, pixel_index in enumerate(pixels):
        inc_x = pixel_index % subwindow_width  # x coordinate of the pixel within the subwindows
        inc_y = pixel_index / subwindow_height  # y coordinate of the pixel within the subwindows

        for box_index, probas in enumerate(y[i]):
            px = boxes[box_index][0] + inc_x  # x coordinate of the pixel within the tile
            py = boxes[box_index][1] + inc_y  # y coordinate of the pixel within the tile
            votes_class[py, px, :] += probas

    return votes_class


def polygon_2_component(polygon):
    """
    To convert a polygon into a component

    Parameters
    ----------
    polygon: shapely.geometry.Polygon
        The polygon to convert to a componen

    Returns
    -------
    tuple(list, list)
        the first list contains the coordinates of the exterior ring
        the second list contains the interior rings, each defined by a list of coordinates
    """
    exterior = list(polygon.exterior.coords)
    interiors = []
    for interior in polygon.interiors:
        interiors.append(list(interior.coords))
    return exterior, interiors


def rasterize_tile_roi_union(nx, ny, tile_polygon, roi_annotations_union, reader):
    """
    # To convert a union of roi polygons into a rasterized tile mask

    Parameters
    ----------
    nx: int
        tile height
    ny: int
        tile width
    tile_polygon: Polygon
        polygon describing the tile
    roi_annotations_union: MultiPolygon
        union of the regions of interest
    reader: CytomineReader
        Cytomine reader used to read image tiles

    Returns
    -------
    ndarray
        mask to use for the tile
    """
    intersection = tile_polygon.intersection(roi_annotations_union)

    mask = np.zeros((ny, nx))

    if isinstance(intersection, shapely.geometry.MultiPolygon):
        intersection_polygons = [poly for poly in intersection]
    elif isinstance(intersection, shapely.geometry.Polygon):
        intersection_polygons = [intersection]
    else:  # no intersection between tile and ROI => mask is false everywhere
        return mask.astype(np.bool)

    intersection_components = [polygon_2_component(poly) for poly in intersection_polygons]
    local_intersection_components = reader.convert_to_local_coordinates(intersection_components)
    local_intersection_polygons = [Polygon(component[0], component[1]) for component in local_intersection_components]

    for poly in local_intersection_polygons:
        coords = np.array([[(int(x), int(y)) for x, y in poly.exterior.coords]])
        cv2.fillPoly(mask, coords, 1)
        for interior in poly.interiors:
            coords = np.array([[(int(x), int(y)) for x, y in interior.coords]])
            cv2.fillPoly(mask, coords, 0)

    return mask.astype(np.bool)


def process_mask(mask):
    """
    Process mask to remove unvalid polygon patterns

    Parameters
    ----------
    mask: ndarray
        The mask to process

    Returns
    -------
    ndarray
        The processed mask
    """
    # remove down-left to up-right diagonal pattern
    structure1 = np.zeros((3, 3))
    structure1[0, 2] = 1
    structure1[1, 1] = 1
    structure2 = np.zeros((3, 3))
    structure2[0, 1] = 1
    structure2[1, 2] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1=structure1,
                                                               structure2=structure2).astype(np.uint8)
    pattern_mask[pattern_mask == 1] = 255
    pattern_mask[pattern_mask == 0] = 0
    mask = mask - pattern_mask

    # remove up-left to down-right diagonal pattern
    structure1 = np.zeros((3, 3))
    structure1[0, 0] = 1
    structure1[1, 1] = 1
    structure2 = np.zeros((3, 3))
    structure2[0, 1] = 1
    structure2[1, 0] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1=structure1,
                                                               structure2=structure2).astype(np.uint8)
    pattern_mask[pattern_mask == 1] = 255
    pattern_mask[pattern_mask == 0] = 0
    mask = mask - pattern_mask
    # TODO the question is :
    # Does removing the second pattern can recreate the first one ? If so, how to avoid it? (iterative way?)

    # remove up line
    structure1 = np.zeros((3, 3))
    structure1[2, 1] = 1
    structure1[1, 1] = 1
    structure2 = np.zeros((3, 3))
    structure2[1, 0] = 1
    structure2[1, 2] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1=structure1,
                                                               structure2=structure2).astype(np.uint8)
    pattern_mask[pattern_mask == 1] = 255
    pattern_mask[pattern_mask == 0] = 0
    mask = mask - pattern_mask

    # remove down line
    structure1 = np.zeros((3, 3))
    structure1[0, 1] = 1
    structure1[1, 1] = 1
    structure2 = np.zeros((3, 3))
    structure2[1, 0] = 1
    structure2[1, 2] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1=structure1,
                                                               structure2=structure2).astype(np.uint8)
    pattern_mask[pattern_mask == 1] = 255
    pattern_mask[pattern_mask == 0] = 0
    mask = mask - pattern_mask

    # remove left line
    structure1 = np.zeros((3, 3))
    structure1[1, 1] = 1
    structure1[1, 2] = 1
    structure2 = np.zeros((3, 3))
    structure2[0, 1] = 1
    structure2[2, 1] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1=structure1,
                                                               structure2=structure2).astype(np.uint8)
    pattern_mask[pattern_mask == 1] = 255
    pattern_mask[pattern_mask == 0] = 0
    mask = mask - pattern_mask

    # remove right line
    structure1 = np.zeros((3, 3))
    structure1[1, 1] = 1
    structure1[1, 0] = 1
    structure2 = np.zeros((3, 3))
    structure2[0, 1] = 1
    structure2[2, 1] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1=structure1,
                                                               structure2=structure2).astype(np.uint8)
    pattern_mask[pattern_mask == 1] = 255
    pattern_mask[pattern_mask == 0] = 0
    mask = mask - pattern_mask

    return mask

# -----------------------------------------------------------------------------------------------------------


def run(cyto_job, parameters):
    logging.info("----- segmentation_prediction v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    project = cyto_job.project
    current_tile_annotation = None

    working_path = os.path.join("tmp", str(job.id))
    if not os.path.exists(working_path):
        logging.info("Creating annotation directory: %s", working_path)
        os.makedirs(working_path)

    try:
        # Initialization
        pyxit_target_width = parameters.pyxit_target_width
        pyxit_target_height = parameters.pyxit_target_height
        tile_size = parameters.cytomine_tile_size
        zoom = parameters.cytomine_zoom_level
        predictionstep = int(parameters.cytomine_predict_step)
        mindev = parameters.cytomine_tile_min_stddev
        maxmean = parameters.cytomine_tile_max_mean

        logging.info("Loading prediction model (local)")
        fp = open(parameters.pyxit_load_from, "r")
        logging.debug(fp)
        pickle.load(fp)  # classes => not needed
        pyxit = pickle.load(fp)
        pyxit.n_jobs = parameters.pyxit_nb_jobs  # multithread subwindows extraction in pyxit
        pyxit.base_estimator.n_jobs = parameters.pyxit_nb_jobs  # multithread tree propagation

        # loop for images in the project id TODO let user specify the images to process
        images = ImageInstanceCollection().fetch_with_filter("project", project.id)
        nb_images = len(images)
        logging.info("# images in project: %d", nb_images)
        progress = 0
        progress_delta = 100 / nb_images

        # Go through all images
        for (i, image) in enumerate(images):
            image_str = "{} ({}/{})".format(image.instanceFilename, i+1, nb_images)
            job.update(progress=progress, statusComment="Analyzing image {}...".format(image_str))
            logging.debug("Image id: %d width: %d height: %d resolution: %f magnification: %d filename: %s", image.id,
                          image.width, image.height, image.resolution, image.magnification, image.filename)

            image.colorspace = "RGB"  # required for correct handling in CytomineReader

            # Create local object to access the remote whole slide
            logging.debug("Creating connector to Slide Image from Cytomine server")
            whole_slide = WholeSlide(image)
            logging.debug("Wholeslide: %d x %d pixels", whole_slide.width, whole_slide.height)

            # endx and endy allow to stop image analysis at a given x, y position  (for debugging)
            endx = parameters.cytomine_endx if parameters.cytomine_endx else whole_slide.width
            endy = parameters.cytomine_endy if parameters.cytomine_endy else whole_slide.height

            # initialize variables and tools for ROI
            nx = tile_size
            ny = tile_size

            local_tile_component = ([(0, 0), (0, ny), (nx, ny), (nx, 0), (0, 0)], [])

            # We can apply the segmentation model either in the whole slide (including background area), or only within
            # multiple ROIs (of a given term)
            # For example ROI could be generated first using a thresholding step to detect the tissue
            # Here we build a polygon union containing all roi_annotations locations (user or reviewed annotations) to
            # later match tile with roi masks
            if parameters.cytomine_roi_term:
                logging.debug("Retrieving ROI annotations")
                roi_annotations = AnnotationCollection(image=image.id, term=parameters.cytomine_roi_term, showWKT=True,
                                                       showTerm=True, reviewed=parameters.cytomine_reviewed_roi).fetch()

                roi_annotations_locations = []
                for roi_annotation in roi_annotations:
                    roi_annotations_locations.append(shapely.wkt.loads(roi_annotation.location))
                roi_annotations_union = shapely.ops.unary_union(roi_annotations_locations)

            else:  # no ROI used
                # We build a rectangular roi_mask corresponding to the whole image filled with ones
                logging.debug("Processing all tiles")
                roi_mask = np.ones((ny, nx), dtype=np.bool)

            # Initiate the reader object which browse the whole slide image with tiles of size tile_size
            logging.info("Initiating the Slide reader")
            reader = CytomineReader(whole_slide,
                                    window_position=Bounds(parameters.cytomine_startx, parameters.cytomine_starty,
                                                           tile_size, tile_size),
                                    zoom=zoom,
                                    # overlap needed because the predictions at the borders of the tile are removed
                                    overlap=pyxit_target_width + 1)

            wsi = 0  # tile number

            logging.info("Starting browsing the image %s using tiles", image.instanceFilename)
            while True:
                tile_component = reader.convert_to_real_coordinates([local_tile_component])[0]
                tile_polygon = shapely.geometry.Polygon(tile_component[0], tile_component[1])

                # Get rasterized roi mask to match with this tile (if no ROI used, the roi_mask was built before and
                # corresponds to the whole image).
                if parameters.cytomine_roi_term:
                    roi_mask = rasterize_tile_roi_union(nx, ny, tile_polygon, roi_annotations_union, reader)

                if np.count_nonzero(roi_mask) == 0:
                    logging.info("Tile %d is not included in any ROI, skipping processing", wsi)

                else:
                    # Browse the whole slide image with catch exception
                    while True:
                        try:
                            reader.read()
                            break

                        except socket.timeout:
                            logging.error("Socket timeout for tile %d: %s", wsi, socket.timeout)
                            time.sleep(1)

                        except socket.error:
                            logging.error("Socket error for tile %d: %s", wsi, socket.error)
                            time.sleep(1)

                    tile = reader.data

                    # Get statistics about the current tile
                    logging.info("Computing tile %d statistics", wsi)
                    pos = reader.window_position
                    logging.debug("Tile zoom: %d, posx: %d, posy: %d, poswidth: %d, posheight: %d",
                                  zoom, pos.x, pos.y, pos.width, pos.height)
                    tilemean = ImageStat.Stat(tile).mean
                    logging.debug("Tile mean pixel values: %d %d %d", tilemean[0], tilemean[1], tilemean[2])
                    tilestddev = ImageStat.Stat(tile).stddev
                    logging.debug("Tile stddev pixel values: %d %d %d", tilestddev[0], tilestddev[1], tilestddev[2])

                    # Criteria to determine if tile is empty, specific to this application
                    if ((tilestddev[0] < mindev and tilestddev[1] < mindev and tilestddev[2] < mindev) or
                            (tilemean[0] > maxmean and tilemean[1] > maxmean and tilemean[2] > maxmean)):
                        logging.info("Tile %d empty (filtered by min stddev or max mean)", wsi)

                    else:
                        # This tile is not empty, we process it

                        # Add current tile annotation on server just for progress visualization purpose
                        current_tile_annotation = Annotation(tile_polygon.wkt, image.id).save()

                        # Save the tile image locally
                        image_filename = "%s/%d-zoom_%d-tile_%d_x%d_y%d_w%d_h%d.png" \
                                         % (working_path, image.id, zoom, wsi, pos.x, pos.y, pos.width, pos.height)
                        tile.save(image_filename, "PNG")

                        logging.debug("Tile file: %s", image_filename)
                        logging.info("Extraction of subwindows in tile %d", wsi)
                        width, height = tile.size

                        half_subwindow_width = int(pyxit_target_width / 2)
                        half_subwindow_height = int(pyxit_target_height / 2)

                        # Coordinates of centers of extracted subwindows
                        y_roi = range(half_subwindow_height, height - half_subwindow_height, predictionstep)
                        x_roi = range(half_subwindow_width, width - half_subwindow_width, predictionstep)
                        logging.info("%d subwindows to extract", len(x_roi)*len(y_roi))

                        n_jobs = parameters.cytomine_nb_jobs
                        n_jobs, _, starts = _partition_images(n_jobs, len(y_roi))

                        # Parallel extraction of subwindows in the current tile
                        all_data = Parallel(n_jobs=n_jobs)(
                            delayed(_parallel_crop_boxes)(
                                y_roi[starts[k]:starts[k + 1]],
                                x_roi,
                                image_filename,
                                half_subwindow_width,
                                half_subwindow_height,
                                parameters.pyxit_colorspace)
                            for k in xrange(n_jobs))

                        # Reduce
                        boxes = np.vstack(box for box, _ in all_data)
                        _X = np.vstack([X for _, X in all_data])

                        logging.info("Prediction of subwindows for tile %d", wsi)
                        # Propagate subwindow feature vectors (X) into trees and get probabilities
                        _Y = pyxit.base_estimator.predict_proba(_X)

                        # Warning: we get output vectors for all classes for pixel (0,0) for all subwindows, then pixel
                        # predictions for pixel (0,1) for all subwindows, ... We do not get predictions window after
                        # window, but output after output
                        # => Y is a list of length m, where m = nb of pixels by subwindow ;
                        #    each element of the list is itself a list of size n, where n = nb of subwindows
                        #    for each subwindow, the probabilities for each class are given

                        # <optimized code
                        logging.info("Parallel construction of confidence map in current tile")
                        pixels = range(pyxit_target_width * pyxit_target_height)
                        n_jobs, _, starts = _partition_images(n_jobs, len(pixels))

                        all_votes_class = Parallel(n_jobs=n_jobs)(
                            delayed(_parallel_confidence_map)(
                                pixels[starts[k]:starts[k + 1]],
                                _Y[starts[k]:starts[k + 1]],
                                boxes,
                                width,
                                height,
                                pyxit.base_estimator.n_classes_[0],
                                pyxit_target_width,
                                pyxit_target_height)
                            for k in xrange(n_jobs))

                        votes_class = all_votes_class[0]
                        for v in all_votes_class[1:]:
                            votes_class += v
                        # optimized code>

                        logging.info("Delete borders")
                        # Delete predictions at borders
                        for k in xrange(0, width):
                            for j in xrange(0, half_subwindow_height):
                                votes_class[j, k, :] = [1, 0]
                            for j in xrange(height - half_subwindow_height, height):
                                votes_class[j, k, :] = [1, 0]

                        for j in xrange(0, height):
                            for k in xrange(0, half_subwindow_width):
                                votes_class[j, k, :] = [1, 0]
                            for k in xrange(width - half_subwindow_width, width):
                                votes_class[j, k, :] = [1, 0]

                        votes = np.argmax(votes_class, axis=2) * 255

                        # only predict in roi region based on roi mask
                        votes[np.logical_not(roi_mask)] = 0

                        # process mask
                        votes = process_mask(votes)
                        votes = votes.astype(np.uint8)

                        # Save of confidence map locally
                        logging.info("Creating output tile file locally")
                        output = Image.fromarray(votes)
                        outputfilename = "%s/%d-zoom_%d-tile_%d_xxOUTPUT-%dx%d.png" \
                                         % (working_path, image.id, zoom, wsi, pyxit_target_width, pyxit_target_height)
                        output.save(outputfilename, "PNG")
                        logging.debug("Tile OUTPUT file: %s", outputfilename)

                        # Convert and transfer annotations of current tile
                        logging.info("Find components")
                        components = ObjectFinder(votes).find_components()
                        components = reader.convert_to_real_coordinates(components)
                        polygons = [Polygon(component[0], component[1]) for component in components]

                        logging.info("Uploading annotations...")
                        logging.debug("Number of polygons: %d" % len(polygons))
                        start = time.time()

                        for poly in polygons:
                            geometry = poly.wkt

                            if not poly.is_valid:
                                logging.warning("Invalid geometry, try to correct it with buffer")
                                logging.debug("Geometry prior to modification: %s", geometry)
                                new_poly = poly.buffer(0)
                                if not new_poly.is_valid:
                                    logging.error("Failed to make valid geometry, skipping this polygon")
                                    continue
                                geometry = new_poly.wkt

                            logging.debug("Uploading geometry %s", geometry)

                            startsingle = time.time()
                            while True:
                                try:
                                    # TODO: save collection of annotations
                                    annot = Annotation(geometry, image.id, [parameters.cytomine_predict_term]).save()
                                    if not annot:
                                        logging.error("Annotation could not be saved ; location = %s", geometry)
                                    break
                                except socket.timeout, socket.error:
                                    logging.error("socket timeout/error add_annotation")
                                    time.sleep(1)

                            endsingle = time.time()
                            logging.debug("Elapsed time for adding single annotation: %d", endsingle - startsingle)

                        # current time
                        end = time.time()
                        logging.debug("Elapsed time for adding all annotations: %d", end - start)

                        # Delete current tile annotation (progress visualization)
                        current_tile_annotation.delete()

                wsi += 1

                if not reader.next() or (reader.window_position.x > endx and reader.window_position.y > endy):
                    break  # end of browsing the whole slide

            # Postprocessing to remove small/large annotations according to min/max area
            if parameters.cytomine_postproc:
                logging.info("Post-processing before union...")
                job.update(progress=progress + progress_delta/4,
                           statusComment="Post-processing image {}...".format(image_str))
                while True:
                    try:
                        annotations = AnnotationCollection(id_user=job.userJob, id_image=image.id, showGIS=True)
                        break
                    except socket.timeout, socket.error:
                        logging.error("Socket timeout/error when fetching annotations")
                        time.sleep(1)

                # remove/edit useless annotations
                start = time.time()
                for annotation in annotations:
                    if (annotation.area == 0 or annotation.area < parameters.cytomine_min_size
                            or annotation.area > parameters.cytomine_max_size):
                        annotation.delete()
                    else:
                        logging.debug("Keeping annotation %d", annotation.id)

                end = time.time()
                logging.debug("Elapsed time for post-processing all annotations: %d" % (end - start))

            # Segmentation model was applied on individual tiles. We need to merge geometries generated from each tile.
            # We use a groovy/JTS script that downloads annotation geometries and perform union locally to relieve the
            # Cytomine server
            if parameters.cytomine_union:
                logging.info("Union of polygons for image %s", image.instanceFilename)
                job.update(progress=progress + progress_delta/3,
                           statusComment="Union of polygons in image {}...".format(image_str))
                start = time.time()
                union_command = ("groovy -cp \"lib/jars/*\" lib/union4.groovy " +
                                 "%s %s %s %d %d %d %d %d %d %d %d %d %d"
                                 % (cyto_job._base_url(False),
                                    parameters.publicKey,
                                    parameters.privateKey,
                                    image.id,
                                    job.userJob,
                                    parameters.cytomine_predict_term,
                                    parameters.cytomine_union_min_length,
                                    parameters.cytomine_union_bufferoverlap,
                                    parameters.cytomine_union_min_point_for_simplify,
                                    parameters.cytomine_union_min_point,
                                    parameters.cytomine_union_max_point,
                                    parameters.cytomine_union_nb_zones_width,
                                    parameters.cytomine_union_nb_zones_height)
                                 )
                logging.info("Union command: %s", union_command)
                os.system(union_command)
                end = time.time()
                logging.info("Elapsed time union: %d s", end - start)

            # Perform classification of detected geometries using a classification model (pkl)
            if parameters.pyxit_post_classification:
                logging.info("Post classification of all candidates")
                job.update(progress=progress + progress_delta*2/3,
                           statusComment="Post-classification in image {}...".format(image_str))

                # Retrieve locally annotations from Cytomine core produced by the segmentation job as candidates
                candidate_annotations = AnnotationCollection(user=job.userJob, image=image.id, showWKT=True,
                                                             showMeta=True).fetch()

                folder_name = "%s/crops-candidates-%d/zoom-%d/" % (working_path, image.id, zoom)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                dest_pattern = os.path.join(folder_name, "{id}.png")
                for annotation in candidate_annotations:
                    annotation.dump(dest_pattern, mask=True, alpha=True)
                    # np_image = cv2.imread(annotation.filename, -1)
                    # if np_image is not None:
                    #     alpha = np.array(np_image[:, :, 3])
                    #     image = np.array(np_image[:, :, 0:3])
                    # image[alpha == 0] = (255,255,255)  # to replace surrounding by white
                    # cv2.imwrite(annotation.filename, image)

                logging.debug("Building attributes from %s", folder_name)
                # Extract subwindows from all candidates
                x, y = build_from_dir(folder_name)
                post_fp = open(parameters.pyxit_post_classification_save_to, "r")
                classes = pickle.load(post_fp)
                pyxit = pickle.load(post_fp)
                logging.debug(pyxit)

                # pyxit parameters are in the model file
                y_proba = pyxit.predict_proba(x)
                y_predict = classes.take(np.argmax(y_proba, axis=1), axis=0)
                y_rate = np.max(y_proba, axis=1)

                # We classify each candidate annotation and keep only those predicted as cytomine_predict_term
                for annotation in candidate_annotations:
                    j = np.where(x == annotation.filename)[0][0]
                    new_term = int(y_predict[j])
                    accepted = (new_term == parameters.cytomine_predict_term)
                    logging.debug("Annotation %d %s during post-classification (class: %d proba: %d)",
                                  annotation.id, "accepted" if accepted else "rejected", int(y_predict[j]), y_rate[j])

                    if not accepted:
                        AlgoAnnotationTerm(annotation.id, parameters.cytomine_predict_term).delete()
                        AlgoAnnotationTerm(annotation.id, new_term).save()

                logging.info("End of post-classification")
                # ...

            # Perform stats (counting) in roi area
            if parameters.cytomine_count and parameters.cytomine_roi_term:
                logging.info("Compute statistics")
                # Count number of annotations in roi area
                # Get Rois
                roi_annotations = AnnotationCollection(image=image.id, term=parameters.cytomine_roi_term,
                                                       showGIS=True).fetch()

                # Count included annotations (term = predict_term) in each ROI
                for roi_annotation in roi_annotations:
                    included_annotations = AnnotationCollection(image=image.id, user=job.userJob,
                                                                bboxAnnotation=roi_annotation.id).fetch()
                    logging.info("Stats of image %s: %d annotations included in ROI %d (%d %s)",
                                 image.instanceFilename,
                                 len(included_annotations),
                                 roi_annotation.id,
                                 roi_annotation.area,
                                 roi_annotation.areaUnit)

            logging.info("Finished processing image %s", image.instanceFilename)
            progress += progress_delta
    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)

        if current_tile_annotation:
            current_tile_annotation.delete()

        logging.debug("Leaving run()")


if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)
