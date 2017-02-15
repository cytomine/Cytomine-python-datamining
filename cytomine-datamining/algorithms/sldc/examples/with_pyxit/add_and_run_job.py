import tempfile
from argparse import ArgumentParser

import os
import cv2
import numpy as np
from cytomine import Cytomine
from cytomine.models import AlgoAnnotationTerm
from cytomine_sldc import CytomineSlide, CytomineTileBuilder
from shapely.affinity import affine_transform, translate
from sklearn.utils import check_random_state
from sldc import DispatchingRule, ImageWindow, Loggable, Logger, Segmenter, StandardOutputLogger, WorkflowBuilder

from cytomine_utilities import CytomineJob
from pyxit_classifier import PyxitClassifierAdapter


def _upload_annotation(cytomine, img_inst, polygon, label=None, proba=1.0):
    """Upload an annotation and its term (if provided)"""
    image_id = img_inst.id

    # Transform polygon to match cytomine (bottom-left) origin point
    polygon = affine_transform(polygon, [1, 0, 0, -1, 0, img_inst.height])

    annotation = cytomine.add_annotation(polygon.wkt, image_id)
    if label is not None and annotation is not None:
        cytomine.add_annotation_term(annotation.id, label, label, proba, annotation_term_model=AlgoAnnotationTerm)


class DemoSegmenter(Segmenter):
    def __init__(self, threshold):
        """A simple segmenter that performs a simple thresholding on the Green channel of the image"""
        self._threshold = threshold

    def segment(self, image):
        mask = np.array(image[:, :, 1] < self._threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask * 255


class ValidAreaRule(DispatchingRule):
    def __init__(self, min_area):
        """A rule which matches polygons of which the area is greater than min_area"""
        super(ValidAreaRule, self).__init__()
        self._min_area = min_area

    def evaluate(self, image, polygon):
        return self._min_area < polygon.area


class DemoJob(CytomineJob, Loggable):

    def __init__(self, cytomine, software_id, project_id, job_parameters,
                 tile_overlap, tile_width, tile_height, n_jobs, threshold,
                 min_area, model_path, rseed, working_path):
        """An example job implementing an sldc workflow.

        Parameters
        ----------
        cytomine: Cytomine
            Cytomine client
        software_id: int
            Cytomine software id
        project_id: int
            Cytomine project id
        job_parameters: dict
            Job parameters
        tile_overlap: int
            Number of pixel of overlap between the tiles
        tile_width: int
            Maximum width of the tiles
        tile_height: int
            Maximum height of the tiles
        n_jobs: int
            Number of available jobs
        threshold: int
            Segmentation threshold in [0, 255]
        min_area: int
            Minimum area of the valid objects in pixel squared
        model_path: str
            Path to the pickled pyxit model
        rseed: int
            Random seed
        working_path: str
            Working path of the workflow (for temporary files)
        """
        CytomineJob.__init__(self, cytomine, software_id, project_id, parameters=job_parameters)
        Loggable.__init__(self, logger=StandardOutputLogger(Logger.INFO))
        self._cytomine = cytomine

        # create workflow component
        random_state = check_random_state(rseed)
        tile_builder = CytomineTileBuilder(cytomine, working_path=working_path)
        segmenter = DemoSegmenter(threshold)
        area_rule = ValidAreaRule(min_area)
        classifier = PyxitClassifierAdapter.build_from_pickle(
            model_path, tile_builder, self.logger,
            random_state=random_state,
            n_jobs=n_jobs,
            working_path=working_path
        )

        builder = WorkflowBuilder()
        builder.set_n_jobs(n_jobs)
        builder.set_logger(self.logger)
        builder.set_overlap(tile_overlap)
        builder.set_tile_size(tile_width, tile_height)
        builder.set_tile_builder(tile_builder)
        builder.set_segmenter(segmenter)
        builder.add_classifier(area_rule, classifier, dispatching_label="valid")
        self._workflow = builder.get()

    def run(self, slide):
        """Run the workflow on the given image and upload the results to cytomine"""
        results = self._workflow.process(slide)

        # Upload results
        for polygon, dispatch, cls, proba in results:
            if cls is not None:
                # if image is a window, the polygon must be translated
                if isinstance(slide, ImageWindow):
                    polygon = translate(polygon, slide.abs_offset_x, slide.abs_offset_y)
                # actually upload the annotation
                _upload_annotation(
                    self._cytomine,
                    slide.image_instance,
                    polygon,
                    label=cls,
                    proba=proba
                )

        return results


def main(argv):
    parser = ArgumentParser(prog="Demo_SLDC_Workflow_With_Pyxit", description="Demo software for SLDC Workflow on Cytomine")
    parser.add_argument('--cytomine_host', dest="cytomine_host", default='demo.cytomine.be')
    parser.add_argument('--cytomine_public_key', dest="cytomine_public_key")
    parser.add_argument('--cytomine_private_key', dest="cytomine_private_key")
    parser.add_argument('--cytomine_base_path', dest="cytomine_base_path", default='/api/')
    default_working_path = os.path.join(tempfile.gettempdir(), "cytomine")
    parser.add_argument('--cytomine_working_path', dest="cytomine_working_path", default=default_working_path)
    parser.add_argument('--cytomine_id_software', dest="cytomine_id_software", type=int)
    parser.add_argument("--cytomine_id_project", dest="cytomine_id_project", type=int)
    parser.add_argument("--cytomine_id_image", dest="cytomine_id_image", type=int)
    parser.add_argument("--sldc_tile_overlap", dest="sldc_tile_overlap", type=int, default=10)
    parser.add_argument("--sldc_tile_width", dest="sldc_tile_width", type=int, default=768)
    parser.add_argument("--sldc_tile_height", dest="sldc_tile_height", type=int, default=768)
    parser.add_argument("--pyxit_model_path", dest="pyxit_model_path")
    parser.add_argument("--n_jobs", dest="n_jobs", type=int, default=1)
    parser.add_argument("--min_area", dest="min_area", type=int, default=500)
    parser.add_argument("--threshold", dest="threshold", type=int, default=215)
    parser.add_argument("--rseed", dest="rseed", type=int, default=0)
    default_workflow_wpath = os.path.join(tempfile.gettempdir(), "sldc")
    parser.add_argument("--working_path", dest="working_path", default=default_workflow_wpath)

    params, other = parser.parse_known_args(argv)

    # Initialize cytomine client
    cytomine = Cytomine(
        params.cytomine_host,
        params.cytomine_public_key,
        params.cytomine_private_key,
        working_path=params.cytomine_working_path,
        base_path=params.cytomine_base_path
    )

    if not os.path.exists(params.working_path):
        os.makedirs(params.working_path)
    if not os.path.exists(params.cytomine_working_path):
        os.makedirs(params.cytomine_working_path)

    with DemoJob(cytomine, params.cytomine_id_software, params.cytomine_id_project, params.__dict__,
                 params.sldc_tile_overlap, params.sldc_tile_width, params.sldc_tile_height, params.n_jobs,
                 params.threshold, params.min_area, params.pyxit_model_path, params.rseed, params.working_path) as job:
        slide = CytomineSlide(cytomine, params.cytomine_id_image)
        job.run(slide)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
