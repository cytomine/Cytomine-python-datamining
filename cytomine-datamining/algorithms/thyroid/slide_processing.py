import cv2
import numpy as np
from sldc import Segmenter, DispatcherClassifier, SLDCWorkflow, SilentLogger
from dispatching_rules import CellRule, AggregateRule
from helpers.datamining.colordeconvoluter import ColorDeconvoluter


def get_standard_kernel():
    """Return the standard color deconvolution kernel"""
    kernel = np.array([[56.24850493, 71.98403122, 22.07749587],
                       [48.09104103, 62.02717516, 37.36866958],
                       [9.17867488, 10.89206473, 5.99225756]])
    return kernel


class SlideSegmenter(Segmenter):
    """
    A segmenter performing :
    - Color deconvolution (see :class:`ColorDeconvoluter`)
    - Static thresholding (identify cells)
    - Morphological closure (remove holes in cells)
    - Morphological opening (remove small objects)
    - Morphological closure (merge close objects)

    Format
    ------
    the given numpy.ndarrays are supposed to be RGB images :
    np_image.shape = (height, width, color=3) with values in the range
    [0, 255]

    Constructor parameters
    ----------------------
    color_deconvoluter : :class:`ColorDeconvoluter` instance
        The color deconvoluter performing the deconvolution
    threshold : int [0, 255] (default : 120)
        The threshold value. The higher, the more true/false positive
    struct_elem : binary numpy.ndarray (default : None)
        The structural element used for the morphological operations. If
        None, a default will be supplied
    nb_morph_iter : sequence of int (default [1,3,7])
        The number of iterations of each morphological operation.
            nb_morph_iter[0] : number of first closures
            nb_morph_iter[1] : number of openings
            nb_morph_iter[2] : number of second closures
    """

    def __init__(self, color_deconvoluter, threshold=120,
                 struct_elem=None, nb_morph_iter=None):
        self._color_deconvoluter = color_deconvoluter
        self._threshold = threshold
        self._struct_elem = struct_elem
        if self._struct_elem is None:
            self._struct_elem = np.array([[0, 0, 1, 1, 1, 0, 0],
                                          [0, 1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1],
                                          [0, 1, 1, 1, 1, 1, 0],
                                          [0, 0, 1, 1, 1, 0, 0], ],
                                         dtype=np.uint8)
        self._nb_morph_iter = [1, 3, 7] if nb_morph_iter is None else nb_morph_iter

    def segment(self, np_image):
        tmp_image = self._color_deconvoluter.transform(np_image)

        # Static thresholding on the gray image
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(tmp_image, self._threshold, 255, cv2.THRESH_BINARY_INV)

        # Remove holes in cells in the binary image
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._struct_elem, iterations=self._nb_morph_iter[0])

        # Remove small objects in the binary image
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._struct_elem, iterations=self._nb_morph_iter[1])

        # Union architectural paterns
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._struct_elem, iterations=self._nb_morph_iter[2])

        return binary


class SlideDispatcherClassifier(DispatcherClassifier):
    def __init__(self, cell_classifier, aggregate_classifier, cell_dispatch_classifier,
                 aggregate_dispatch_classifier, logger=SilentLogger()):
        """Constructor for SlideDispatcherClassifier objects
        Objects which aren't neither cells, nor aggregate are classified None

        Parameters
        ----------
        cell_classifier: PolygonClassifier
            The classifiers for cells
        aggregate_classifier: PolygonClassifier
            The classifiers for aggregates
        cell_dispatch_classifier: PyxitClassifierAdapter
            The classifier for dispatching cells
        aggregate_dispatch_classifier: PyxitClassifierAdapater
            The classifier for dispaching aggregates
        logger: Logger (optional, default: a SilentLogger instance)
            A logger object
        """
        cell_rule = CellRule(cell_dispatch_classifier, logger=logger)
        aggregate_rule = AggregateRule(aggregate_dispatch_classifier, logger=logger)
        rules = [cell_rule, aggregate_rule]
        classifiers = [cell_classifier, aggregate_classifier]
        DispatcherClassifier.__init__(self, rules, classifiers, logger=logger)


class SlideProcessingWorkflow(SLDCWorkflow):
    """
    The workflow for performing the first processing of a whole slide
    """

    def __init__(self, tile_builder, cell_classifier, aggregate_classifier, cell_dispatch_classifier,
                 aggregate_dispatch_classifier, tile_max_width=1024, tile_max_height=1024, overlap=15,
                 logger=SilentLogger()):
        """Constructor for SlideProcessingWorkflow objects

        Parameters
        ----------
        tile_builder: TileBuilder
            The tile builder
        cell_classifier: PolygonClassifier
            The classifier for cells
        aggregate_classifier: PolygonClassifier
            The classifier for aggregates
        cell_dispatch_classifier: PyxitClassifierAdapter
            The classifier for dispatching cells
        aggregate_dispatch_classifier: PyxitClassifierAdapater
            The classifier for dispaching aggregates
        tile_max_width: int
            The maximum width of the tile to use when iterating over the images
        tile_max_height: int
            The maximum height of the tile to use when iterating over the images
        overlap: int
            The number of pixels of overlap between the tiles when iterating over the images
        logger: Logger (optional, default: a SilentLogger instance)
            A logger object
        """
        color_deconvoluter = ColorDeconvoluter()
        color_deconvoluter.set_kernel(get_standard_kernel())
        segmenter = SlideSegmenter(color_deconvoluter)
        dispatcher_classifier = SlideDispatcherClassifier(aggregate_classifier, cell_classifier,
                                                          cell_dispatch_classifier, aggregate_dispatch_classifier,
                                                          logger=logger)
        SLDCWorkflow.__init__(self, segmenter, dispatcher_classifier, tile_builder, tile_max_width=tile_max_width,
                              tile_max_height=tile_max_height, tile_overlap=overlap, logger=logger)
