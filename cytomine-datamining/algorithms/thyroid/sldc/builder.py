# -*- coding: utf-8 -*-
from joblib import Parallel

from chaining import WorkflowChain
from dispatcher import DispatcherClassifier, CatchAllRule
from workflow import SLDCWorkflow
from logging import SilentLogger
from errors import MissingComponentException

__author__ = "Mormont Romain <romainmormont@hotmail.com>"
__version__ = "0.1"


class WorkflowBuilder(object):
    """A class for building SLDC Workflow objects. When several instances of SLDCWorkflow should be built, they should
    be with the same Builder object, especially if the workflows should work in parallel.
    """
    def __init__(self, n_jobs=1):
        """Constructor for WorkflowBuilderObjects
        Parameters
        ----------
        n_jobs: int
            Number of jobs to use for executing the workflow
        """
        # Pool is preserved for building several instances of the workflow
        self._pool = Parallel(n_jobs=n_jobs)
        # Fields below are reset after each get()
        self._segmenter = None
        self._rules = None
        self._classifiers = None
        self._tile_max_width = None
        self._tile_max_height = None
        self._overlap = None
        self._boundary_thickness = None
        self._logger = None
        self._tile_builder = None
        self._parallel = None
        self._reset()

    def _reset(self):
        """Reset the sldc workflow fields to their default values"""
        self._segmenter = None
        self._tile_builder = None
        self._rules = []
        self._classifiers = []
        self._tile_max_width = 1024
        self._tile_max_height = 1024
        self._overlap = 5
        self._boundary_thickness = 7
        self._parallel = False
        self._logger = SilentLogger()

    def set_parallel(self):
        """Enable parallel processing for the workflow

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._parallel = True
        return self

    def set_segmenter(self, segmenter):
        """Set the segmenter (mandatory)
        Parameters
        ----------
        segmenter: Segmenter
            The segmenter

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._segmenter = segmenter
        return self

    def set_logger(self, logger):
        """Set the logger. If not called, a SilentLogger is provided by default.
        Parameters
        ----------
        logger: Logger
            The logger

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._logger = logger
        return self

    def set_tile_builder(self, tile_builder):
        """Set the tile builder (mandatory)
        Parameters
        ----------
        tile_builder: TileBuilder
            The tile builder

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._tile_builder = tile_builder
        return self

    def set_tile_size(self, width, height):
        """Set the tile sizes. If not called, sizes (1024, 1024) are provided by default.
        Parameters
        ----------
        width: int
            The maximum width of the tiles
        height: int
            The maximum height of the tiles

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._tile_max_width = width
        self._tile_max_height = height
        return self

    def set_overlap(self, overlap):
        """Set the tile overlap. If not called, an overlap of 5 is provided by default.
        Parameters
        ----------
        overlap: int
            The tile overlap

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._overlap = overlap
        return self

    def set_boundary_thickness(self, thickness):
        """Set the boundary thickness. If not called, a thickness of 7 is provided by default.
        Parameters
        ----------
        thickness: int
            The boundary thickness

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._boundary_thickness = thickness
        return self

    def add_classifier(self, rule, classifier):
        """Add a classifier to which polygons can be dispatched (mandatory, at least on time).

        Parameters
        ----------
        rule: DispatchingRule
            The dispatching rule that matches the polygons to be dispatched to the classifier
        classifier: PolygonClassifier
            The polygon that classifies polygons

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._rules.append(rule)
        self._classifiers.append(classifier)
        return self

    def add_catchall_classifier(self, classifier):
        """Add a classifier which is dispatched all the polygons that were note dispatched by the previously added
        classifiers.
        """
        return self.add_classifier(CatchAllRule(), classifier)

    def get(self):
        """Build the workflow with the set parameters
        Returns
        -------
        workflow: SLDCWorkflow
            The SLDC Workflow

        Raises
        ------
        MissingComponentException:
            If some mandatory elements were not provided to the builder
        """
        if self._segmenter is None:
            raise MissingComponentException("Missing segmenter.")
        if self._tile_builder is None:
            raise MissingComponentException("Missing tile builder.")
        if len(self._rules) == 0 or len(self._classifiers) == 0:
            raise MissingComponentException("Missing classifiers.")

        dispatcher_classifier = DispatcherClassifier(self._rules, self._classifiers, logger=self._logger)
        workflow = SLDCWorkflow(self._segmenter, dispatcher_classifier, self._tile_builder,
                                boundary_thickness=self._boundary_thickness,
                                tile_max_height=self._tile_max_height, tile_max_width=self._tile_max_width,
                                tile_overlap=self._overlap, logger=self._logger,
                                worker_pool=self._pool if self._parallel else None)
        self._reset()
        return workflow


class WorkflowChainBuilder(object):
    """A class for building workflow chains objects
    """
    def __init__(self):
        self._executors = None
        self._provider = None
        self._post_processor = None
        self._logger = None
        self._reset()

    def _reset(self):
        """Resets the builder so that it can build a new workflow chain
        """
        self._executors = []
        self._provider = None
        self._post_processor = None
        self._logger = SilentLogger()

    def add_executor(self, workflow_executor):
        """Adds a workflow executor to be executed by the workflow chain.
        The executors added through this method are submitted to the built WorkflowChain in the same order.

        Parameters
        ----------
        workflow_executor: WorkflowExecutor
            The workflow executor

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._executors.append(workflow_executor)
        return self

    def set_post_processor(self, post_processor):
        """Set the post processor of the workflow chain

        Parameters
        ----------
        post_processor: PostProcessor
            The post processor

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._post_processor = post_processor
        return self

    def set_image_provider(self, image_provider):
        """Set the image provider of the workflow chain

        Parameters
        ----------
        image_provider: ImageProvider

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._provider = image_provider
        return self

    def set_logger(self, logger):
        """Set the logger of the workflow chain

        Parameters
        ----------
        logger: Logger
            The logger

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._logger = logger
        return self

    def get(self):
        """Build the workflow chain with the set parameters
        Returns
        -------
        workflow: WorkflowChain
            The workflow chain

        Raises
        ------
        MissingComponentException:
            If some mandatory elements were not provided to the builder
        """
        if self._provider is None:
            raise MissingComponentException("Missing image provider.")
        if self._post_processor is None:
            raise MissingComponentException("Missing post processor")
        if len(self._executors) <= 0:
            raise MissingComponentException("At least one workflow executor should be provided.")

        chain = WorkflowChain(self._provider, self._executors, self._post_processor, logger=self._logger)
        self._reset()
        return chain
