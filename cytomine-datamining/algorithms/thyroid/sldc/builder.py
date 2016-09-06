# -*- coding: utf-8 -*-
from joblib import Parallel

from .chaining import WorkflowChain, WorkflowExecutor, DefaultFilter
from .dispatcher import DispatcherClassifier, CatchAllRule
from .errors import MissingComponentException
from .image import DefaultTileBuilder
from .logging import SilentLogger
from .workflow import SLDCWorkflow

__author__ = "Mormont Romain <romainmormont@hotmail.com>"
__version__ = "0.1"


class WorkflowBuilder(object):
    """A class for building SLDC Workflow objects. When several instances of SLDCWorkflow should be built, they should
    be with the same Builder object, especially if the workflows should work in parallel.
    """
    def __init__(self):
        """Constructor for WorkflowBuilderObjects
        Parameters
        ----------
        n_jobs: int
            Number of jobs to use for executing the workflow
        """
        # Fields below are reset after each get()
        self._segmenter = None
        self._rules = None
        self._classifiers = None
        self._dispatching_labels = None
        self._tile_max_width = None
        self._tile_max_height = None
        self._overlap = None
        self._distance_tolerance = None
        self._logger = None
        self._tile_builder = None
        self._parallel_dc = None
        self._n_jobs = None
        self._reset()

    def _reset(self):
        """Reset the sldc workflow fields to their default values"""
        self._segmenter = None
        self._tile_builder = DefaultTileBuilder()
        self._rules = []
        self._classifiers = []
        self._dispatching_labels = []
        self._tile_max_width = 1024
        self._tile_max_height = 1024
        self._overlap = 7
        self._distance_tolerance = 1
        self._parallel_dc = False
        self._n_jobs = 1
        self._logger = SilentLogger()

    def set_n_jobs(self, n_jobs):
        """Set the number of available jobs (optional)
        Parameters
        ----------
        n_jobs: int
            The number of jobs available to execute the workflow

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._n_jobs = n_jobs
        return self

    def set_parallel_dc(self, parallel_dc):
        """Specify whether the dispatching and classification will be parallelized at the workflow level (optional)
        Parameters
        ----------
        parallel_dc: boolean
            True for enabling parallelization of dispatching and classification at the workflow level

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._parallel_dc = parallel_dc
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
        """Set the tile builder
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

    def set_default_tile_builder(self):
        """Set the default tile builder as tile builder

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._tile_builder = DefaultTileBuilder()
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

    def set_distance_tolerance(self, tolerance):
        """Set the distance tolerance. If not called, a thickness of 7 is provided by default.
        Parameters
        ----------
        tolerance: int
            The maximal distance between two polygons so that they are considered from the same object

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._distance_tolerance = tolerance
        return self

    def add_classifier(self, rule, classifier, dispatching_label=None):
        """Add a classifier to which polygons can be dispatched (mandatory, at least on time).

        Parameters
        ----------
        rule: DispatchingRule
            The dispatching rule that matches the polygons to be dispatched to the classifier
        classifier: PolygonClassifier
            The polygon that classifies polygons
        dispatching_label: key (optional, default: None)
            The dispatching label for this classifier. By default, (n) is used where n is the number of rules and
            classifiers added before.

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._dispatching_labels.append(dispatching_label if dispatching_label is not None else len(self._rules))
        self._rules.append(rule)
        self._classifiers.append(classifier)
        return self

    def add_catchall_classifier(self, classifier, dispatching_label="catchall"):
        """Add a classifier which is dispatched all the polygons that were note dispatched by the previously added
        classifiers.

        Parameters
        ----------
        classifier: PolygonClassifier
            The classifier
        dispatching_label: key (optional, default: "catchall")
            The dispatching label
        """
        return self.add_classifier(CatchAllRule(), classifier, dispatching_label=dispatching_label)

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

        dispatcher_classifier = DispatcherClassifier(self._rules, self._classifiers,
                                                     dispatching_labels=self._dispatching_labels, logger=self._logger)
        workflow = SLDCWorkflow(self._segmenter, dispatcher_classifier, self._tile_builder,
                                dist_tolerance=self._distance_tolerance,
                                tile_max_height=self._tile_max_height, tile_max_width=self._tile_max_width,
                                tile_overlap=self._overlap, logger=self._logger, n_jobs=self._n_jobs,
                                parallel_dc=self._parallel_dc)
        self._reset()
        return workflow


class WorkflowChainBuilder(object):
    """A class for building workflow chains objects
    """
    def __init__(self):
        self._first_workflow = None
        self._executors = None
        self._filters = None
        self._labels = None
        self._logger = None
        self._reset()

    def _reset(self):
        """Resets the builder so that it can build a new workflow chain
        """
        self._first_workflow = None
        self._executors = []
        self._filters = []
        self._labels = []
        self._logger = SilentLogger()

    def set_first_workflow(self, workflow, label=None):
        """Set the workflow that will process the full image
        Parameters
        ----------
        workflow: SLDCWorkflow
            The workflow
        label: hashable (optional)
            The label identifying the workflow. If not set, this label is set to 0.

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        actual_label = 0 if label is None else label
        if self._first_workflow is None:
            self._labels.insert(0, actual_label)
        else:
            self._labels[0] = actual_label
        self._first_workflow = workflow
        return self

    def add_executor(self, workflow, filter=DefaultFilter(), label=None, logger=SilentLogger(), n_jobs=1):
        """Adds a workflow executor to be executed by the workflow chain.

        Parameters
        ----------
        workflow: SLDCWorkflow
            The workflow object
        filter: PolygonFilter (optional, default: DefaultFilter)
            The polygon filter implementing the filtering of polygons of which the windows will be processed to
            the workflow.
        label: hashable (optional)
            The label identifying the executor. If not set, the number of the executor is used instead (starting at 1)
        logger: Logger (optional, default: SilentLogger)
            The logger to be used by the executor object
        n_jobs: int (optional, default: 1)
            The number of jobs for executing the workflow on the images

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._executors.append(WorkflowExecutor(workflow, logger=logger, n_jobs=n_jobs))
        self._filters.append(filter)
        actual_label = len(self._executors) if label is None else label
        self._labels.append(actual_label)
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
        if self._first_workflow is None:
            raise MissingComponentException("Missing first workflow.")
        if len(self._labels) != len(self._executors) + 1:
            raise MissingComponentException("The number of labels ({}) should be the".format(len(self._labels)) +
                                            " same as the number of workflows ({}).".format(len(self._executors) + 1))
        if len(self._filters) != len(self._executors):
            raise MissingComponentException("The number of filters ({}) should be the".format(len(self._filters)) +
                                            " same as the number of executors ({}).".format(len(self._executors)))

        chain = WorkflowChain(self._first_workflow, self._executors, self._filters, self._labels, logger=self._logger)
        self._reset()
        return chain
