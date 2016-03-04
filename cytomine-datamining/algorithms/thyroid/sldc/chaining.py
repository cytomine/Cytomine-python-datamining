# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"


class ImageProvider(object):
    """
    An interface for any component that genre
    """
    __metaclass__ = ABCMeta

    def __init__(self, silent_fail=False):
        """Constructs instances of ImageProvider

        Parameters
        ----------
        silent_fail: bool
            True for putting the image provider in silent fail mode. In this situation, when an image cannot be
            extracted, the provider simply ignore the error and skip the image. Otherwise, when set to False,
            the provider raises an error when an image extraction fails.
        """
        self._silent_fail = silent_fail

    @abstractmethod
    def get_images(self):
        """
        Return the images to be processed by instances of the workflow

        Returns
        -------
        images: array
            An array of images

        Exceptions
        ----------
        ImageExtractionError:
            Raised when an image cannot be extracted. This error is never raised when the image provider is in
            silent_fail mode. In this situation, the provider fetches as many images as possible and returns only the
            successfully fetched images in the array.
        """
        pass


class WorkflowLinker(object):
    """
    An interface that links two different workflows. It is given the outputs of an execution of a workflow
    instance and generates images to be processed by a second workflow instance
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_images(self, image, polygons_classes):
        """Given result of the application of an instance of the sldc workflow, produces images objects for the next
        steps

        Parameters
        ----------
        image: Image
            The image processed by the previous step
        polygons_classes: Array of tuples
            The polygons and their predicted classes as produced by the previous class. Tuples are structured
            as (polygon, class) when polygon is an instance of shapely.geometry.Polygon and class is an integer
            code representing the actual class
        """
        pass


class PostProcessor(object):
    """
    A post processor is a class encapsulating the processing of the results of several SLDCWorkflow
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def post_process(self, image, polygons_classes):
        """Actually process the results

        Parameters
        ----------
        image: Image
            The image processed by the previous step
        polygons_classes: Array of tuples
            The polygons and their predicted classes as produced by the previous class. Tuples are structured
            as (polygon, class) when polygon is an instance of shapely.geometry.Polygon and class is an integer
            code representing the actual class
        """
        pass


class WorkflowChain(object):
    """
    This class encapsulates the sequential execution of several instances of the sldc workflow on the same image.
    A processing chain might look like this :

    {ImageProvider} --images--> {Workflow}
        [ --polygons_classes--> {WorkflowLinker} --images--> {Workflow2} [...] ]

    All the generated polygons_classes are then post_processed by the PostProcessor.
    """

    def __init__(self, image_provider, workflow, post_processor, n_jobs=1):
        """Constructor for WorkflowChain objects

        Parameters
        ----------
        image_provider: ImageProvider
            An image provider that will provide the images to be processed by the first workflow
        workflow: SLDCWorkflow
            The first instance of the workflow to be applied
        post_processor: PostProcessor
            The post-processor to execute when an image has gone through the whole processing chain
        n_jobs: int, optional (default: 1)
            The number of jobs that can be used to process the images in parallel, -1 for using the number of available
            cores
        """
        self._post_processor = post_processor
        self._image_provider = image_provider
        self._first_workflow = workflow
        self._workflows = list()
        self._linkers = list()
        self._n_jobs = n_jobs

    def append_workflow(self, workflow, workflow_linker):
        """Append a workflow to apply after the current registered sequence
        Parameters
        ----------
        workflow: SLDCWorkflow
            The workflow to append at the end of the chain
        workflow_linker: WorkflowLinker
            A linker to produce images from the prediction produced by the last workflow in the chain
        """
        self._workflows.append(workflow)
        self._linkers.append(workflow_linker)

    # TODO implement parallel implementation
    def execute(self):
        """
        Execute the processing
        """
        images = self._image_provider.get_images()

        for image in images:
            self._process_image(image)

    def _process_image(self, image):
        """
        Execute one image's processing
        Parameters
        ----------
        image: Image
            The image to process
        """
        polygons_classes = list()
        prev = self._first_workflow.process(image)
        polygons_classes.extend(prev)

        for workflow, linker in zip(self._workflows, self._linkers):
            sub_images = linker.get_images(image, prev)
            curr = list()
            for sub_image in sub_images:
                curr.extend(workflow.process(sub_image))
            polygons_classes.extend(curr)
            prev = curr

        self._post_processor.post_process(image, polygons_classes)
