from sldc.chaining import ImageProvider, PostProcessor, WorkflowLinker
from pizza_workflow import ObjectType


class PizzaImageProvider(ImageProvider):
    def __init__(self, images):
        ImageProvider.__init__(self)
        self._images = images

    def get_images(self):
        return self._images


class PizzaOliveLinker(WorkflowLinker):
    def get_images(self, image, polygons_classes):
        images = []
        for polygon, cls in polygons_classes:
            if cls == ObjectType.BIG_RED or cls == ObjectType.BIG_YELLOW:
                minx, miny, maxx, maxy = polygon.bounds
                images.append(image.window((int(miny), int(minx)), int(maxx - minx), int(maxy - miny)))
        return images


class PizzaPostProcessor(PostProcessor):
    def __init__(self, results):
        self._results = results

    @property
    def results(self):
        return self._results

    def post_process(self, image, polygons_classes):
        self._results.extend(polygons_classes)
