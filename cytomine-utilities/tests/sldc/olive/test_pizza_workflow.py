from unittest import TestCase
import numpy as np
from PIL.Image import fromarray
from PIL.ImageDraw import ImageDraw
from shapely.geometry import Point
from test.olive.pizza_workflow import PizzaWorkflow, ObjectType
from test.olive.olive_workflow import OliveWorkflow
from test.olive.workflow import PizzaImageProvider, PizzaPostProcessor, PizzaOliveLinker
from test.olive.image import NumpyImage

from sldc.chaining import WorkflowChain


def draw_poly(image, poly, fill=255, edge=255):
    pil = fromarray(image)
    drawer = ImageDraw(pil)
    drawer.polygon(poly.boundary.coords, fill=fill, outline=edge)
    return np.asarray(pil)


def make_pizza(pizza_color, center, w, h, radius):
    image = np.ones((w, h, 3)).astype("uint8") * 255
    return draw_pizza_olive(image, center, radius, pizza_color, 5)


def random_point_in_circle(center, radius):
    r = np.random.randint(0, high=radius)
    theta = np.random.uniform(0, 2 * np.pi)
    x = int(r * np.cos(theta))
    y = int(r * np.sin(theta))
    return center[0] + x, center[1] + y


def draw_circle(image, radius, center, color):
    pizza_center = Point(*center)
    pizza_polygon = pizza_center.buffer(radius)
    pil_image = fromarray(image)
    draw = ImageDraw(pil_image)
    draw.polygon(pizza_polygon.boundary.coords, fill=tuple(color), outline=tuple(color))
    return np.asarray(pil_image)


def draw_pizza_olive(image, pizza_center, pizza_radius, pizza_color, count_olives):
    image = draw_circle(image, pizza_radius, pizza_center, pizza_color)
    olive_colors = [[0,255,0], [0,0,255]]
    olive_colors_ids = [ObjectType.GREEN, ObjectType.BLUE]
    olives = []
    for i in range(0, count_olives):
        color = np.random.randint(0, high=2)
        center = random_point_in_circle(pizza_center, pizza_radius - 20)
        image = draw_circle(image, 20, center, olive_colors[color])
        olives.append((center, olive_colors_ids[color]))
    return image, olives


def draw_polygon(w, h, poly, dst=None):
    im = fromarray(np.zeros((w,h,3)).astype("uint8")) if dst is None else dst
    color = 255, 255, 255
    ImageDraw(im).polygon(poly.boundary.coords, fill=color, outline=color)
    return np.asarray(im)


class TestPizzaWorkflow(TestCase):
    def testPizzaWorkflow(self):
        # draw images
        pizza1, olives1 = make_pizza([255, 255, 0], (1000, 900), 2000, 2000, 650)  # yellow pizza
        pizza2, olives2 = make_pizza([255, 0, 0], (1500, 1250), 3000, 3000, 1000)  # red pizza
        found_pizzas = []

        # initialize and launch the workflow
        workflow = PizzaWorkflow()
        image_provider = PizzaImageProvider([NumpyImage(pizza1), NumpyImage(pizza2)])
        post_processor = PizzaPostProcessor(found_pizzas)
        chain = WorkflowChain(image_provider, workflow, post_processor)
        chain.execute()

        # results
        results = post_processor.results

        self.assertEqual(2, len(results), "Two pizzas found")

        # the order of processing of the various is not known !
        found_pizza1 = post_processor.results[0]
        found_pizza2 = post_processor.results[1]

        # one of the pizza should be red and the other should be yellow
        self.assertTrue(found_pizza1[1] != found_pizza2[1] and
                        (found_pizza2[1] == ObjectType.BIG_YELLOW or
                         found_pizza2[1] == ObjectType.BIG_RED) and
                        (found_pizza1[1] == ObjectType.BIG_YELLOW or
                         found_pizza1[1] == ObjectType.BIG_RED))


class TestOlivePizzaWorkflow(TestCase):
    def testOlivePizzaWorkflow(self):
        # draw images
        pizza1, olives1 = make_pizza([255, 255, 0], (1000, 900), 2000, 2000, 650)  # yellow pizza
        pizza2, olives2 = make_pizza([255, 0, 0], (1500, 1250), 3000, 3000, 1000)  # red pizza
        found_pizzas = []

        # initialize and launch the workflow
        pizza_workflow = PizzaWorkflow()
        image_provider = PizzaImageProvider([NumpyImage(pizza1), NumpyImage(pizza2)])
        olive_workflow = OliveWorkflow()
        olive_linker = PizzaOliveLinker()
        post_processor = PizzaPostProcessor(found_pizzas)
        chain = WorkflowChain(image_provider, pizza_workflow, post_processor)
        chain.append_workflow(olive_workflow, olive_linker)
        chain.execute()

        # results
        results = post_processor.results
        self.assertTrue(2 < len(results), "More than two pizzas found")