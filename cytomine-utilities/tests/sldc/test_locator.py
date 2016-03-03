import numpy as np
from unittest import TestCase

from shapely.geometry import Polygon
from shapely.affinity import translate

from sldc import Locator
from util import mk_gray_img, draw_circle, draw_poly


class TestLocatorNothingToLocate(TestCase):
    def testLocator(self):
        image = mk_gray_img(2000, 3000)
        locator = Locator()
        polygons = locator.locate(image)
        self.assertEqual(0, len(polygons), "No polygon found on black image")


class TestLocatorRectangle(TestCase):
    def testLocator(self):
        image = mk_gray_img(2000, 3000)

        # draw a rectangle
        A = (25, 25)
        B = (25, 1500)
        C = (1250, 25)
        D = (1250, 1500)
        ABCD = Polygon([A, B, D, C, A])
        image = draw_poly(image, ABCD)

        # locate it
        locator = Locator()
        polygons = locator.locate(image)

        self.assertEqual(1, len(polygons), "One polygon found")
        self.assertTrue(ABCD.equals(polygons[0]), "Found polygon has the same shape")

        # test locate with an offset
        locator2 = Locator()
        polygons2 = locator2.locate(image, offset=(250, 200))
        self.assertEqual(1, len(polygons2), "One polygon found")
        self.assertTrue(translate(ABCD, 250, 200).equals(polygons2[0]), "Found translated polygon")


class TestLocatorCircleAndRectangle(TestCase):
    def testLocator(self):
        image = mk_gray_img(2000, 3000)

        # draw a rectangle
        A = (25, 400)
        B = (25, 1500)
        C = (1250, 400)
        D = (1250, 1500)
        ABCD = Polygon([A, B, D, C, A])
        image = draw_poly(image, ABCD)
        image, circle = draw_circle(image, (2500, 1500), 400)

        # test locator
        locator = Locator()
        polygons = locator.locate(image)

        self.assertEqual(2, len(polygons), "Two polygons found")
        self.assertTrue(ABCD.equals(polygons[1]), "Rectangle polygon is found")

        # use recall and false discovery rate to evaluate the error on the surface
        tpr = circle.difference(polygons[0]).area / circle.area
        fdr = polygons[0].difference(circle).area / polygons[0].area
        self.assertLessEqual(tpr, 0.002, "Recall is low for circle area")
        self.assertLessEqual(fdr, 0.002, "False discovery rate is low for circle area")
