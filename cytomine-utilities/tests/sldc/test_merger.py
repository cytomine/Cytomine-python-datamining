from unittest import TestCase
from fake import FakeTileBuilder, FakeImage
from shapely.geometry import Polygon, box, Point
from shapely.affinity import translate
from sldc.merger import Merger


class TestMergerNoPolygon(TestCase):
    def test_merge(self):
        fake_image = FakeImage(11, 8, 3)
        fake_builder = FakeTileBuilder()
        topology = fake_image.tile_topology(5, 5, 1)

        tile1 = topology.tile(1, fake_builder)
        tile2 = topology.tile(2, fake_builder)
        tile3 = topology.tile(3, fake_builder)
        tile4 = topology.tile(4, fake_builder)

        #    0    5    10 (col)
        #  0 +---------+
        #    |    |    |
        #    |    |    |
        #    |    |    |
        #  4 |----+----+
        #    |    |    |
        #    |    |    |
        #  7 +----G----H
        # (row)
        polygons_tiles = [(tile1, []),
                          (tile2, []),
                          (tile3, []),
                          (tile4, [])]

        merger = Merger(1)
        polygons = merger.merge(polygons_tiles, topology)
        self.assertEqual(len(polygons), 0, "Number of found polygon")


class TestMergerSingleTil(TestCase):
    def test_merge(self):
        fake_image = FakeImage(11, 8, 3)
        fake_builder = FakeTileBuilder()
        topology = fake_image.tile_topology(11, 8, 1)

        tile1 = topology.tile(1, fake_builder)

        #    0    5    10 (col)
        #  0 +---------+
        #    | A--B    |
        #    | |  |    |
        #    | C--D    |
        #  4 |         |
        #    |    E----F
        #    |    |    |
        #  7 +----G----H
        # (row)

        A = (1, 2)
        B = (1, 5)
        C = (3, 1)
        D = (2, 5)

        E = (5, 5)
        F = (5, 10)
        G = (7, 5)
        H = (7, 10)

        ABCD = Polygon([A, B, D, C, A])
        EFGH = Polygon([E, F, H, G, E])

        polygons_tiles = [(tile1, [ABCD, EFGH])]

        merger = Merger(1)
        polygons = merger.merge(polygons_tiles, topology)
        self.assertEqual(len(polygons), 2, "Number of found polygon")
        self.assertTrue(polygons[0].equals(ABCD), "ABCD polygon")
        self.assertTrue(polygons[1].equals(EFGH), "EFHG polygon")


class TestMergerRectangle(TestCase):
    def test_merge(self):
        fake_image = FakeImage(30, 11, 3)
        fake_builder = FakeTileBuilder()
        topology = fake_image.tile_topology(12, 9, 2)

        tile1 = topology.tile(1, fake_builder)
        tile2 = topology.tile(2, fake_builder)
        tile3 = topology.tile(3, fake_builder)
        tile4 = topology.tile(4, fake_builder)
        tile5 = topology.tile(5, fake_builder)
        tile6 = topology.tile(6, fake_builder)

        #    0    5    10   15   20        30  (col)
        #  0 +---------+---------+---------+
        #    |         | E--F    |         |
        #    |         | |  |    |         |
        #    |         | G--H    |         |
        #  4 |         |         |         |
        #    |    A----z----B    |  I---J  |
        #    |    |    |    |    |  |   |  |
        #  7 +----u----t----s----+--p---q--+
        #    |    |    |    |    |  |   |  |
        #  9 |    C----w----D    |  K---L  |
        #    |         |         |         |
        # 11 +---------+---------+---------+
        # (row)

        A = (5, 5)
        B = (5, 15)
        C = (9, 5)
        D = (9, 15)

        E = (1, 12)
        F = (1, 15)
        G = (3, 12)
        H = (3, 15)

        I = (5, 23)
        J = (5, 27)
        K = (9, 23)
        L = (9, 27)

        p = (7, 23)
        q = (7, 27)
        s = (7, 15)
        t = (7, 10)
        u = (7, 5)
        w = (9, 10)
        z = (5, 10)

        EFHG = Polygon([E, F, H, G, E])
        Aztu = Polygon([A, z, t, u, A])
        zBst = Polygon([z, B, s, t, z])
        tsDw = Polygon([t, s, D, w, t])
        utwC = Polygon([u, t, w, C, u])
        IJqp = Polygon([I, J, q, p, I])
        pqLK = Polygon([p, q, L, K, p])
        ABCD = Polygon([A, B, D, C, A])
        IJLK = Polygon([I, J, L, K, I])

        polygons_tiles = [(tile1, [Aztu]),
                          (tile2, [EFHG, zBst]),
                          (tile3, [IJqp]),
                          (tile4, [utwC]),
                          (tile5, [tsDw]),
                          (tile6, [pqLK])]

        polygons = Merger(1).merge(polygons_tiles, topology)
        self.assertEqual(len(polygons), 3, "Number of found polygon")
        self.assertTrue(polygons[0].equals(ABCD), "ABCD polygon")
        self.assertTrue(polygons[1].equals(EFHG), "EFHG polygon")
        self.assertTrue(polygons[2].equals(IJLK), "IJLK polygon")


class TestMergerBigCircle(TestCase):
    def testMerger(self):
        # build chunks for the polygons
        tile_box = box(0, 0, 512, 256)  # a box having the same dimension as the tile
        circle = Point(600, 360)
        circle = circle.buffer(250)

        circle_part1 = tile_box.intersection(circle)
        circle_part2 = translate(tile_box, xoff=512).intersection(circle)
        circle_part3 = translate(tile_box, yoff=256).intersection(circle)
        circle_part4 = translate(tile_box, xoff=512, yoff=256).intersection(circle)
        circle_part5 = translate(tile_box, yoff=512).intersection(circle)
        circle_part6 = translate(tile_box, xoff=512, yoff=512).intersection(circle)

        # create topology
        fake_image = FakeImage(1024, 768, 3)
        fake_builder = FakeTileBuilder()
        topology = fake_image.tile_topology(512, 256)

        tile1 = topology.tile(1, fake_builder)
        tile2 = topology.tile(2, fake_builder)
        tile3 = topology.tile(3, fake_builder)
        tile4 = topology.tile(4, fake_builder)
        tile5 = topology.tile(5, fake_builder)
        tile6 = topology.tile(6, fake_builder)

        polygons_tiles = [(tile1, [circle_part1]),
                          (tile2, [circle_part2]),
                          (tile3, [circle_part3]),
                          (tile4, [circle_part4]),
                          (tile5, [circle_part5]),
                          (tile6, [circle_part6])]

        polygons = Merger(5).merge(polygons_tiles, topology)
        self.assertEqual(len(polygons), 1, "Number of found polygon")

        # use recall and false discovery rate to evaluate the error on the surface
        tpr = circle.difference(polygons[0]).area / circle.area
        fdr = polygons[0].difference(circle).area / polygons[0].area
        self.assertLessEqual(tpr, 0.002, "Recall is low for circle area")
        self.assertLessEqual(fdr, 0.002, "False discovery rate is low for circle area")
