from unittest import TestCase
from fake import FakeImage, FakeTileBuilder


class TestTileFromImage(TestCase):
    def testTile(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(2500, 1750, 3)
        # Simple tile extraction
        tile = fake_image.tile(fake_builder, (1250, 1300), 250, 300)
        self.assertEqual(1250, tile.offset_x, "Tile from image : x offset")
        self.assertEqual(1300, tile.offset_y, "Tile from image : y offset")
        self.assertEqual(250, tile.width, "Tile from image : width")
        self.assertEqual(300, tile.height, "Tile from image : height")

        # Overflowing tile extraction
        tile = fake_image.tile(fake_builder, (1250, 1300), 1000, 1000)
        self.assertEqual(1250, tile.offset_x, "Overflowing tile from image : x offset")
        self.assertEqual(1300, tile.offset_y, "Overflowing tile from image : y offset")
        self.assertEqual(1000, tile.width, "Overflowing tile from image : width")
        self.assertEqual(450, tile.height, "Overflowing tile from image : height")

        # Both dimension overflowing
        tile = fake_image.tile(fake_builder, (2400, 1650), 300, 300)
        self.assertEqual(2400, tile.offset_x, "Both overflowing tile from image : x offset")
        self.assertEqual(1650, tile.offset_y, "Both overflowing tile from image : y offset")
        self.assertEqual(100, tile.width, "Both overflowing tile from image : width")
        self.assertEqual(100, tile.height, "Both overflowing tile from image : height")


class TestSingleTileTopology(TestCase):
    def testSingleTileTopology(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(700, 700, 3)
        topology = fake_image.tile_topology(700, 700, 100)

        # topology metrics
        self.assertEqual(1, topology.tile_count, "Topology : tile count")
        self.assertEqual(1, topology.tile_horizontal_count, "Topology : tile horizontal count")
        self.assertEqual(1, topology.tile_vertical_count, "Topology : tile vertical count")

        tile = topology.tile(1, fake_builder)
        self.assertEqual(0, tile.offset_x, "Tile from image : x offset")
        self.assertEqual(0, tile.offset_y, "Tile from image : y offset")
        self.assertEqual(700, tile.width, "Tile from image : width")
        self.assertEqual(700, tile.height, "Tile from image : height")


class TestFittingTileTopology(TestCase):
    def testFittingTileTopology(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(700, 700, 3)
        topology = fake_image.tile_topology(300, 300, 100)

        # topology metrics
        self.assertEqual(9, topology.tile_count, "Topology : tile count")
        self.assertEqual(3, topology.tile_horizontal_count, "Topology : tile horizontal count")
        self.assertEqual(3, topology.tile_vertical_count, "Topology : tile vertical count")

        # Topology that fits exactely the image
        tile = topology.tile(1, fake_builder)
        self.assertEqual(1, tile.identifier, "Topology, tile 1 : x offset")
        self.assertEqual(0, tile.offset_x, "Topology, tile 1 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 1 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 1 : width")
        self.assertEqual(300, tile.height, "Topology, tile 1 : height")

        tile = topology.tile(2, fake_builder)
        self.assertEqual(2, tile.identifier, "Topology, tile 2 : x offset")
        self.assertEqual(200, tile.offset_x, "Topology, tile 2 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 2 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 2 : width")
        self.assertEqual(300, tile.height, "Topology, tile 2 : height")

        tile = topology.tile(3, fake_builder)
        self.assertEqual(3, tile.identifier, "Topology, tile 3 : x offset")
        self.assertEqual(400, tile.offset_x, "Topology, tile 3 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 3 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 3 : width")
        self.assertEqual(300, tile.height, "Topology, tile 3 : height")

        tile = topology.tile(4, fake_builder)
        self.assertEqual(4, tile.identifier, "Topology, tile 4 : x offset")
        self.assertEqual(0, tile.offset_x, "Topology, tile 4 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 4 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 4 : width")
        self.assertEqual(300, tile.height, "Topology, tile 4 : height")

        tile = topology.tile(5, fake_builder)
        self.assertEqual(5, tile.identifier, "Topology, tile 5 : x offset")
        self.assertEqual(200, tile.offset_x, "Topology, tile 5 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 5 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 5 : width")
        self.assertEqual(300, tile.height, "Topology, tile 5 : height")

        tile = topology.tile(6, fake_builder)
        self.assertEqual(6, tile.identifier, "Topology, tile 6 : x offset")
        self.assertEqual(400, tile.offset_x, "Topology, tile 6 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 6 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 6 : width")
        self.assertEqual(300, tile.height, "Topology, tile 6 : height")

        tile = topology.tile(7, fake_builder)
        self.assertEqual(7, tile.identifier, "Topology, tile 7 : x offset")
        self.assertEqual(0, tile.offset_x, "Topology, tile 7 : x offset")
        self.assertEqual(400, tile.offset_y, "Topology, tile 7 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 7 : width")
        self.assertEqual(300, tile.height, "Topology, tile 7 : height")

        tile = topology.tile(8, fake_builder)
        self.assertEqual(8, tile.identifier, "Topology, tile 8 : x offset")
        self.assertEqual(200, tile.offset_x, "Topology, tile 8 : x offset")
        self.assertEqual(400, tile.offset_y, "Topology, tile 8 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 8 : width")
        self.assertEqual(300, tile.height, "Topology, tile 8 : height")

        tile = topology.tile(9, fake_builder)
        self.assertEqual(9, tile.identifier, "Topology, tile 9 : x offset")
        self.assertEqual(400, tile.offset_x, "Topology, tile 9 : x offset")
        self.assertEqual(400, tile.offset_y, "Topology, tile 9 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 9 : width")
        self.assertEqual(300, tile.height, "Topology, tile 9 : height")

        # neighbours
        self.assertEqual(topology.tile_neighbours(1), (None, 4, None, 2))
        self.assertEqual(topology.tile_neighbours(2), (None, 5, 1, 3))
        self.assertEqual(topology.tile_neighbours(3), (None, 6, 2, None))
        self.assertEqual(topology.tile_neighbours(4), (1, 7, None, 5))
        self.assertEqual(topology.tile_neighbours(5), (2, 8, 4, 6))
        self.assertEqual(topology.tile_neighbours(6), (3, 9, 5, None))
        self.assertEqual(topology.tile_neighbours(7), (4, None, None, 8))
        self.assertEqual(topology.tile_neighbours(8), (5, None, 7, 9))
        self.assertEqual(topology.tile_neighbours(9), (6, None, 8, None))


class TestOverflowingTopology(TestCase):

    def testOverFlowingTopology(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(600, 450, 3)
        topology = fake_image.tile_topology(300, 300, 100)

        # topology metrics
        self.assertEqual(6, topology.tile_count, "Topology : tile count")
        self.assertEqual(3, topology.tile_horizontal_count, "Topology : tile horizontal count")
        self.assertEqual(2, topology.tile_vertical_count, "Topology : tile vertical count")

        # Topology that fits exactely the image
        tile = topology.tile(1, fake_builder)
        self.assertEqual(1, tile.identifier, "Topology, tile 1 : x offset")
        self.assertEqual(0, tile.offset_x, "Topology, tile 1 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 1 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 1 : width")
        self.assertEqual(300, tile.height, "Topology, tile 1 : height")

        tile = topology.tile(2, fake_builder)
        self.assertEqual(2, tile.identifier, "Topology, tile 2 : x offset")
        self.assertEqual(200, tile.offset_x, "Topology, tile 2 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 2 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 2 : width")
        self.assertEqual(300, tile.height, "Topology, tile 2 : height")

        tile = topology.tile(3, fake_builder)
        self.assertEqual(3, tile.identifier, "Topology, tile 3 : x offset")
        self.assertEqual(400, tile.offset_x, "Topology, tile 3 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 3 : y offset")
        self.assertEqual(200, tile.width, "Topology, tile 3 : width")
        self.assertEqual(300, tile.height, "Topology, tile 3 : height")

        tile = topology.tile(4, fake_builder)
        self.assertEqual(4, tile.identifier, "Topology, tile 4 : x offset")
        self.assertEqual(0, tile.offset_x, "Topology, tile 4 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 4 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 4 : width")
        self.assertEqual(250, tile.height, "Topology, tile 4 : height")

        tile = topology.tile(5, fake_builder)
        self.assertEqual(5, tile.identifier, "Topology, tile 5 : x offset")
        self.assertEqual(200, tile.offset_x, "Topology, tile 5 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 5 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 5 : width")
        self.assertEqual(250, tile.height, "Topology, tile 5 : height")

        tile = topology.tile(6, fake_builder)
        self.assertEqual(6, tile.identifier, "Topology, tile 6 : x offset")
        self.assertEqual(400, tile.offset_x, "Topology, tile 6 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 6 : y offset")
        self.assertEqual(200, tile.width, "Topology, tile 6 : width")
        self.assertEqual(250, tile.height, "Topology, tile 6 : height")

        # neighbours
        self.assertEqual(topology.tile_neighbours(1), (None, 4, None, 2))
        self.assertEqual(topology.tile_neighbours(2), (None, 5, 1, 3))
        self.assertEqual(topology.tile_neighbours(3), (None, 6, 2, None))
        self.assertEqual(topology.tile_neighbours(4), (1, None, None, 5))
        self.assertEqual(topology.tile_neighbours(5), (2, None, 4, 6))
        self.assertEqual(topology.tile_neighbours(6), (3, None, 5, None))