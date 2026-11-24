from types import SimpleNamespace

import pytest

from config import WorldConfig
from world import World


class TestWorld:

    def test_initialization(self):
        config = WorldConfig(width=10, height=5)
        world = World(config)

        assert world.width == 10
        assert world.height == 5
        assert world.grid.shape == (5, 10)
        assert world.foods == set()

    def test_generate_food_count(self):
        config = WorldConfig(width=5, height=5)
        world = World(config, seed=42)

        world.generate_food(6)
        assert len(world.foods) == 6

        world.generate_food(4)
        assert len(world.foods) == 10

    def test_food_positions_within_bounds(self):
        config = WorldConfig(width=7, height=4)
        world = World(config, seed=42)

        world.generate_food(15)

        for x, y in world.foods:
            assert 0 <= x < config.width
            assert 0 <= y < config.height

    def test_food_capacity_exceeded(self):
        config = WorldConfig(width=3, height=3)
        world = World(config, seed=42)

        world.generate_food(9)
        assert len(world.foods) == 9

        with pytest.raises(ValueError):
            world.generate_food(1)

    def test_food_seed_reproducibility(self):
        config = WorldConfig(width=6, height=6)

        world1 = World(config, seed=42)
        world2 = World(config, seed=42)

        world1.generate_food(14)
        world2.generate_food(14)

        assert world1.foods == world2.foods

    def test_perception_empty_world(self):
        world = World(WorldConfig(width=5, height=5))
        agent = SimpleNamespace(x=2, y=2)

        perception = world.get_perception(agent, radius=1)

        assert perception == [0] * 9

    def test_perception_detects_food(self):
        world = World(WorldConfig(width=5, height=5))
        agent = SimpleNamespace(x=2, y=2)

        world.foods.add((3, 2))

        perception = world.get_perception(agent, radius=1)

        assert perception == [
            0, 0, 0,
            0, 0, 1,
            0, 0, 0
        ]

    def test_perception_at_boundary(self):
        world = World(WorldConfig(width=5, height=5))
        agent = SimpleNamespace(x=0, y=0)

        perception = world.get_perception(agent, radius=1)

        assert perception == [
            -1, -1, -1,
            -1,  0,  0,
            -1,  0,  0
        ]
