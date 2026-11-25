import pytest

from config import WorldConfig, OrganismConfig, DQNConfig, RewardConfig
from organism import Organism
from world import World


def create_organism(position=(2, 2)):
    return Organism(
        position,
        OrganismConfig(),
        DQNConfig(),
        RewardConfig()
    )


def create_world(width=5, height=5, seed=42):
    return World(
        WorldConfig(width=width, height=height),
        seed=seed
    )


class TestOrganism:

    def test_initialization(self):
        organism = create_organism((2, 3))

        assert organism.x == 2
        assert organism.y == 3
        assert organism.energy == OrganismConfig().max_energy
        assert organism.alive is True

    def test_move(self):
        organism = create_organism()
        world = create_world()

        organism.move("down", world)
        assert organism.y == 3

        organism.move("right", world)
        assert organism.x == 3

        assert organism.energy == (
            OrganismConfig().max_energy
            - 2 * OrganismConfig().move_energy
        )

        assert organism.move("down", world) == RewardConfig().move

    def test_border_collision(self):
        organism = create_organism((0, 0))
        world = create_world(width=3, height=3)

        reward = organism.move("up", world)

        assert reward == RewardConfig().border
        assert (organism.x, organism.y) == (0, 0)
        assert organism.energy == 99

    def test_eat_food(self):
        organism = create_organism()
        world = create_world()

        organism.energy = 50
        world.foods.add((2, 1))

        reward = organism.move("up", world)

        assert reward == RewardConfig().food
        assert organism.energy == 59
        assert (2, 1) not in world.foods

    def test_death(self):
        organism = create_organism()
        world = create_world()

        organism.energy = 1

        assert organism.move("down", world) == RewardConfig().death
        assert organism.alive is False

        with pytest.raises(RuntimeError):
            organism.move("right", world)

    def test_reset(self):
        organism = create_organism()
        world = create_world()

        organism.energy = 1
        organism.move("up", world)
        organism.reset()

        assert organism.energy == OrganismConfig().max_energy
        assert organism.alive is True
        assert (organism.x, organism.y) == organism.start_position

    def test_max_energy(self):
        organism = create_organism()
        world = create_world()

        world.foods.add((2, 1))
        reward = organism.move("up", world)

        assert reward == RewardConfig().food
        assert organism.energy == OrganismConfig().max_energy

    def test_perception(self):
        organism = create_organism()
        world = create_world()

        organism.energy = 50
        world.foods.add((3, 2))

        perception = organism.perceive(world, radius=2)

        assert len(perception) == 26
        assert perception[13] == 1
        assert perception[-1] == pytest.approx(0.5)

    def test_experience_buffer(self, monkeypatch):
        organism = create_organism()
        world = create_world()

        def always_move_right(state, epsilon):
            return 3

        monkeypatch.setattr(
            organism.brain,
            "choose_action",
            always_move_right
        )

        organism.thoughtful_move(
            world, radius=2, epsilon=0.0, training=True
        )

        assert len(organism.experience_buffer) == 1

        organism.thoughtful_move(
            world, radius=2, epsilon=0.0, training=False
        )

        assert len(organism.experience_buffer) == 1
