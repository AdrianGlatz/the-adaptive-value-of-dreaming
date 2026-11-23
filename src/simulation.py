import random
import torch

from config import ExperimentConfig
from world import World
from organism import Organism


class Simulation:
    def __init__(self, world, organisms, config):
        self.world = world
        self.organisms = organisms
        self.config = config
        self.steps = 0

    def step(self, training=True, auto_reset=True):
        if training:
            dqn = self.config.dqn

            epsilon = max(
                dqn.epsilon_end,
                dqn.epsilon_start
                - (dqn.epsilon_start - dqn.epsilon_end)
                * self.steps / dqn.epsilon_decay_steps
            )
        else:
            epsilon = 0.0

        rewards = []

        for organism in self.organisms:
            if not organism.alive:
                if auto_reset:
                    organism.reset()
                else:
                    rewards.append(None)
                    continue

            result = organism.thoughtful_moment(
                self.world,
                organism.range,
                self.config.dqn.gamma,
                epsilon,
                training
            )

            rewards.append(result[1] if result is not None else None)

        self.steps += 1

        if len(self.world.foods) <= self.config.world.food_threshold:
            self.world.generate_food(self.config.world.food_refill)

        return rewards


def create_simulation(config: ExperimentConfig):
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    world = World(config.world, seed=config.seed)
    world.generate_food(config.world.initial_food)

    position = (
        world.width // 2,
        world.height // 2
    )

    organism = Organism(
        position,
        config.organism,
        config.dqn,
        config.reward
    )

    return Simulation(world, [organism], config)