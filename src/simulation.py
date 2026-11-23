import numpy as np
import constants as C


class Simulation:
    def __init__(self, world, organisms):
        self.world = world
        self.organisms = organisms
        self.steps = 0

    def step(self, training=True, auto_reset=True):
        if training:
            epsilon = max(0.05, 1.0 - 0.95 * self.steps / 10_000)
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
                C.GAMMA,
                epsilon,
                training
            )

            rewards.append(result[1] if result is not None else None)

        self.steps += 1

        if len(self.world.foods) <= C.FOOD_THRESHOLD:
            self.world.generate_food(C.FOOD_REFILL)

        return rewards