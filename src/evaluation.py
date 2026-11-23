import random
import numpy as np
import torch

import constants as C
from organism import Organism
from simulation import Simulation
from world import World


class Evaluation:
    def __init__(self, simulation):
        self.simulation = simulation

    def evaluate_episode(self, simulation, max_steps=C.EVALUATION_STEPS):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if len(simulation.organisms) != 1:
            raise ValueError("Evaluation requires exactly one agent")
        agent = simulation.organisms[0]

        if not agent.alive:
            agent.reset()

        total_reward = 0
        food_count = 0
        steps = 0

        while agent.alive and steps < max_steps:
            reward = simulation.step(
                training=False,
                auto_reset=False
            )[0]

            total_reward += reward
            steps += 1

            if reward == C.FOOD_REWARD:
                food_count += 1

        return {
            "total_reward": total_reward,
            "food_count": food_count,
            "steps": steps,
            "died": not agent.alive
        }

    def evaluate_agent(self, simulation, steps=C.EVALUATION_STEPS):
        if steps <= 0:
            raise ValueError("steps must be positive")
        total_reward = 0
        food_count = 0
        death_count = 0
        survival_times = []
        elapsed_steps = 0

        while elapsed_steps < steps:
            episode = self.evaluate_episode(
                simulation,
                steps - elapsed_steps
            )

            total_reward += episode["total_reward"]
            food_count += episode["food_count"]
            elapsed_steps += episode["steps"]

            if episode["died"]:
                death_count += 1
                survival_times.append(episode["steps"])

        mean_survival = (
            float(np.mean(survival_times))
            if survival_times else None
        )

        return {
            "total_reward": total_reward,
            "food_count": food_count,
            "death_count": death_count,
            "mean_survival": mean_survival
        }

    def evaluate_multiple_seeds(self, organism, seeds, steps=C.EVALUATION_STEPS):
        if steps <= 0:
            raise ValueError("steps must be positive")
        results = {}

        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            world = World(
                self.simulation.world.height,
                self.simulation.world.width,
                seed=seed
            )
            world.generate_food(C.INITIAL_FOOD)

            agent = Organism(organism.start_position)
            agent.brain.load_state_dict(organism.brain.state_dict())
            agent.brain.eval()

            test_simulation = Simulation(world, [agent])

            results[seed] = self.evaluate_agent(
                test_simulation, steps
            )

        return results

        

