import random
import numpy as np
import torch

from organism import Organism
from simulation import Simulation
from world import World


class Evaluation:
    def __init__(self, simulation):
        self.simulation = simulation

    def evaluate_episode(self, simulation, max_steps=None):
        if max_steps is None:
            max_steps = simulation.config.evaluation_steps
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

            if reward == simulation.config.reward.food:
                food_count += 1

        return {
            "total_reward": total_reward,
            "food_count": food_count,
            "steps": steps,
            "died": not agent.alive
        }

    def evaluate_agent(self, simulation, steps=None):
        if steps is None:
            steps = simulation.config.evaluation_steps
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

    def evaluate_multiple_seeds(self, organism, seeds, steps=None):
        if steps is None:
            steps = self.simulation.config.evaluation_steps
        results = {}

        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            world = World(self.simulation.config.world, seed=seed)
            world.generate_food(self.simulation.config.world.initial_food)

            agent = Organism(
                organism.start_position,
                self.simulation.config.organism,
                self.simulation.config.dqn,
                self.simulation.config.reward
            )
            agent.brain.load_state_dict(organism.brain.state_dict())
            agent.brain.eval()

            test_simulation = Simulation(
                world,
                [agent],
                self.simulation.config
            )

            results[seed] = self.evaluate_agent(test_simulation, steps)

        return results

        

