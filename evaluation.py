import constants as C
import numpy as np

class Evaluation():
    def __init__(self, simulation):
        self.simulation = simulation
        self.world = simulation.world
        self.organisms = simulation.organisms

    def evaluate_agent(self, organism, gamma):
        total_reward = 0
        food_count = 0
        death_count = 0
        survival_time = 0
        survival_times = []
        for i in range(1000):
            reward = organism.thoughtful_moment(self.world, organism.range, gamma, 0, False)
            total_reward += reward
            if reward == C.FOOD_REWARD:
                food_count += 1
            elif reward == C.DEATH_REWARD:
                death_count += 1
                organism.reset()
                survival_times.append(survival_time)
                survival_time = 0
                continue
            survival_time += 1


        mean_survival = np.mean(survival_times) if survival_times else None

        return total_reward, food_count, death_count, mean_survival

        
    def evaluate_episode(self, gamma):
        pass
        
    def evaluate_multiple_seeds(self):
        pass

