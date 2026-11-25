import random
import torch
from brain import Brain
from typing import NamedTuple
from collections import deque

directions = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}


class Experience(NamedTuple):
    state: object
    action: int
    reward: float
    next_state: object
    done: bool

class Organism:
    def __init__(self, position, config, dqn_config, reward_config):
        self.config = config
        self.dqn_config = dqn_config
        self.reward_config = reward_config

        self.x, self.y = position
        self.start_position = position
        self.energy = config.max_energy
        self.alive = True

        self.range = config.perception_radius
        input_size = (2 * config.perception_radius + 1) ** 2 + 1
        self.brain = Brain(dqn_config, input_size)
        self.experience_buffer = deque(maxlen=dqn_config.replay_capacity)

    def move(self, direction, world):
        if not self.alive or self.energy <= 0:
            raise RuntimeError("Cannot move a dead organism")
        
        dx, dy = directions[direction]
        new_x = self.x + dx
        new_y = self.y + dy

        if 0 <= new_y < world.height and 0 <= new_x < world.width:
            self.x = new_x
            self.y = new_y


            self.energy -= self.config.move_energy
            
            if (self.x, self.y) in world.foods:
                self.energy = min(
                    self.config.max_energy, 
                    self.energy + self.config.food_energy
                )
                world.foods.remove((self.x, self.y))
                reward = self.reward_config.food
            else:
                reward = self.reward_config.move

            if self.energy <= 0:
                self.alive = False
                return self.reward_config.death

            return reward

        else:
            self.energy -= self.config.move_energy
            if self.energy <= 0:
                self.alive = False
                return self.reward_config.death
            return self.reward_config.border


    def random_move(self, world):
        if self.energy < 0 or not self.alive:
            return
        
        direction = random.choice(list(directions.keys()))
        self.move(direction, world)


    def perceive(self, world, radius):
        if self.energy < 0 or not self.alive:
            return None
        situation = world.get_perception(self, radius)
        situation.append(self.energy/self.config.max_energy)
        return situation


    def thoughtful_move(self, world, radius, epsilon, training=True):
        if self.energy <= 0 or not self.alive:
            return None

        situation = self.perceive(world, radius)
        situation_tensor = torch.tensor(situation, dtype=torch.float32)

        action = self.brain.choose_action(situation_tensor, epsilon)
        direction = list(directions.keys())[action]

        reward = self.move(direction, world)
        done = not self.alive

        if done:
            next_state = torch.zeros_like(situation_tensor)
        else:
            next_state = torch.tensor(
                self.perceive(world, radius), dtype=torch.float32
            )

        experience = Experience(
            situation_tensor, action, reward, next_state, done
        )

        if training:
            self.experience_buffer.append(experience)

        return experience, reward


    def thoughtful_moment(self, world, radius, gamma, epsilon, training=True):
        result = self.thoughtful_move(world, radius, epsilon, training)

        if result is None:
            return None

        experience, reward = result

        if training and len(self.experience_buffer) >= self.dqn_config.batch_size:
            batch = random.sample(list(self.experience_buffer), self.dqn_config.batch_size)
            self.brain.train_batch(batch, gamma)

        return experience, reward

    def reset(self):
        self.x = self.start_position[0]
        self.y = self.start_position[1]

        self.energy = self.config.max_energy

        self.alive = True




