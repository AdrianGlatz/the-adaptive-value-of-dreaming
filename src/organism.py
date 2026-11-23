import random
import torch
from brain import Brain
from typing import NamedTuple
from collections import deque
import constants as C

directions = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}



MOVE_ENERGY = 1
FOOD_ENERGY = 10
MAX_ENERGY = 100

class Experience(NamedTuple):
    state: object
    action: int
    reward: float
    next_state: object
    done: bool

class Organism:
    def __init__(self, position):
        self.x = position[0]
        self.y = position[1]
        self.energy = MAX_ENERGY
        self.alive = True
        self.brain = Brain()
        self.range = 2
        self.experience_buffer = deque(maxlen=1000)
        self.start_position = position

    def move(self, direction, world):
        if self.energy < 0 or not self.alive:
            return C.DEATH_REWARD
        
        dx, dy = directions[direction]
        new_x = self.x + dx
        new_y = self.y + dy

        if 0 <= new_y < world.height and 0 <= new_x < world.width:
            self.x = new_x
            self.y = new_y


            self.energy -= MOVE_ENERGY
            
            if (self.x, self.y) in world.foods:
                self.energy = min(MAX_ENERGY, self.energy + FOOD_ENERGY)
                world.foods.remove((self.x, self.y))
                reward = C.FOOD_REWARD
            else:
                reward = C.MOVE_REWARD

            if self.energy <= 0:
                self.alive = False
                return C.DEATH_REWARD

            return reward

        else:
            self.energy -= MOVE_ENERGY
            if self.energy <= 0:
                self.alive = False
                return C.DEATH_REWARD
            return C.BORDER_REWARD


    def random_move(self, world):
        if self.energy < 0 or not self.alive:
            return
        
        direction = random.choice(list(directions.keys()))
        self.move(direction, world)


    def perceive(self, world, radius):
        if self.energy < 0 or not self.alive:
            return None
        situation = world.get_perception(self, radius)
        situation.append(self.energy/MAX_ENERGY)
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

        if training and len(self.experience_buffer) >= 32:
            batch = random.sample(list(self.experience_buffer), 32)
            self.brain.train_batch(batch, gamma)

        return experience, reward

    def reset(self):
        self.x = self.start_position[0]
        self.y = self.start_position[1]

        self.energy = MAX_ENERGY

        self.alive = True




