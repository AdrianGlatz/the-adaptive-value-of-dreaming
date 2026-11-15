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
    def __init__(self, position):
        self.x = position[0]
        self.y = position[1]
        self.energy = 100
        self.alive = True
        self.brain = Brain()
        self.range = 2
        self.experience_buffer = deque(maxlen=1000)
        self.start_position = position

    def move(self, direction, world):
        if self.energy < 0 or not self.alive:
            return -1
        
        dx, dy = directions[direction]
        new_x = self.x + dx
        new_y = self.y + dy

        if 0 <= new_y < world.height and 0 <= new_x < world.width:
            self.x = new_x
            self.y = new_y


            self.energy -= 0
            if self.energy <= 0:
                self.alive = False
                return -1
            
            if (self.x, self.y) in world.foods:
                self.energy += 10
                world.foods.remove((self.x, self.y))
                return 1
            else:
                return -0.2

        else:
            self.energy -= 1
            if self.energy <= 0:
                self.alive = False
                return -1
            return -0.4


    def random_move(self, world):
        if self.energy < 0 or not self.alive:
            return
        
        direction = random.choice(list(directions.keys()))
        self.move(direction, world)


    def perceive(self, world, radius):
        if self.energy < 0 or not self.alive:
            return None
        situation = world.get_perception(self, radius)
        situation.append(self.energy/100)
        return situation


    def thoughtful_move(self, world, radius, epsilon):
        if self.energy <= 0 or not self.alive:
            return
        brain = self.brain
        situation = self.perceive(world, radius)
        situation_tensor = torch.tensor(situation, dtype=torch.float32)
        output = brain.choose_action(situation_tensor, epsilon)
        direction = list(directions.keys())[output]
        reward = self.move(direction, world)

        done = not self.alive

        if done:
            next_situation_tensor = torch.zeros_like(situation_tensor)
        else:
            next_situation_tensor = torch.tensor(self.perceive(world, radius), dtype=torch.float32)

        experience = Experience(state=situation_tensor, action=output, reward=reward, next_state=next_situation_tensor, done=not self.alive)
        self.experience_buffer.append(experience)
        return experience


    def thoughtful_moment(self, world, radius, gamma, epsilon):
        experience = self.thoughtful_move(world, radius, epsilon)

        if experience is None:
            return

        if len(self.experience_buffer) >= 32:
            batch = random.sample(list(self.experience_buffer), 32)
            self.brain.train_batch(batch, gamma)


    def reset(self):
        self.x = self.start_position[0]
        self.y = self.start_position[1]

        self.energy = 100

        self.alive = True




