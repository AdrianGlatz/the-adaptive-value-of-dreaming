import numpy as np
from world import World
from organism import Organism

world = World(50, 50)
world.generate_food(10)
organism = Organism((25, 25))
world.grid[24, 25] = 1
print(f"Initial position: ({organism.x}, {organism.y}), Energy: {organism.energy}, Alive: {organism.alive}")
organism.move("up", world)
organism.move("left", world)
organism.move("down", world)
print(f"Final position: ({organism.x}, {organism.y}), Energy: {organism.energy}, Alive: {organism.alive}")
