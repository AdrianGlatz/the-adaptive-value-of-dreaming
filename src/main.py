import numpy as np
from simulation import Simulation
from world import World
from organism import Organism
import visualisation

world = World(50, 50)
world.generate_food(150)
organisms = [Organism((25, 25)), Organism((10, 10)), Organism((40, 40))]
simulation = Simulation(world, organisms)
for step in range(10_000):
    simulation.step()

    if (step + 1) % 1_000 == 0:
        print(f"Training: {step + 1}/100000")

visualisation.visualize_world(simulation)
