import numpy as np
from simulation import Simulation
from world import World
from organism import Organism
import visualisation

world = World(50, 50)
world.generate_food(150)
organisms = [Organism((25, 25)), Organism((10, 10)), Organism((40, 40))]
simulation = Simulation(world, organisms)
visualisation.visualize_world(simulation)
