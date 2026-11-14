
gamma = 0.99

class Simulation:
    def __init__(self, world, organisms):
        self.world = world
        self.organisms = organisms

    def step(self):
        for organism in self.organisms:
            organism.thoughtful_moment(self.world, organism.range, gamma)
