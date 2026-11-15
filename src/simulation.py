gamma = 0.9

class Simulation:
    def __init__(self, world, organisms):
        self.world = world
        self.organisms = organisms
        self.steps = 0

    def step(self):
        epsilon = max(0.05, 1.0 - 0.95 * self.steps / 10_000)

        for organism in self.organisms:
            if not organism.alive:
                organism.reset()

            organism.thoughtful_moment(
                self.world, organism.range, gamma, epsilon
            )

        self.steps += 1