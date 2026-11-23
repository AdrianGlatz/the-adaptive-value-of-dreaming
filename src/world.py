import numpy as np

class World:

    def __init__(self, config, seed=None):
        self.name = "World"
        self.height = config.height
        self.width = config.width
        self.grid = np.zeros((config.height, config.width), dtype=int)
        self.foods = set()

        self.seed = seed
        self.rng = np.random.default_rng(seed)


    def generate_food(self, amount):
        placed = 0

        if amount > self.grid.size - len(self.foods):
            raise ValueError("Not enough empty fields")

        while placed < amount:
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))

            if (x, y) not in self.foods:
                placed += 1
                self.foods.add((x, y))


    def get_perception(self, organism, radius):
        perception = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = organism.x + dx
                y = organism.y + dy
                if  not 0 <= x < self.width or not 0 <= y < self.height:
                    perception.append(-1)
                elif (x, y) in self.foods:
                    perception.append(1)
                else:
                    perception.append(0)
        return perception
