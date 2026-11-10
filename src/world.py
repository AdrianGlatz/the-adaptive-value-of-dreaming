import numpy as np

class World:

    def __init__(self, height, width):
        self.name = "World"
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)

    def generate_food(self, amount):
        placed = 0

        if amount > np.count_nonzero(self.grid == 0):
            raise ValueError("Not enough empty fields")

        while placed < amount:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)

            if self.grid[y, x] == 0:
                self.grid[y, x] = 1
                placed += 1