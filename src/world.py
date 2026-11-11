import numpy as np

class World:

    def __init__(self, height, width):
        self.name = "World"
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)
        self.foods = set()

    def generate_food(self, amount):
        placed = 0

        if amount > self.grid.size - len(self.foods):
            raise ValueError("Not enough empty fields")

        while placed < amount:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)

            if (x, y) not in self.foods:
                placed += 1
                self.foods.add((x, y))
