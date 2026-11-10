
directions = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}


class Organism:
    def __init__(self, position):
        self.x = position[0]
        self.y = position[1]
        self.energy = 100
        self.alive = True

    def move(self, direction, world):
        if self.energy <= 0 or not self.alive:
            return
        
        dx, dy = directions[direction]
        new_x = self.x + dx
        new_y = self.y + dy

        if 0 <= new_y < world.height and 0 <= new_x < world.width:
            self.x = new_x
            self.y = new_y

            self.energy -= 1
            if self.energy <= 0:
                self.alive = False
                return

            if world.grid[self.y, self.x] == 1:
                self.energy += 10
                world.grid[self.y, self.x] = 0
        