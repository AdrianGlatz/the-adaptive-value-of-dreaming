import pygame
import simulation



def draw_simulation(world, organisms, screen):
    for food in world.foods:
        pygame.draw.rect(screen,
                         (0, 200, 0),
                            (food[0] * 16, food[1] * 16, 16, 16))

    for organism in organisms:
        pygame.draw.rect(screen,
                         (0, 0, 200),
                            (organism.x * 16, organism.y * 16, 16, 16))


def visualize_world(simulation):
    world = simulation.world
    organisms = simulation.organisms
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("Organism Simulation")

    running = True
    clock = pygame.time.Clock()
    elapsed = 0
    interval = 1/6

    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        screen.fill((255, 255, 255))
        draw_simulation(world, organisms, screen)
        pygame.display.flip()
        dt = clock.tick(60) / 1000.0
        elapsed += dt
        if elapsed >= interval:
            if any(organism.alive for organism in organisms):
                simulation.step()

            elapsed -= interval
    pygame.quit()

