from simulation import create_simulation
import visualisation


def main():
    simulation = create_simulation()
    for step in range(10_000):
        simulation.step()

        if (step + 1) % 1_000 == 0:
            print(f"Training: {step + 1} steps")

    visualisation.visualize_world(simulation)


if __name__ == "__main__":
    main()
