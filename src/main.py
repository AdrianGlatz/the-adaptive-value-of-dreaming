from config import ExperimentConfig
from simulation import create_simulation
import visualisation


def main():
    config = ExperimentConfig()
    simulation = create_simulation(config)

    for step in range(config.training_steps):
        simulation.step()

        if (step + 1) % 1_000 == 0:
            print(f"Training: {step + 1} steps")

    visualisation.visualize_world(simulation)


if __name__ == "__main__":
    main()