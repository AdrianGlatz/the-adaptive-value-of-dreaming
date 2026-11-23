from dataclasses import dataclass, field

@dataclass
class WorldConfig:
    width: int = 50
    height: int = 50
    initial_food: int = 250
    food_threshold: int = 240
    food_refill: int = 10

@dataclass
class OrganismConfig:
    max_energy: int = 100
    food_energy: int = 10
    move_energy: int = 1

    perception_radius: int = 2

@dataclass
class DQNConfig:
    gamma: float = 0.9
    learning_rate: float = 0.001
    batch_size: int = 32
    replay_capacity: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000

@dataclass
class RewardConfig:
    move: float = -0.05
    border: float = -0.6
    food: float = 2.0
    death: float = -3.0

@dataclass
class ExperimentConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    organism: OrganismConfig = field(default_factory=OrganismConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    training_steps: int = 10_000
    seed: int = 42
    evaluation_steps: int = 1000