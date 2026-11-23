import random

import torch
import torch.nn as nn

class Brain(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()

        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

    def choose_action(self, state, epsilon):

        if random.random() < epsilon:
            action = random.randrange(4)

        else:
            with torch.no_grad():
                q_values = self(state)
                action = torch.argmax(q_values).item()

        return action




    
    def train_batch(self, batch, gamma):
        states = torch.stack([exp.state for exp in batch])
        next_states = torch.stack([exp.next_state for exp in batch])

        actions = torch.tensor([exp.action for exp in batch], dtype=torch.long)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.bool)

        predicted_q_values = self(states)
        selected_q_values = predicted_q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self(next_states)
            max_next_q = next_q_values.max(dim=1).values
            targets = rewards + gamma * max_next_q * (~dones).float()

        loss = ((selected_q_values - targets) ** 2).mean()

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
