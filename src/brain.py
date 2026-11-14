import random

import torch
import torch.nn as nn

class Brain(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(26, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

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
        losses = []
        for experience in batch:
            predicted_q_values = self(experience.state)
            with torch.no_grad():
                if experience.done:                    
                    target_q_values = experience.reward
                else:
                    target_q_values = experience.reward + gamma * torch.max(self(experience.next_state))
            loss = (predicted_q_values[experience.action] - target_q_values) ** 2
            losses.append(loss)

        average_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        average_loss.backward()
        self.optimizer.step()