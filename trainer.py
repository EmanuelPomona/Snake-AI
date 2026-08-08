import torch
import torch.nn as nn


class QTrainer:
    def __init__(self, model, target_model, learning_rate, gamma):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
        action_tensor = torch.as_tensor(action)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32)
        done_tensor = torch.as_tensor(done, dtype=torch.bool)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            next_state_tensor = next_state_tensor.unsqueeze(0)

        batch_size = state_tensor.shape[0]

        if reward_tensor.dim() == 0:
            reward_tensor = reward_tensor.unsqueeze(0)

        if done_tensor.dim() == 0:
            done_tensor = done_tensor.unsqueeze(0)

        if action_tensor.dim() == 2:
            action_indices = torch.argmax(action_tensor, dim=1)
        elif action_tensor.dim() == 1 and action_tensor.numel() == 3 and batch_size == 1:
            action_indices = torch.argmax(action_tensor).unsqueeze(0)
        else:
            action_indices = action_tensor.long()

        predicted_q_values = self.model(state_tensor)
        predicted_action_q_values = predicted_q_values.gather(
            1,
            action_indices.long().unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_state_tensor)
            max_next_q_values = torch.max(next_q_values, dim=1).values
            target_q_values = reward_tensor + self.gamma * max_next_q_values
            target_q_values = torch.where(done_tensor, reward_tensor, target_q_values)

        loss = self.criterion(predicted_action_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
