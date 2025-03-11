import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # Apply CNN layers first
        x = self.conv1(x)  # Output will be [batch_size, 64, height, width]

        # Flatten the spatial dimensions to create a sequence
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1, channels)  # Reshape to [batch_size, sequence_length, channels]

        # Pass through LSTM
        x, _ = self.lstm1(x)  # LSTM input shape: [batch_size, sequence_length, channels]

        # Take the last output from the LSTM
        x = x[:, -1, :]  # Get the last time-step output
        x = self.fc(x)
        return x
