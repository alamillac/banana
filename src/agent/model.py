import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)


class QNetworkVisual(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_shape, action_size, seed, dropout=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_shape ((int,int,int,int)): Dimension of each state (batch, channels, height, width)
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(state_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2), # 80 -> 40
            nn.ReLU(),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2), # 40 -> 20
            nn.ReLU(),
            nn.Dropout2d(dropout),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2), # 20 -> 10
            nn.ReLU(),
            nn.Dropout2d(dropout),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2), # 10 -> 5
            nn.ReLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(), # flatten all dimensions except batch

            # Block
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            # Block
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block
            nn.Linear(512, action_size),
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.backbone(state) # Extract features
        x = self.avg_pool(x) # Use Global Average Pooling (GAP) Layer
        x = self.head(x)
        return x
