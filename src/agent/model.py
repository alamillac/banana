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

    def __init__(self, state_shape, action_size, seed):
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
            nn.Conv2d(state_shape[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(3136, 512), # 3136 = 64*7*7
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.backbone(state) # Extract features
        #x = self.avg_pool(x) # Use Global Average Pooling (GAP) Layer
        x = x.view(x.size(0), -1) # flatten all dimensions except batch
        x = self.head(x)
        return x
