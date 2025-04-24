import torch
import torch.nn as nn

class TeamFFN(nn.Module):
    def __init__(self, input_size=210):
        super(TeamFFN, self).__init__()
        self.subnet = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.subnet(x)

class DualTeamFFN(nn.Module):
    def __init__(self, input_team_size=210):
        super(DualTeamFFN, self).__init__()
        self.primary_net = TeamFFN(input_size=input_team_size)
        self.opposing_net = TeamFFN(input_size=input_team_size)

        self.final_layers = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # binary classification
        )

    def forward(self, x):
        # Split the 420-dim input into two 210-dim parts
        x_primary = x[:, :210]
        x_opposing = x[:, 210:]

        h_primary = self.primary_net(x_primary)
        h_opposing = self.opposing_net(x_opposing)

        combined = torch.cat([h_primary, h_opposing], dim=1)
        out = self.final_layers(combined)
        return out