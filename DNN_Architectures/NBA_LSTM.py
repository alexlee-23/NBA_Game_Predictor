import torch.nn as nn

class NBA_LSTM(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_layers=2, dropout=0.3):
        super(NBA_LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Binary classification
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch, seq_len, hidden_size]
        last_time_step = lstm_out[:, -1, :]  # take output at the last timestep
        output = self.fc(last_time_step)
        return output