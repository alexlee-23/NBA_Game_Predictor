import torch
import torch.nn as nn

class SiameseDualLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, num_layers=2, dropout=0.3):
        super(SiameseDualLSTM, self).__init__()
        
        # LSTM for primary team
        self.lstm_primary = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # LSTM for opposing team
        self.lstm_opposing = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers after concatenating both LSTM outputs
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, primary_input, opposing_input):
        # Primary team LSTM
        _, (primary_out, _) = self.lstm_primary(primary_input)
        
        # Opposing team LSTM
        _, (opposing_out, _) = self.lstm_opposing(opposing_input)
        
        # Concatenate the final hidden states from both LSTMs (batch_size, hidden_size*2)
        combined = torch.cat((primary_out[-1], opposing_out[-1]), dim=1)
        
        # Feed through fully connected layers
        x = self.fc1(combined)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        
        #Using BCE with Logits, no need for sigmoid
        return x

