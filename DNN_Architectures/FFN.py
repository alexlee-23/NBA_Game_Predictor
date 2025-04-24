import torch.nn as nn

class FFN(nn.Module): 
    def __init__(self, input_size=420):
        super(FFN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),  # Increase to 512 neurons
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            
            nn.Linear(128, 1)
            )


    def forward(self, x):
        return self.model(x)