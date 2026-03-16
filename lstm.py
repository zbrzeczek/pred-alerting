import torch.nn as nn

class LstmModel(nn.Module):
    def __init__(self, 
                 input_size=36, # input size to 12 landmarkow x 3 wymiary
                 hidden_size=128, 
                 num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        _, (h_n, _) = self.lstm(x)  # h_n shape: (num_layers, batch, hidden_size)
        h_last = h_n[-1]  # take last layer hidden state
        return self.fc(h_last)  # output shape: (batch, num_classes)