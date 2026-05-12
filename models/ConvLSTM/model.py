import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config.hyperparameters.convlstm import *

# ======================================================
class WeatherDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = X
        self.y = y
        self.mask = mask

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()

        # padding to save map's size 5x5 + 1 -> 7x7 -> 3x3
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim+hidden_dim,
            4*hidden_dim,
            kernel_size,
            padding=padding,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        # x shape = (batch, channels, H, W)
        # h, c shape = (batch, hidden_dim, H, W)
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)

        i, f, o, g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(i) #input
        f = torch.sigmoid(f) #forget
        o = torch.sigmoid(o) #output
        g = torch.tanh(g) #candidate

        c_next = f*c + i*g
        h_next = o*torch.tanh(c_next)

        return h_next, c_next

# ConvLSTM model
class ConvLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, dropout=0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout2d(dropout)
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.output = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        # x shape = (batch, time, channels, H, W)
        batch, seq_len, C, H, W = x.shape
        h = torch.zeros(batch, self.hidden_dim, H, W, device=x.device)
        c = torch.zeros(batch, self.hidden_dim, H, W, device=x.device)

        for t in range(seq_len):
            x_t = self.dropout(x[:, t])
            h, c = self.cell(x_t, h, c)

        out = self.output(h)
        return out.squeeze(1) # shape (batch, H, W)