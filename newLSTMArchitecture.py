import torch
import torch.nn as nn
import torch.nn.functional as F

"""
High-level idea (short)

Train a shared LSTM (or stacked LSTM) that learns general temporal dynamics across all stations.

Give each station a small conditional network (station embedding + small MLP) that produces a low-dimensional vector that conditions the shared LSTM.

Condition the LSTM by either:

Concatenating the station embedding to each time-step input, or

Using FiLM-style modulation (learned scale & shift applied to layer activations), or

Using a learned initial hidden/cell state for the LSTM coming from the station network.

Train end-to-end on mixed batches of stations (so shared weights generalize), optionally fine-tune the small per-station network for each station later.
"""

class StationEncoder(nn.Module):
    def __init__(self, stat_in_dim, emb_dim=16, film_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stat_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
        )
        # optional FiLM outputs
        self.film_dim = film_dim
        if film_dim is not None:
            self.gamma = nn.Linear(emb_dim, film_dim)
            self.beta  = nn.Linear(emb_dim, film_dim)

    def forward(self, station_stats):
        # station_stats: (batch, stat_in_dim)
        emb = self.net(station_stats)               # (B, emb_dim)
        if self.film_dim:
            gamma = self.gamma(emb)
            beta  = self.beta(emb)
            return emb, gamma, beta
        return emb, None, None

class SharedLSTMForecast(nn.Module):
    def __init__(self, feat_dim, emb_dim, hidden=256, layers=2, film_dim=None):
        super().__init__()
        self.lstm_in_dim = feat_dim + emb_dim
        self.lstm = nn.LSTM(self.lstm_in_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # single-step forecast; change for multi-horizon
        )
        self.film_dim = film_dim
        if film_dim:
            # a small projector to match hidden dim if applying FiLM to hidden
            self.project = nn.Linear(film_dim, hidden)

    def forward(self, x_seq, emb, gamma=None, beta=None, init_states=None):
        # x_seq: (B, T, feat_dim)    ; emb: (B, emb_dim)
        B, T, _ = x_seq.shape
        emb_rep = emb.unsqueeze(1).expand(-1, T, -1)    # (B, T, emb_dim)
        lstm_input = torch.cat([x_seq, emb_rep], dim=-1) # (B, T, feat+emb)
        out, (hn, cn) = self.lstm(lstm_input, init_states)  # out: (B, T, hidden)
        # optional FiLM applied to final time step hidden
        last = out[:, -1, :]   # (B, hidden)
        if gamma is not None and beta is not None:
            # gamma/beta may be (B, film_dim) -> project to hidden
            g = self.project(gamma)
            b = self.project(beta)
            last = g * last + b
        pred = self.head(last)  # (B, 1)
        return pred.squeeze(-1)

# usage
stat_in_dim = 3   # lat, lon, elev (example)
feat_dim = 10     # features per time step
emb_dim = 16
film_dim = 16

station_enc = StationEncoder(stat_in_dim, emb_dim, film_dim)
model = SharedLSTMForecast(feat_dim, emb_dim, hidden=256, layers=2, film_dim=film_dim)

# forward for a batch:
# x_seq: (B, T, feat_dim), station_stats: (B, stat_in_dim)
emb, gamma, beta = station_enc(station_stats)
pred = model(x_seq, emb, gamma=gamma, beta=beta)
