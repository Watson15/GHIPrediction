import torch
import torch.nn as nn
from torch.utils.data import Dataset


class GHIDataset(Dataset):
    def __init__(self, chunked_data, device='cuda', station_data=[{"lat":0, "long":0, "elevation":0, "distance_to_main_station":0}]):
        super(GHIDataset, self).__init__()
        self.chunked_data = chunked_data.to(device) #chunked data needs to be passed in as a tensor
        self.device = device
        self.station_data = station_data

    def __len__(self):
        return len(self.chunked_data)

    def __getitem__(self, i):
        element = self.chunked_data[i] #need to break down to zeroth element of list and first element of that is GHI of that which is the target
        # inputs = []
        # for el in element:
        #   inputs.append(el[0:-1])
        #print(element.shape)
        return element[0:-1], element[-1,0] #ensure first station in dataset is the one we are wanting to predict
    
# Given lat, long, elevation, and distance to main station, 
# find how much this should effect other features like using wind speed, 
# and GHI from one station to effect next station next hour or not
# Will make a small NN for each feature that takes in the feature, the lat, long, elevation, and distance to main station
class LocationSpecificNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + 4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, feature, meta):
        # feature: (batch, seq, 1)
        # meta: (batch, 4)

        batch, seq, _ = feature.shape

        # reshape feature for linear layer
        feature_flat = feature.reshape(batch, seq)

        # meta must match (batch, seq, 4)
        meta_expanded = meta.unsqueeze(1).expand(batch, seq, 4)

        # feature_flat: (batch, seq)
        # we need (batch, seq, 1)
        feature_expanded = feature_flat.unsqueeze(2)

        inp = torch.cat([feature_expanded, meta_expanded], dim=2)

        # run dense layers
        out = self.net(inp)

        return out  # (batch, seq, 1)

class Main_LSTM(nn.Module):
    def __init__(self, dropout=0.0, num_aux_stations=3):
        """
        input_dim: Number of features per time step (here, 5)
        hidden_dim: Number of hidden units in the LSTM
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability (applied between LSTM layers if num_layers > 1)
        """
        super(Main_LSTM, self).__init__()

        self.num_aux_stations = num_aux_stations

        num_input = 10 + num_aux_stations*8

        self.Stationlstm = nn.LSTM(num_input, 128, 3,
                            batch_first=True, dropout=dropout)
        
        self.Auxlstm = nn.LSTM(8, 8, 2,
                            batch_first=True, dropout=dropout)#may need to change 16 back to 8
        # A fully-connected layer to map the LSTM output to a single GHI prediction
        self.station_fc = nn.Linear(128, 1)
        

    def forward(self, x, station_data):
        """
        x: (batch, seq, 10 + 8*num_aux)
        station_data: list of length num_aux, each (batch, 4)
        """

        batch, seq, feat = x.shape

        main_station_data = x[:, :, :10]
        aux_raw = x[:, :, 10:]

        aux_lstm_outs = []

        # ------------------------------------------
        # Process each auxiliary station independently
        # ------------------------------------------
        for s in range(self.num_aux_stations):

            start = s * 8
            end = start + 8

            aux_slice = aux_raw[:, :, start:end]  # (batch, seq, 8)
            corrected_features = aux_slice.clone() # So when editing in place we don't mess up original data

            # apply correction per feature
            for f in range(8):
                feature_tensor = aux_slice[:, :, f].unsqueeze(2)  # (batch, seq, 1)
                meta = station_data[s]  # (batch, 4)

                corr = self.LocationSpecificNN(feature_tensor, meta)  # (batch, seq, 1)
                corrected_features[:, :, f] += corr.squeeze(2)

            # run small LSTM
            lstm_out, _ = self.Auxlstm(corrected_features)
            aux_lstm_outs.append(lstm_out)

        # ------------------------------------------
        # Concatenate all aux LSTM outputs + main data
        # ------------------------------------------
        concat = torch.cat([main_station_data] + aux_lstm_outs, dim=2)

        # run main LSTM
        lstm_out, _ = self.Stationlstm(concat)

        # final hidden state from last time step
        final = lstm_out[:, -1, :]

        ghi_pred = self.station_fc(final)

        return ghi_pred
    
# class Main_LSTM(nn.Module):
#     def __init__(self, dropout=0.0, num_aux_stations=3):
#         """
#         input_dim: Number of features per time step (here, 5)
#         hidden_dim: Number of hidden units in the LSTM
#         num_layers: Number of stacked LSTM layers
#         dropout: Dropout probability (applied between LSTM layers if num_layers > 1)
#         """
#         super(Main_LSTM, self).__init__()

#         self.num_aux_stations = num_aux_stations

#         num_input = 10 + num_aux_stations*8

#         self.Stationlstm = nn.LSTM(num_input, 128, 3,
#                             batch_first=True, dropout=dropout)
        
#         self.Auxlstm = nn.LSTM(8, 8, 2,
#                             batch_first=True, dropout=dropout)#may need to change 16 back to 8
#         # A fully-connected layer to map the LSTM output to a single GHI prediction
#         self.station_fc = nn.Linear(128, 1)
        

#     def forward(self, x, station_data):
#         # x shape: (batch_size, sequence_length, input_dim)
#         # LSTM returns output for every time step, and the final hidden and cell states.

#         #torch.Size([32, 24, 64])
#         main_station_data = x[:, :, :10]  # First 10 features are for main station
#         aux_station_data = x[:, :, 10:]  # Remaining features are for auxiliary stations
#         aux_lstm_outs = []
#         #other idea: ex)range start at 10 go to 34 add 8 each time as 3 auxillaries
#         for i in range(0, self.num_aux_stations*8, 8): #range start at 0 go to 8*num_aux_stations add 8 each time as num_aux_stations auxillary stations              
#           lstm_out, _ = self.Auxlstm(aux_station_data[:, :, i:i+8])
#           aux_lstm_outs.append(lstm_out)
#         concat = torch.cat(([main_station_data]+aux_lstm_outs), 2)
#         #print(concat.shape)
#         lstm_out, (h_n, c_n) = self.Stationlstm(concat)
#         # Use the output from the last time step as a summary of the sequence
#         # lstm_out[:, -1, :] has shape (batch_size, hidden_dim)
#         #print("Main: ", lstm_out.shape)
#         final_feature = lstm_out[:, -1, :] #Not sure if this WORKSSSS (Right size?)
#         #torch.Size([32, 64])

#         # Pass through the fully-connected layer to produce a single output
#         ghi_pred = self.station_fc(final_feature)  # shape: (batch_size, 1)
#         return ghi_pred