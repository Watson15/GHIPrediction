import torch
import torch.nn as nn
from torch.utils.data import Dataset

class GHIDataset(Dataset):
    def __init__(self, chunked_data, device='cuda'):
        super(GHIDataset, self).__init__()
        self.chunked_data = chunked_data.to(device) #chunked data needs to be passed in as a tensor
        self.device = device

    def __len__(self):
        return len(self.chunked_data)

    def __getitem__(self, i):
        element = self.chunked_data[i] #need to break down to zeroth element of list and first element of that is GHI of that which is the target
        # inputs = []
        # for el in element:
        #   inputs.append(el[0:-1])
        #print(element.shape)
        return element[0:-1], element[-1,0] #ensure first station in dataset is the one we are wanting to predict
    

class Main_LSTM(nn.Module):
    def __init__(self, dropout=0.0):
        """
        input_dim: Number of features per time step (here, 5)
        hidden_dim: Number of hidden units in the LSTM
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability (applied between LSTM layers if num_layers > 1)
        """
        super(Main_LSTM, self).__init__()
        self.Stationlstm = nn.LSTM(34, 128, 3,
                            batch_first=True, dropout=dropout)
        self.Auxlstm = nn.LSTM(8, 8, 2,
                            batch_first=True, dropout=dropout)#may need to change 16 back to 8
        # A fully-connected layer to map the LSTM output to a single GHI prediction
        self.station_fc = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        # LSTM returns output for every time step, and the final hidden and cell states.

        #torch.Size([32, 24, 64])
        main_station_data = x[:, :, :10]  # First 10 features are for main station
        aux_station_data = x[:, :, 10:]  # Remaining features are for auxiliary stations
        aux_lstm_outs = []
        #other idea: range start at 10 go to 34 add 8 each time as 3 auxillaries
        for i in range(0, 24, 8): #range start at 0 go to 24 add 8 each time as 3 auxillary stations
          lstm_out, _ = self.Auxlstm(aux_station_data[:, :, i:i+8])
          aux_lstm_outs.append(lstm_out)
        concat = torch.cat(([main_station_data]+aux_lstm_outs), 2)
        #print(concat.shape)
        lstm_out, (h_n, c_n) = self.Stationlstm(concat)
        # Use the output from the last time step as a summary of the sequence
        # lstm_out[:, -1, :] has shape (batch_size, hidden_dim)
        #print("Main: ", lstm_out.shape)
        final_feature = lstm_out[:, -1, :] #Not sure if this WORKSSSS (Right size?)
        #torch.Size([32, 64])

        # Pass through the fully-connected layer to produce a single output
        ghi_pred = self.station_fc(final_feature)  # shape: (batch_size, 1)
        return ghi_pred