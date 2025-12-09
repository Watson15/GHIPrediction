import torch
import torch.nn as nn
from torch.utils.data import Dataset


class GHIDataset(Dataset):
    """
    Dataset for Global Horizontal Irradiance (GHI) prediction.
    Stores chunked time-series data and station metadata.
    """
    def __init__(self, chunked_data, device='cuda', station_data=[{"lat":0, "long":0, "elevation":0, "distance_to_main_station":0}]):
        super(GHIDataset, self).__init__()
        # Move data to specified device (GPU/CPU)
        self.chunked_data = chunked_data.to(device)
        self.device = device
        # Metadata for each station (latitude, longitude, elevation, distance)
        self.station_data = station_data

    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.chunked_data)

    def __getitem__(self, i):
        """
        Returns a single training sample.
        Input: all timesteps except the last
        Target: GHI value at the last timestep for the main station
        """
        element = self.chunked_data[i]
        # Split into input sequence and target
        # element[0:-1] = all timesteps except last (input features)
        # element[-1, 0] = last timestep, first feature (target GHI)
        return element[0:-1], element[-1, 0]

    
class LocationSpecificNN(nn.Module):
    """
    Small neural network that adjusts features based on station location metadata.
    Takes a single feature value and station metadata (lat, long, elevation, distance)
    and outputs a correction factor to account for spatial differences between stations.
    
    OPTIMIZATION: Processes all 8 features at once instead of one at a time.
    """
    def __init__(self):
        super().__init__()
        # OPTIMIZED: Input 8 features + 4 metadata = 12 inputs (batch processing)
        # Hidden layer: 64 units (scales with more inputs)
        # Output: 8 corrections (one per feature)
        self.net = nn.Sequential(
            nn.Linear(8 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, features, meta):
        """
        Args:
            features: (batch, seq, 8) - All 8 features at once
            meta: (batch, 4) - Station metadata [lat, long, elevation, distance]
        
        Returns:
            out: (batch, seq, 8) - Correction values for all features
        """
        batch, seq, _ = features.shape

        # Expand metadata to match the sequence length dimension
        # meta: (batch, 4) -> (batch, seq, 4)
        meta_expanded = meta.unsqueeze(1).expand(batch, seq, 4)
        
        # Concatenate all features with metadata along the last dimension
        # Result: (batch, seq, 12) where 12 = 8 features + 4 metadata
        inp = torch.cat([features, meta_expanded], dim=2)
        
        # Pass through the neural network to get corrections for all features at once
        out = self.net(inp)  # (batch, seq, 8)
        
        return out


class Main_LSTM(nn.Module):
    """
    Main LSTM model for GHI prediction using data from multiple weather stations.
    
    Architecture:
    1. Main station data (10 features) is passed directly
    2. Auxiliary station data (8 features each) is:
       - Corrected using LocationSpecificNN based on spatial metadata (BATCH PROCESSED)
       - Processed through individual LSTMs
    3. All processed data is concatenated and fed to main LSTM
    4. Final prediction is made from the last LSTM hidden state
    
    OPTIMIZATIONS:
    - Batch process all 8 features per station simultaneously
    - Pre-allocate tensors where possible
    - Reduce loop iterations
    """
    def __init__(self, dropout=0.05, num_aux_stations=3):
        """
        Args:
            dropout: Dropout probability for regularization
            num_aux_stations: Number of auxiliary (secondary) weather stations
        """
        super(Main_LSTM, self).__init__()

        self.num_aux_stations = num_aux_stations
        
        # Total input features: 10 from main station + 8 per auxiliary station
        num_input = 10 + num_aux_stations * 8

        # Main LSTM processes combined data from all stations
        # Input: 10 (main) + num_aux*8 (auxiliary processed) features
        # Hidden: 128 units across 3 layers
        self.Stationlstm = nn.LSTM(num_input, 128, 3,
                                    batch_first=True, dropout=dropout)
        
        # Smaller LSTM for each auxiliary station
        # Processes the 8 corrected features from each auxiliary station
        self.Auxlstm = nn.LSTM(8, 8, 2,
                               batch_first=True, dropout=dropout)
        
        # OPTIMIZED: Create one LocationSpecificNN per auxiliary station
        # Each processes ALL 8 features at once (not one at a time)
        self.location_nns = nn.ModuleList([
            LocationSpecificNN() for _ in range(num_aux_stations)
        ])
        
        # Final fully-connected layer: maps LSTM output to single GHI prediction
        self.station_fc = nn.Linear(128, 1)
        

    def forward(self, x, station_data):
        """
        Forward pass through the model.
        
        Args:
            x: (batch, seq, 10 + 8*num_aux) - Input features
               First 10 features: main station data
               Remaining: auxiliary station data (8 features per station)
            station_data: List of length num_aux_stations
                         Each element is (batch, 4) containing [lat, long, elevation, distance]
        
        Returns:
            ghi_pred: (batch, 1) - Predicted GHI values
        """
        batch, seq, _ = x.shape

        # ============================================
        # STEP 1: Split input into main and auxiliary data
        # ============================================
        # Extract first 10 features (main station data)
        main_station_data = x[:, :, :10]  # (batch, seq, 10)
        
        # Extract remaining features (all auxiliary stations combined)
        aux_raw = x[:, :, 10:]  # (batch, seq, 8*num_aux)

        # OPTIMIZATION: Pre-allocate list with known size
        aux_lstm_outs = []

        # ============================================
        # STEP 2: Process each auxiliary station independently
        # ============================================
        for s in range(self.num_aux_stations):
            # Extract the 8 features for this specific auxiliary station
            start = s * 8
            end = start + 8
            aux_slice = aux_raw[:, :, start:end]  # (batch, seq, 8)

            # ============================================
            # STEP 3: Apply location-specific correction to ALL features at once
            # ============================================
            # OPTIMIZATION: Process all 8 features in a single forward pass
            # instead of looping through each feature individually
            meta = station_data[s]  # (batch, 4)
            
            # Apply location-specific neural network correction to all features
            corr = self.location_nns[s](aux_slice, meta)  # (batch, seq, 8)
            
            # Add correction to original features
            corrected_features = aux_slice + corr  # (batch, seq, 8)

            # ============================================
            # STEP 4: Process corrected data through auxiliary LSTM
            # ============================================
            # Each auxiliary station gets its own LSTM processing
            lstm_out, _ = self.Auxlstm(corrected_features)  # (batch, seq, 8)
            aux_lstm_outs.append(lstm_out)

        # ============================================
        # STEP 5: Combine all station data
        # ============================================
        # OPTIMIZATION: Single concatenation operation
        # Concatenate main station data with all processed auxiliary data
        # Result: (batch, seq, 10 + num_aux*8)
        concat = torch.cat([main_station_data] + aux_lstm_outs, dim=2)

        # ============================================
        # STEP 6: Process combined data through main LSTM
        # ============================================
        lstm_out, _ = self.Stationlstm(concat)  # (batch, seq, 128)

        # ============================================
        # STEP 7: Make final prediction
        # ============================================
        # Extract the final hidden state (last timestep)
        final = lstm_out[:, -1, :]  # (batch, 128)

        # Map to single GHI prediction value
        ghi_pred = self.station_fc(final)  # (batch, 1)

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