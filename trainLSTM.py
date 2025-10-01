import glob  # For loading multiple files

#import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#import torch.nn.functional as F
import torch.optim as optim
from constants import DTYPE_NON_CLOUD, USECOLS_NON_CLOUD
from LSTMArchitecture import GHIDataset, Main_LSTM
from processAllData import getAllReadyForStationByLatAndLongAndK
from torch.utils.data import DataLoader
from tqdm import tqdm


def test_model(model, test_loader, criterion, device):
    # if(combined_test_chunked_data_tensor.shape):
    #   dataset = GHIDataset(combined_test_chunked_data_tensor, device=device)
    #   train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # dataset = GHIDataset(combined_chunked_data_tensor, device=device)
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.eval()
    outputlist = []
    targetData=[]
    colourList = []
    for inputs, targets in test_loader:
        output = model(inputs)
        colourList.append(inputs[:,-1].detach().cpu().numpy())#Saving hour before predicted hour
        targetData.append(targets.detach().cpu().numpy())
        outputlist.append(output.detach().cpu().numpy())
    colourList = np.concatenate(colourList[:-1], axis=0)#Last batch size is too small so getting rid of it with the -1
    outputlist = np.concatenate(outputlist[:-1], axis=0)
    targetData = np.concatenate(targetData[:-1], axis=0)
    #del combined_chunked_data_tensor #Saving RAM space
    return outputlist, targetData, colourList

def main():
    path = 'Datasets/CWEEDS_2020_BC_raw'
    csv_files = glob.glob(os.path.join(path, "*.csv"))

    file_path = 'Datasets/stationsName_lat_long_data.csv'
    num_aux_stations = 4
    stationsName_lat_long_datadf = pd.read_csv(file_path, delimiter=',', on_bad_lines='skip')
    wanted_chunked_tensors, wanted_station_name, aux_chunked_tensors, aux_chunked_station_order, meanGHI1, stdGHI1 = getAllReadyForStationByLatAndLongAndK(stationsName_lat_long_datadf.copy(), -131.75, 54.5, num_aux_stations, csv_files, usecols=USECOLS_NON_CLOUD, dtype=DTYPE_NON_CLOUD)
    combined_chunked_data_tensor1 = torch.cat(wanted_chunked_tensors + aux_chunked_tensors, dim=2)
    wanted_chunked_tensors, wanted_station_name, aux_chunked_tensors, aux_chunked_station_order, meanGHI2, stdGHI2 = getAllReadyForStationByLatAndLongAndK(stationsName_lat_long_datadf.copy(), -123, 50,num_aux_stations, csv_files, usecols=USECOLS_NON_CLOUD, dtype=DTYPE_NON_CLOUD)
    combined_chunked_data_tensor2 = torch.cat(wanted_chunked_tensors + aux_chunked_tensors, dim=2)
    wanted_chunked_tensors, wanted_station_name, aux_chunked_tensors, aux_chunked_station_order, meanGHI3, stdGHI3 = getAllReadyForStationByLatAndLongAndK(stationsName_lat_long_datadf.copy(), -123.5, 48,num_aux_stations, csv_files, usecols=USECOLS_NON_CLOUD, dtype=DTYPE_NON_CLOUD)
    combined_chunked_data_tensor3 = torch.cat(wanted_chunked_tensors + aux_chunked_tensors, dim=2)
    wanted_chunked_tensors, wanted_station_name, aux_chunked_tensors, aux_chunked_station_order, meanGHI4, stdGHI4 = getAllReadyForStationByLatAndLongAndK(stationsName_lat_long_datadf.copy(), -122.5, 60,num_aux_stations, csv_files, usecols=USECOLS_NON_CLOUD, dtype=DTYPE_NON_CLOUD)
    combined_chunked_data_tensor4 = torch.cat(wanted_chunked_tensors + aux_chunked_tensors, dim=2)
    wanted_chunked_tensors, wanted_station_name, aux_chunked_tensors, aux_chunked_station_order, meanGHI5, stdGHI5 = getAllReadyForStationByLatAndLongAndK(stationsName_lat_long_datadf.copy(), -127.75, 51,num_aux_stations, csv_files, usecols=USECOLS_NON_CLOUD, dtype=DTYPE_NON_CLOUD)
    combined_chunked_data_tensor5 = torch.cat(wanted_chunked_tensors + aux_chunked_tensors, dim=2)
    combined_chunked_data_tensor = torch.cat([combined_chunked_data_tensor1, combined_chunked_data_tensor2, combined_chunked_data_tensor3, combined_chunked_data_tensor4, combined_chunked_data_tensor5], dim=0)

    mean_MeansGHI = np.mean([meanGHI1, meanGHI2, meanGHI3, meanGHI4, meanGHI5])
    std_MeansGHI = np.mean([stdGHI1, stdGHI2, stdGHI3, stdGHI4, stdGHI5])
   
    #Deleting all tensors and data no longer needed to save space on RAM
    del wanted_chunked_tensors, wanted_station_name, aux_chunked_tensors, aux_chunked_station_order
    del combined_chunked_data_tensor1, combined_chunked_data_tensor2, combined_chunked_data_tensor3, combined_chunked_data_tensor4, combined_chunked_data_tensor5
    # Check if GPU is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    #dataset = GHIDataset(combined_chunked_data_tensor, device=device)
    dataset = GHIDataset(combined_chunked_data_tensor, device=device)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    mainModel = Main_LSTM(num_aux_stations=num_aux_stations).to(device)
    mainModel = torch.compile(mainModel)
    #mainModel = Main_LSTM()
    criterion = nn.MSELoss()  # Mean Squared Error is common for regression tasks
    optimizer_main = optim.Adam(mainModel.parameters(), lr=0.001)
    num_epochs = 100
    loss_per_epoch = []
    for epoch in tqdm(range(num_epochs)):
        mainModel.train()
        #epoch_lossList = []
        epoch_loss = 0.0
        for inputs, target in train_loader:
            # Move data to the same device as the model
            #inputs = inputs.to(device)      # shape: (batch_size, 24, 6)
            #target = target.to(device)      # shape: (batch_size,)

            optimizer_main.zero_grad()

            # Forward pass: predict GHI
            output = mainModel(inputs)          # shape: (batch_size, 1)

            # Compute loss (unsqueeze target if necessary to match output shape)
            loss = criterion(output, target.unsqueeze(1))

            # Backward pass and optimization
            loss.backward()
            optimizer_main.step()
            epoch_loss += loss.item() * inputs.size(0)

            #epoch_lossList.append(loss.item())

        epoch_loss /= len(dataset)
        loss_per_epoch.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.8f}")
    # Saving the model to the correct folder. Use the correct naming schema outlined in the README
    from pathlib import Path  #For loading a specific file
    try: 
        # Current working directory
        current_directory = Path().resolve()
        
        # Path to Models directory
        models_dir = current_directory / "Models"
        
        # Make sure the directory exists
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Final save path
        save_path = str(models_dir / "model_3_5_1.pth")
        
        print(f"Model will be saved to: {save_path}")
        torch.save(mainModel.state_dict(), save_path)
        
    except Exception:
        # Current working directory
        current_directory = Path().resolve()
        
        # Path to GHIPrediction/Models
        relative_path = Path("GHIPrediction") / "Models"
        models_dir = current_directory / relative_path
        
        # Make sure the directory exists
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Final save path
        save_path = str(models_dir / "model_3_5_1.pth")
        
        print(f"Model will be saved to: {save_path}")
        torch.save(mainModel.state_dict(), save_path)
        
    except Exception as e:
        print("Error occurred while saving the model:", e)

    try:
        import matplotlib.pyplot as plt
        outputlist, targetData, colourList = test_model(mainModel, train_loader, criterion, device)
        
        targetGHI = targetData*std_MeansGHI + mean_MeansGHI
        outputGHI = outputlist*std_MeansGHI + mean_MeansGHI #normalized
        colourListGHI = colourList[:,0]*std_MeansGHI + mean_MeansGHI
        # angle_rad = np.arcsin(colourList[:,:,6])# 4 For hours. Can also do 6 for months
        # angle_deg = np.degrees(angle_rad)
        # hour = (angle_deg + 90) * 24 / 180  # Map angle from [-90, 90] to [0, 24]
        # month = (angle_deg + 90) * 12 / 180  # Map angle from [-90, 90] to [0, 12]
        plt.scatter(targetGHI.flatten(), outputGHI.flatten()-targetGHI.flatten(), c=colourList[:,10].flatten(), alpha=0.25)
        #plt.scatter(targetData.flatten(), outputlist.flatten()-targetData.flatten(), c=colourList[:,10].flatten(), alpha=0.25)
        plt.axis('equal')
        plt.colorbar(label="Auxiliary station 1 GHI")
        plt.xlabel('Actual GHI')
        plt.ylabel('Difference')
        plt.title('LSTM Difference Plot') # from training on ROSE-SPIT-(AUT), WHISTLER---NESTERS, ESQUIMALT-HARBOUR, FORT-NELSON-A and HERBERT-ISLAND-(AUT)
        plt.grid()
        plt.savefig("LSTM_Difference_Plot_3_5_1.png")
        plt.show()

        plt.scatter(targetGHI.flatten(), outputGHI.flatten(), c=colourList[:,10].flatten(), alpha=0.25)
        #plt.scatter(targetData.flatten(), outputlist.flatten(), c=colourList[:,10].flatten(), alpha=0.25)
        plt.axis('equal')
        plt.colorbar(label="Auxiliary station 1 GHI")
        plt.xlabel('Actual GHI')
        plt.ylabel('Predicted GHI')
        plt.title('LSTM Predicted vs Actual GHI Plot')# from training on ROSE-SPIT-(AUT), WHISTLER---NESTERS, ESQUIMALT-HARBOUR, FORT-NELSON-A and HERBERT-ISLAND-(AUT)
        plt.grid()
        plt.savefig("LSTM_Predicted_vs_Actual_GHI_Plot_3_5_1.png")
        plt.show()
    except Exception as e:
        print("Error occurred while testing the model:", e)

if __name__=="__main__":
    main()