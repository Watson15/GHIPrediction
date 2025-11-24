import re

import numpy as np
from dataRetrieval import euclidean_distance

#from plotting import plot_stations_matplotlib

def getStationTimePeriodFromYears(csv_file):
  match = re.search(r'(\d{4})-(\d{4})', csv_file)
  if match:
    start_year = int(match.group(1))
    end_year = int(match.group(2))
    #print(start_year, end_year)  # Output: 2005 2017
    return (start_year, end_year)
  

def denormalizeHourOrMonth(anglesin, anglecos, cycleLength):
  angle = np.arctan2(anglesin, anglecos)
  cycleBeforeModulo = (angle / (2 * np.pi)) * cycleLength + 1
  return (cycleBeforeModulo % cycleLength)

def denormalize_non_cyclical(col, mean, std):
  denormalized_col = (col * std) + mean
  return denormalized_col

# Function to find the K nearest stations (not returning nearest station given)
def find_k_nearest_stations(df, station_name, k):
  if station_name not in df['station'].values:
      return f"Station {station_name} not found in the dataset."

  # Get the coordinates of the given station
  station_coords = df[df['station'] == station_name][['Latitude', 'Longitude']].values[0]
  lat1, lon1 = station_coords

  # Calculate the distance from the given station to all other stations
  df['distance'] = df.apply(lambda row: euclidean_distance(lat1, lon1, row['Latitude'], row['Longitude']), axis=1)

  # station_coords = df[df['station'] == station_name][['norm_lat', 'norm_long']].values[0]
  # lat1, lon1 = station_coords

  # # Calculate the distance from the given station to all other stations
  # df['distance'] = df.apply(lambda row: euclidean_distance(lat1, lon1, row['norm_lat'], row['norm_long']), axis=1)

  # Sort the DataFrame by distance and get the K nearest stations
  #nearest_stations = df[df['station'] != station_name].sort_values(by='distance').head(k)
  nearest_stations = df[df['station'] != station_name].sort_values(by='distance', key=lambda x: x.apply(lambda y: y[0])).head(k)

  return nearest_stations[['station', 'norm_lat', 'norm_long', 'distance', 'StartTime', 'EndTime']]


#To get station name print(str(wanted_station.station.iloc[0]))
def find_nearest_station_given_normed_long_lat(df,Longitude_Normed, Latitude_Normed):
  # Find the station with the smallest Euclidean distance to the given coordinates
  df['distance'] = df.apply(lambda row: euclidean_distance(Longitude_Normed, Latitude_Normed, row['norm_lat'], row['norm_long']), axis=1)
  nearest_station = df.sort_values(by='distance', key=lambda x: x.apply(lambda y: y[0])).head(1)
  return nearest_station[['station', 'norm_lat', 'norm_long', 'distance', 'StartTime', 'EndTime']]

  #To get station name print(str(wanted_station.station.iloc[0]))

def modify_nearest_stations(nearest_stations):
  modified_nearest_stations = []
  for station in nearest_stations['station']:
    modified_station = station.replace(" ", "-")
    modified_nearest_stations.append(modified_station)
  return modified_nearest_stations


def cyclicalEncoding(data, cycleLength):
  newDatasin = np.sin(2 * np.pi * (data - 1) / cycleLength)  # Sine encoding for hours (adjust for 0-23)
  newDatacos = np.cos(2 * np.pi * (data - 1) / cycleLength)  # Cosine encoding for hours (adjust for 0-23)
  return newDatasin, newDatacos



# def angle_from_wanted_station(row, wanted_station_long, wanted_station_lat):
#   print(row["station"])
#   lat, lon = row['Latitude'], row['Longitude']
#   delta_lon = lon - wanted_station_long
#   delta_lat = lat - wanted_station_lat
#   angle = np.arctan2(delta_lat, delta_lon)  # Angle in radians
#   angle_degrees = np.degrees(angle)
#   if angle_degrees < 0:
#     angle_degrees += 360  # Normalize to [0, 360]
#   return angle_degrees

# def sortAuxillaryStations(nearest_stations_df, wanted_station_lat, wanted_station_long):
#   """
#   From the auxillary stations found, sort them to have the first one be the eastern most station, the second be the western most station the third be the northern most station and the fourth be the southern most station
#   """

#   nearest_stations_df['angle'] = nearest_stations_df.apply(angle_from_wanted_station, axis=1, wanted_station_long=wanted_station_long, wanted_station_lat=wanted_station_lat)
#   sorted_stations = nearest_stations_df.sort_values(by='angle').reset_index(drop=True)
#   print("Sorted Stations by angle (0 degrees is east, 90 degrees is north, 180 degrees is west, 270 degrees is south):")
#   print(sorted_stations.head())
#   return sorted_stations
  

