import matplotlib.pyplot as plt


def plot_stations_matplotlib(center_latlon, neighbors_df, all_stations_df, figsize=(8, 6)):
    center_lat, center_lon = center_latlon

    # Create set of picked station names for easy lookup
    picked_names = set(neighbors_df['station'])

    plt.figure(figsize=figsize)

    # Plot unpicked stations (from all_stations_df, excluding the neighbors and center)
    for _, row in all_stations_df.iterrows():
        if (row['Latitude'], row['Longitude']) == tuple(center_latlon):
            continue
        if row['station'] not in picked_names:
            plt.scatter(row['Longitude'], row['Latitude'], color='gray', alpha=0.9, label='_nolegend_')
            #plt.text(row['Longitude'] + 0.002, row['Latitude'], row['station'], fontsize=8, color='gray')

    # Plot the selected neighbor stations
    plt.scatter(
        neighbors_df['Longitude'],
        neighbors_df['Latitude'],
        color='blue',
        label='Picked Neighbors'
    )
    for _, row in neighbors_df.iterrows():
        plt.text(row['Longitude'] -1.5, row['Latitude']+0.25, row['station'], fontsize=9)
    print("Plotted neighbors:")

    # Plot the target station
    plt.scatter(center_lon, center_lat, color='red', marker='*', s=150, label='Target Station')
    #plt.text(center_lon + 0.002, center_lat, 'Target', fontsize=10, fontweight='bold')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Target Station and Spatially Diverse Neighbors')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()