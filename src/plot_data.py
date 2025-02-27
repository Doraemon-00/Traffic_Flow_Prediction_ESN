import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the DADAS dataset
dadas_file_path = r'C:\Users\doraemon\Documents\GitHub\Traffic_Flow_Prediction_ESN\data\raw\DADAS.csv'
dadas_data = pd.read_csv(dadas_file_path)

# Convert the 'date' column to datetime format, handling mixed formats
dadas_data['date'] = pd.to_datetime(dadas_data['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Drop rows with NaT in 'date' column
dadas_data = dadas_data.dropna(subset=['date'])

# Filter for a specific sensor
sensor_id = 43013  # Specific
# sensor_id = dadas_data['id'].iloc[0]  # Or the first sensor ID
sensor_data = dadas_data[dadas_data['id'] == sensor_id]

# Sort by date and get the first date
sensor_data = sensor_data.sort_values(by='date')
start_date = sensor_data['date'].iloc[0]  # Now start_date is in datetime format
end_date = start_date + pd.Timedelta(days=7)

# Filter for the first 7 days of data
month_data = sensor_data[(sensor_data['date'] >= start_date) & (sensor_data['date'] < end_date)]

def plot_month_traffic_intensity(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['date'], data['traffic_intensity'], color='b', linewidth=0.5)
    plt.title(f"Traffic Intensity Over 7 Days for Sensor {sensor_id}")
    plt.xlabel("Time")
    plt.ylabel("Traffic Intensity (Vehicles per Hour)")
    
    # Set x-axis major ticks to every 2 hours
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %H'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Run the plotting function
if __name__ == "__main__":
    plot_month_traffic_intensity(month_data)
