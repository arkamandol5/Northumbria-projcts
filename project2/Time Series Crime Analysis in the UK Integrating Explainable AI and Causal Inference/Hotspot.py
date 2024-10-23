#%%
import pandas as pd
import numpy as np

# Path to your dataset
file_path = '/Users/arkamandol/DataspellProjects/Desertation_arka_23023023/data_files/uk_crime_lat_long.csv'
data = pd.read_csv(file_path)

#%%
data.head()

#%%
# Convert 'Month' to datetime format for better date handling
data['Month'] = pd.to_datetime(data['Month'])

# Sort the DataFrame by the 'Month' column
data.sort_values('Month', inplace=True)
data.head()
#%%
# Group data by Month, Latitude, and Longitude and count occurrences
monthly_data = data.groupby(['Month', 'Latitude', 'Longitude']).size().reset_index(name='Count')
monthly_data.head()
#%%
from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the 'Count' data to scale it
monthly_data['Normalized_Count'] = scaler.fit_transform(monthly_data[['Count']])

#%%
# Function to create input sequences for LSTM
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# Define the number of past months data to consider for predicting the next month
look_back = 3
X, y = create_dataset(monthly_data['Normalized_Count'].values, look_back)

# Reshape input to be [samples, time steps, features] for LSTM
X = np.reshape(X, (X.shape[0], look_back, 1))

#%%
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))  # 50 LSTM units
model.add(Dense(1))  # Output layer that predicts the future value
model.compile(loss='mean_squared_error', optimizer='adam')

#%%
# Fit the model on the dataset
model.fit(X, y, epochs=10, batch_size=512, verbose=1)

#%%
# Predict future values
predictions = model.predict(X)

#%%
# Inverse transform to get predictions in the original count scale
predictions = scaler.inverse_transform(predictions)

#%%
# Add predictions to the monthly_data DataFrame
monthly_data['Predicted_Count'] = np.concatenate([np.zeros(look_back), predictions.flatten()])

#%%
import folium
from folium.plugins import HeatMap

# Create a map centered around an average location
center_lat, center_lon = monthly_data['Latitude'].mean(), monthly_data['Longitude'].mean()
map = folium.Map(location=[center_lat, center_lon], zoom_start=6)

# Add a heatmap to the map using predicted crime counts
heat_data = [[row['Latitude'], row['Longitude'], row['Predicted_Count']] for index, row in monthly_data.iterrows()]
HeatMap(heat_data).add_to(map)

# Save or display the map
map.save('/Users/arkamandol/DataspellProjects/Desertation_arka_23023023/data_files/crime_hotspots.html')

#%%
from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate RMSE
rmse = sqrt(mean_squared_error(y, predictions))
print('Root Mean Squared Error:', rmse)

#%%


#%%
