import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load accident data (replace with correct path)
merged_gdf = gpd.read_file('accident_data_path.shp')

# Check and clean data
merged_gdf.dropna(subset=['longitude', 'latitude'], inplace=True)
merged_gdf['hour'] = merged_gdf['date'].dt.hour
merged_gdf['day_of_week'] = merged_gdf['date'].dt.weekday

# Encode categorical variables
label_encoder = LabelEncoder()
merged_gdf['weather_conditions_encoded'] = label_encoder.fit_transform(merged_gdf['weather_conditions'])
merged_gdf['road_type_encoded'] = label_encoder.fit_transform(merged_gdf['road_type'])
merged_gdf['junction_detail_encoded'] = label_encoder.fit_transform(merged_gdf['junction_detail'])

# Select features and target variable
features = ['hour', 'day_of_week', 'weather_conditions_encoded', 'road_type_encoded', 'junction_detail_encoded', 'speed_limit', 'number_of_vehicles', 'distance_to_road']
X = merged_gdf[features]
y = merged_gdf['accident_severity_encoded']  # Target variable: encoded accident severity
