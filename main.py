import get_data.retrieve_data as get_data
import ml_training.process_data as process_data
import os

base_url = "https://www.ncei.noaa.gov/data/global-summary-of-the-month/access/"
station_id = "USW00003184"
# station_id = "GHCND:USW00003184"

data_directory = "gsom_data"
os.makedirs(data_directory, exist_ok=True)

file_url = f"{base_url}{station_id}.csv"
save_path = os.path.join(data_directory, f"{station_id}.csv")

try:
    get_data.download_csv_data(file_url, save_path)
except Exception as e:
    print(f"error downloading data: {e}")

try:
    all_features, all_data = process_data.get_features(save_path)
except Exception as e:
    print(f"error when processing features: {e}")


