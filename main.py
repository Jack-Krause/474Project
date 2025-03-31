import get_data.retrieve_data as get_data
import ml_training.process_data as process_data
import os

base_url = "https://www.ncei.noaa.gov/data/global-summary-of-the-month/access/"
station_id = "USW00003184"
# station_id = "GHCND:USW00003184"

data_directory = "data"
os.makedirs(data_directory, exist_ok=True)

file_url = f"{base_url}{station_id}.csv"
save_path = os.path.join(data_directory, f"{station_id}.csv")

try:
    get_data.download_csv_data(file_url, save_path)
except Exception as e:
    print(f"error downloading data: {e}")

all_features, all_data = [], []
try:
    all_features, all_data = process_data.load_data(data_path=save_path,
                                                    write_file=True,
                                                    write_path=data_directory
                                                    )
    print(f"length of all data: {len(all_data)}")
except Exception as e:
    print(f"error when processing features: {e}")

if all_features is None or all_data is None:
    print("error - no data found")
else:
    training_data, testing_data = process_data.separate_sets(all_data)
    y_cols = ["PRCP"]
    Y = process_data.get_selected_features(data_path=os.path.join(data_directory, "USW00003184.csv"),
                                           features_arr=y_cols,
                                           write_file=True,
                                           write_path=data_directory
                                           )




