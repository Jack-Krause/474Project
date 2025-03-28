import os
import csv

def get_features(data_path, write_file=False):
    features_array = []
    data = []
    if (os.path.isfile(data_path)):
        with open(data_path, 'r') as f:
            reader = csv.reader(f)

            i = 0
            for row in reader:
                if i == 0:
                    features_array = row
                    print(f"features: {features_array}")
                else:
                    data.append(row)
                    print(f"data row: {row}")

                i += 1
            return features_array, data

    else:
        raise FileNotFoundError("csv file not found")






