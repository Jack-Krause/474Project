import os
import csv
import random

def get_data(data_path, write_file=False):
    features_array = []
    data = []
    if os.path.isfile(data_path):
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


def separate_sets(data_arr, seed=42):
    random.seed(42)
    random.shuffle(data_arr)

    split_idx = int(len(data_arr) * 0.80)
    training_arr = data_arr[ :split_idx]
    testing_arr = data_arr[split_idx: ]
    print(f"training size: {len(training_arr)}, testing size: {len(testing_arr)}")

    return training_arr, testing_arr








