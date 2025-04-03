import get_data.retrieve_data as get_data
import ml_training.process_data as process_data
import os


root_dir = os.environ.get("ROOT_DIR")
dataloc = os.path.join(root_dir, "data", "Pavement.csv")
parsed_data_dir = os.path.join(root_dir, "parsed_data")

if not os.path.isdir(parsed_data_dir):
    os.mkdir(parsed_data_dir)

if not os.path.isfile(dataloc):
    raise FileNotFoundError(f"file not found: {dataloc}")

data, feature_names = get_data.parse_csv(dataloc, save_headers=True)


print(feature_names[:20])
print(data[:20])

print(f"total data: {len(data)}")

custom_target = process_data.compute_target(
    data,
    feature_conditions = [
        "AADT > 10000",
        "STRUCT_NEED80 > 1",
        "IRI_INDX >= 3"
    ]
)

print(f"custom target: \n {custom_target}")
