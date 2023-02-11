import pandas as pd
import numpy as np


csv_dir = "/home/guests/projects/ukbb/yadu/age_imaging_filtered.csv"


# https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing
df = pd.read_csv(csv_dir)

print("len df: ", len(df))

df['split'] = np.random.randn(df.shape[0], 1)

mask = np.random.rand(len(df)) <= 0.8

train_val = df[mask]
test = df[~mask]

print("len of train val: ", len(train_val))
print("len of test: ", len(test))

train_val.to_csv(path_or_buf="/home/guests/projects/ukbb/yadu/age_imaging_filtered_train_val.csv", index=False)
test.to_csv(path_or_buf="/home/guests/projects/ukbb/yadu/age_imaging_filtered_test.csv", index=False)


