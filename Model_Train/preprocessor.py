import os
from os import listdir
import pandas as pd
from util import find_csv_filenames

df_folder = 'Data/game'
input_data = 'Data/game/dataset'
res_df_folder = 'label_modified'
file_names = find_csv_filenames(input_data)
print(file_names)

# if os.path.isdir(input_data + '/' + res_df_folder):
#     os.remove(input_data + '/' + res_df_folder)
try:
    os.mkdir(input_data + '/' + res_df_folder)
except:
    pass

for file_name in file_names:
    df = pd.read_csv(str(input_data + '/' + file_name))
    
    for index in range(len(df)):
        if df['Result'][index][0] == 'W':
            df['Result'][index] = 1
        else:
            df['Result'][index] = 0

    df.to_csv(f"{df_folder}/{res_df_folder}/res_modified_{file_name}", index=False)
