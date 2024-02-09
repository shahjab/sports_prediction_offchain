from os import listdir
import pandas as pd

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith( suffix )]

def df_concatenator(file_names, directory):    
    
    df = pd.DataFrame()
    for file_name in file_names:
        data = pd.read_csv(f"{directory}/{file_name}")
        df = pd.concat([df, data])

    return df