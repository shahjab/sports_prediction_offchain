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

def equal_extractor(season, team, team_):
    df = pd.read_csv("Predict/team_df.csv")
    for i, item in enumerate(df["Season"]):
        if (season in item) and (team == df.iloc[i]['Team']):
            params = df.iloc[i]
    
    for i, item in enumerate(df["Season"]):
        if (season in item) and (team == df.iloc[i]['Team']):
            params_ = df.iloc[i]

    try:
        res_ = params[["FG", "FGA", "FG%", "2P", "2PA", "2P%", "3P", "3PA", "3P%", "FT", "FTA", "FT%"]]
        res__ = params[["FG", "FGA", "FG%", "2P", "2PA", "2P%", "3P", "3PA", "3P%", "FT", "FTA", "FT%"]]
    except:
        return "No Team Data"
    res = pd.concat([res_, res__], axis=1)
    try:
        return [params, params_]
    except:
        return "No Team Data"
    