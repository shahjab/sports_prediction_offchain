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
            break
    
    for i, item in enumerate(df["Season"]):
        if (season in item) and (team_ == df.iloc[i]['Team']):
            params_ = df.iloc[i]
            break

    try:
        res_ = params[["FG", "FGA", "FG%", "2P", "2PA", "2P%", "3P", "3PA", "3P%", "FT", "FTA", "FT%"]]
        res__ = params_[["FG", "FGA", "FG%", "2P", "2PA", "2P%", "3P", "3PA", "3P%", "FT", "FTA", "FT%"]]
    except:
        return "No Team Data"
    
    res_ = res_.to_frame()
    res__ = res__.to_frame()

    print("\n     **********  Home Team Parameters **********\n")
    print(res_.transpose())

    print("\n\n     **********  Away Team Parameters **********\n")
    print(res__.transpose())
    print("\n")

    res_ = res_.transpose()
    res__ = res__.transpose()

    res = pd.concat([res_.iloc[0], res__.iloc[0]], axis=0)

    return res
    