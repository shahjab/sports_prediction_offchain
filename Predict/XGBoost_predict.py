import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score
from .util import df_concatenator, find_csv_filenames, equal_extractor

# class Predictor:
#     def __init__(self) -> None:
def predictor(season, team, team_):
    model_xgb = xgb.Booster()
    model_xgb.load_model("Model_Train/Models/XGBoost_95.4%_ML-4.json")

    filenames = find_csv_filenames("Data/game/label_modified")

    # season = "23-24"
    # team = "BOS"
    # team_ = "DEN"

    team_data = equal_extractor(season, team, team_)[0]

    data = df_concatenator(file_names=filenames, directory="Data/game/label_modified")

    margin = data['Result']

    data.drop(['Rk', 'Team', 'Date', 'PTS', 'Unnamed: 4', 'Opp', 'Result'], axis=1, inplace=True)
    data = data.values
    data = data.astype(float)
    # print(type(data))

    n = len(data)

    db = xgb.DMatrix(data[:n])
    # print(type(team_data))

    team_data = team_data.to_frame()
    # print(team_data)
    team_data.drop(["Rk", "Season", "Team", "W", "G", "W.1", "L", "W/L%"
    ], inplace=True)

    # print(type(team_data))
    team_data = team_data.values
    team_data = team_data.astype(float)
    n = 1

    team_data = xgb.DMatrix(team_data)
    predictions = model_xgb.predict(team_data)

    res = []
    for predict in predictions:
        if predict[0] < predict[1]:
            res.append(1)
        else:
            res.append(0)
    # print(list(margin[:n]))

    print(res[0])
    # print(int(accuracy_score([res[0]], list(margin[:n]))))
    return res