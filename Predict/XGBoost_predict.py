import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score
from .util import df_concatenator, find_csv_filenames, equal_extractor

# class Predictor:
#     def __init__(self) -> None:
def predictor(season, team, team_):

    print("=============  Selecting Model =============\n")
    print("         Selected XGBoost\n")
    model_xgb = xgb.Booster()

    print("         Model Loading . . . \n")
    model_xgb.load_model("Model_Train/Models/XGBoost_95.4%_ML-4.json")

    # filenames = find_csv_filenames("Data/game/label_modified")

    # season = "23-24"
    # team = "BOS"
    # team_ = "DEN"

    print("=============  Extracting Home Team and Away Team Data . . . =============\n")
    team_data = equal_extractor(season, team, team_)

    # data = df_concatenator(file_names=filenames, directory="Data/game/label_modified")

    # margin = data['Result']

    # data.drop(['Rk', 'Team', 'Date', 'PTS', 'Unnamed: 4', 'Opp', 'Result'], axis=1, inplace=True)
    # data = data.values
    # data = data.astype(float)
    # # print(type(data))

    # n = len(data)

    # db = xgb.DMatrix(data[:n])
    # print(type(team_data))

    print("=============  Preprocessing Team Parameters . . . =============\n")
    # team_data.drop(["Rk", "Season", "Team", "W", "G", "W.1", "L", "W/L%"], inplace=True)

    team_data = team_data.values
    team_data = team_data.astype(float)
    n = 1

    team_data = xgb.DMatrix([team_data])

    print("=============  Predicting . . . =============\n")
    predictions = model_xgb.predict(team_data)
    print(predictions, "\n")
    res = []
    for predict in predictions:
        if predict[0] < predict[1]:
            res.append(1)
        else:
            res.append(0)
    # print(list(margin[:n]))

    if res[0] == 0:
        print(f"{team} (Home Team) will Win ! ! !")
    else:
        print(f"{team_} (Away Team) will Win ! ! !")
    # print(int(accuracy_score([res[0]], list(margin[:n]))))
    return res