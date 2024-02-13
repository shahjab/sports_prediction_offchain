from Model_Train.Naive_Train import Model_builder
from .util import df_concatenator, find_csv_filenames, equal_extractor

model = Model_builder()

def Predictor(season, team, team_):
    team_data = equal_extractor(season, team, team_)
    
    if model.predict([team_data])[0] == 1:
        print("Home team Win")
    else:
        print("Away Team Win")