import pandas as pd
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
import os
from Model_Train.util import find_csv_filenames
from datetime import date
import subprocess
import numpy as np
from fancyimpute import SoftImpute
import datetime

class NBAModel:
    def __init__(self, coe = 28 ) -> None:
        self.teams = ['ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE',
                      'DAL', 'DEN', 'HOU', 'DET', 'GSW', 'IND',
                      'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
                      'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO',
                      'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
        
        self.df_pace = pd.DataFrame(0, index=self.teams, columns=self.teams)
        self.df_OR = pd.DataFrame(0, index=self.teams, columns=self.teams)
        self.coe = coe
        self.today = date.today()
        self.teamname_matcher = {
            "ATLANTA HAWKS": 'ATL',
            "BOSTON CELTICS": 'BOS',
            "BROOKLYN NETS": 'BRK',
            "CHARLOTTE HORNETS": 'CHO',
            "CHICAGO BULLS": 'CHI',
            "CLEVELAND CAVALIERS": 'CLE',
            "DALLAS MAVERICKS": 'DAL',
            "DENVER NUGGETS": 'DEN',
            "HOUSTON ROCKETS": 'HOU',
            "DETROIT PISTONS": 'DET',
            "GOLDEN STATE WARRIORS": 'GSW',
            "INDIANA PACERS": 'IND',
            "LOS ANGELES CLIPPERS": 'LAC',
            "LOS ANGELES LAKERS": 'LAL',
            "MEMPHIS GRIZZLIES": 'MEM',
            "MIAMI HEAT": 'MIA',
            "MILWAUKEE BUCKS": 'MIL',
            "MINNESOTA TIMBERWOLVES": 'MIN',
            "NEW ORLEANS PELICANS": 'NOP',
            "NEW YORK KNICKS": 'NYK',
            "OKLAHOMA CITY THUNDER": 'OKC',
            "ORLANDO MAGIC": 'ORL',
            "PHILADELPHIA 76ERS": 'PHI',
            "PHOENIX SUNS": 'PHO',
            "PORTLAND TRAIL BLAZERS": 'POR',
            "SACRAMENTO KINGS": 'SAC',
            "SAN ANTONIO SPURS": 'SAS',
            "TORONTO RAPTORS": 'TOR',
            "UTAH JAZZ": 'UTA',
            "WASHINGTON WIZARDS": 'WAS',
        }

    def stat_scraper(self):
        days = self.coe

        while days != -1:
            Date = date.today() - datetime.timedelta(days=days)

            if not os.path.exists("Data/team"):
                os.mkdir("Data/team")
                os.mkdir(f"Data/team/{str(self.today)}")
            print(Date.day, Date.month, Date.year, '=--')
            res = client.team_box_scores(day=Date.day, month=Date.month, year=Date.year)
            days -= 1
            for i in range(int(len(res) / 2)):
                dic = {}
                for key in res[i * 2]:
                    dic.update({
                        key: [res[i * 2][key]]
                    })
                df_temp = pd.DataFrame(dic)

                dic = {}
                for key in res[i * 2 + 1]:
                    dic.update({
                        key: [res[i * 2 + 1][key]]
                    })
                df_temp_1 = pd.DataFrame(dic)

                temp_df = pd.concat([df_temp, df_temp_1])
                temp_df.to_csv(f"Data/team/{Date.year}-{Date.month}-{Date.day}_{i}.csv")

    def make_matrices(self, data):
        df_pace, df_OR = self.df_pace, self.df_OR
        df_pace, df_OR = self.full_update(data, df_pace, df_OR)

        return df_pace, df_OR

    def update_df(self, df, team1, team2, value):
        old_value = df[team2].loc[team1]
        if old_value == 0:
            new_value = float(value)
        else:
            new_value = (float(old_value) + float(value)) / 2
        df[team2].loc[team1] = new_value
        # df.loc[team1, team2] = new_value

        return df

    def soft_impute(self):
        # Load data
        df_pace = pd.read_csv('pace.csv')
        df_OR = pd.read_csv('OR.csv')

        # Replace 0 values with NaN
        df_pace.replace(0, np.nan, inplace=True)
        df_OR.replace(0, np.nan, inplace=True)

        # Extract relevant columns and convert to numeric matrices
        df_pace = df_pace.iloc[:, 1:30].values.astype(float)
        df_OR = df_OR.iloc[:, 1:30].values.astype(float)

        # Impute missing values using softImpute
        fits_pace = SoftImpute(max_iters=100, max_rank=10).fit_transform(df_pace)
        fits_OR = SoftImpute(max_iters=100, max_rank=10).fit_transform(df_OR)

        # Multiply the imputed matrices and divide by 100
        predictions = np.multiply(fits_OR, fits_pace) / 100

        # Create DataFrames with original column and row names
        # predictions_df = pd.DataFrame(predictions, columns=df_pace[0, :], index=df_pace[:, 0])
        df_original = pd.read_csv('Predict/Score_Pred/predictions.csv', index_col=0)  # Assuming the first column is the index
        df_original.iloc[:, 1:] = predictions

        # Save the predictions to a CSV file
        # predictions_df.to_csv('predictions.csv', index=False)
        df_original.to_csv('predictions.csv')

    
    def stat_preprocessor(self):
        file_names = find_csv_filenames("Data/team/")
        df_pace, df_OR = self.df_pace, self.df_OR

        for file_name in file_names:
            df = pd.read_csv("Data/team/" + file_name)
            df_ = pd.DataFrame(index=[1], columns=["possession", "possession.1", "pace", "ORtg", "ORtg.1"])

            df_.iloc[0]['possession'] = 0.5 * ((df.iloc[0]["attempted_field_goals"] + 0.4 * df.iloc[0]["attempted_free_throws"] - 1.07 * (df.iloc[0]["offensive_rebounds"] / (df.iloc[0]["offensive_rebounds"] + df.iloc[1]["defensive_rebounds"])) * (df.iloc[0]["attempted_field_goals"] - df.iloc[0]["made_field_goals"]) + df.iloc[0]["turnovers"]) + (df.iloc[1]["attempted_field_goals"] + 0.4 * df.iloc[1]["attempted_free_throws"] - 1.07 * (df.iloc[1]["offensive_rebounds"] / (df.iloc[1]["offensive_rebounds"] + df.iloc[0]["defensive_rebounds"])) * (df.iloc[1]["attempted_field_goals"] - df.iloc[1]["made_field_goals"]) + df.iloc[1]["turnovers"]))
            df_.iloc[0]['possession.1'] = 0.5 * ((df.iloc[1]["attempted_field_goals"] + 0.4 * df.iloc[1]["attempted_free_throws"] - 1.07 * (df.iloc[1]["offensive_rebounds"] / (df.iloc[1]["offensive_rebounds"] + df.iloc[0]["defensive_rebounds"])) * (df.iloc[1]["attempted_field_goals"] - df.iloc[1]["made_field_goals"]) + df.iloc[1]["turnovers"]) + (df.iloc[0]["attempted_field_goals"] + 0.4 * df.iloc[0]["attempted_free_throws"] - 1.07 * (df.iloc[0]["offensive_rebounds"] / (df.iloc[0]["offensive_rebounds"] + df.iloc[1]["defensive_rebounds"])) * (df.iloc[0]["attempted_field_goals"] - df.iloc[0]["made_field_goals"]) + df.iloc[0]["turnovers"]))
            df_.iloc[0]['pace'] = 48 * (df_.iloc[0]['possession'] + df_.iloc[0]['possession.1']) / (2 * df.iloc[0]['minutes_played'] / 5)
            df_.iloc[0]['ORtg'] = df.iloc[0]["points"] * 100 / df_.iloc[0]['pace']
            df_.iloc[0]['ORtg.1'] = df.iloc[1]["points"] * 100 / df_.iloc[0]['pace']

            df_pace = self.update_df(df_pace, self.teamname_matcher[str(df.iloc[0]['team']).split(".")[1].replace("_", " ")], self.teamname_matcher[str(df.iloc[1]['team']).split(".")[1].replace("_", " ")], df_.iloc[0]['pace'])
            df_pace = self.update_df(df_pace, self.teamname_matcher[str(df.iloc[1]['team']).split(".")[1].replace("_", " ")], self.teamname_matcher[str(df.iloc[0]['team']).split(".")[1].replace("_", " ")], df_.iloc[0]['pace'])
            df_OR = self.update_df(df_OR, self.teamname_matcher[str(df.iloc[0]['team']).split(".")[1].replace("_", " ")], self.teamname_matcher[str(df.iloc[1]['team']).split(".")[1].replace("_", " ")], df_.iloc[0]['ORtg'])
            df_OR = self.update_df(df_OR, self.teamname_matcher[str(df.iloc[1]['team']).split(".")[1].replace("_", " ")], self.teamname_matcher[str(df.iloc[0]['team']).split(".")[1].replace("_", " ")], df_.iloc[0]['ORtg.1'])
        
        try:
            os.remove("pace.csv")
            os.remove("OR.csv")
        except:
            pass

        df_pace.to_csv("pace.csv")
        df_OR.to_csv("OR.csv")
            
    def predictor(self, team1, team2):
        predictions = (pd.read_csv('predictions.csv')
                       .assign(**{'Unnamed: 0': self.teams})
                       .set_index('Unnamed: 0'))
        predictions.columns = self.teams
        matching_res = [predictions.loc[team1][team2], predictions.loc[team2][team1]]
        print("\n\n                          ===========     Predicting Score     ===========\n\n")
        print("\t\t\t\t\t", team1, "\t:\t", team2)
        print("\t\t\t\t\t", int(matching_res[0]), "\t:\t", int(matching_res[1]))