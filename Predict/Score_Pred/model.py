from urllib.request import urlopen
from urllib.error import HTTPError
import pickle
import subprocess
import pandas as pd
from bs4 import BeautifulSoup
import time


class NBAModel:
    def __init__(self, update=False):
        self.update = update
        self.urls = ["http://www.basketball-reference.com/leagues/NBA_2023_games-october.html",
                     "http://www.basketball-reference.com/leagues/NBA_2023_games-november.html",
                     "http://www.basketball-reference.com/leagues/NBA_2023_games-december.html",
                     "http://www.basketball-reference.com/leagues/NBA_2024_games-january.html"]
        self.teams = ['ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE',
                      'DAL', 'DEN', 'HOU', 'DET', 'GSW', 'IND',
                      'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
                      'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO',
                      'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
        if update:
            self.box_urls = self.get_urls()
            self.df_pace = pd.DataFrame(0, index=self.teams,
                                        columns=self.teams)
            self.df_OR = pd.DataFrame(0, index=self.teams,
                                      columns=self.teams)
            self.df_pace, self.df_OR = self.make_matrices()
            print(self.df_pace)
            print(self.df_OR)
            self.write_matrices()
            self.soft_impute()
        self.predictions = self.get_predictions()

    def __repr__(self):
        return "NBAModel(update={update})".format(update=self.update)

    def get_urls(self):

        box_urls = []
        for url in self.urls:
            backoff_time=1
            print('****', url)
            while(1):
                try:
                    response = urlopen(url)
                except HTTPError as e:
                    if e.code == 429:
                        print("delaying ", backoff_time, " . . .")
                        time.sleep(backoff_time)
                        backoff_time *= 2

            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            soup.find_all('a')
            for link in soup.find_all('a'):
                if link.get('href').startswith('/boxscores/2'):
                    box_urls.append(str(link.get('href')))
        pickle.dump(box_urls, open("box_urls.p", "wb"))
        return box_urls

    def get_stats(self, url):

        backoff_time=1
        while(1):
            try:
                response = urlopen(url)
            except HTTPError as e:
                if e.code == 429:
                    print("delaying ", backoff_time, " . . .")
                    time.sleep(backoff_time)
                    backoff_time *= 2

        html = response.read()
        html = str(html)
        stat_html = html.replace('<!--', "")
        stat_html = stat_html.replace('-->', "")
        stats = pd.read_html(stat_html)

        index = 0
        while '\\n\\t' in str(stats[index].iloc[0][2]):
            index += 1
        return stats[index + 1]

    def update_df(self, df, team1, team2, value):
        old_value = df[team2].loc[team1]
        if old_value == 0:
            new_value = float(value)
        else:
            new_value = (float(old_value) + float(value)) / 2
        df[team2].loc[team1] = new_value
        return df

    def extract_data(self, table):     
        team1 = table.loc[0][0]
        team2 = table.loc[1][0]
        pace = table.loc[1][1]
        team1_OR = table.loc[0][6]
        team2_OR = table.loc[1][6]

        return team1, team2, team1_OR, team2_OR, pace

    def full_update(self, url, df_pace, df_OR):

        table = self.get_stats(url)
        print(table)
        print(type(table), "***")
        team1, team2, team1_OR, team2_OR, pace = self.extract_data(table)
        print(df_pace, "*****")
        df_pace = self.update_df(df_pace, team1, team2, pace)
        df_pace = self.update_df(df_pace, team2, team1, pace)
        df_OR = self.update_df(df_OR, team1, team2, team1_OR)
        df_OR = self.update_df(df_OR, team2, team1, team2_OR)
        return df_pace, df_OR

    def make_matrices(self):

        df_pace, df_OR = self.df_pace, self.df_OR

        for url in self.box_urls:
            url = 'http://www.basketball-reference.com' + url
            print(url)
            df_pace, df_OR = self.full_update(url, df_pace, df_OR)
        return df_pace, df_OR

    def write_matrices(self):

        self.df_pace.to_csv('pace.csv')
        self.df_OR.to_csv('OR.csv')

    def soft_impute(self):

        subprocess.check_output(['Rscript', './model/predict_soft_impute.R'])

    def get_predictions(self):

        predictions = (pd.read_csv('Predict/Score_Pred/predictions.csv')
                       .assign(**{'Unnamed: 0': self.teams})
                       .set_index('Unnamed: 0'))
        predictions.columns = self.teams
        return predictions

    def get_scores(self, team1, team2):

        team1s = self.predictions.loc[team1][team2]
        team2s = self.predictions.loc[team2][team1]
        # print("Matching Team:")
        print("\t\t\t\t\t", team1, "\t:\t", team2)
        # print(f"Expected result between {team1} and {team2}:")
        print("\t\t\t\t\t", int(team1s), "\t:\t", int(team2s))
        print('')


# model = NBAModel(update=True)
# model.get_scores('PHO', 'IND')
# model.get_scores('GSW', 'IND')
# model.get_scores('MEM', 'CHO')
# model.get_scores('MIA', 'PHI')
# model.get_scores('HOU', 'DET')
# model.get_scores('ORL', 'MIL')
# model.get_scores('BOS', 'MIN')
# model.get_scores('DAL', 'SAS')
# model.get_scores('TOR', 'LAC')

