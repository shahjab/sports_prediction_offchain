from sklearn.naive_bayes import GaussianNB
import pandas as pd
from .Naive_Base import ModelEval
from .util import find_csv_filenames, df_concatenator
from sklearn.model_selection import train_test_split

def Model_builder():
    file_names = find_csv_filenames("Data/game/label_modified")
    directory = "Data/game/label_modified"

    data = df_concatenator(file_names, directory)
    data = data.reset_index(drop=True)

    nb = GaussianNB()

    results_df = pd.DataFrame(columns=['model_name', 'cv_score', 'gs_score', 'train_score', 'test_score'])
    residuals_df = pd.DataFrame(columns=['y_true'])
    residuals_df['y_true'] = data['Result']

    stat_columns = ['FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 
                    'FG.1', 'FGA.1', 'FG%.1', '2P.1', '2PA.1', '2P%.1', '3P.1', '3PA.1', '3P%.1', 'FT.1', 'FTA.1', 'FT%.1']

    X = data[stat_columns]
    y = data['Result']

    stats = train_test_split(X, y, train_size=.80, random_state=99)

    Trainer = ModelEval(nb, 'gnb_12F_10', results_df, residuals_df, stats, None)
    model, results_df = Trainer.full_diag()

    return model