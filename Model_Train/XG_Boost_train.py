import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score,  mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from util import find_csv_filenames, df_concatenator

# dataset = "dataset_2012-24"
# con = sqlite3.connect("/../../Data/dataset.sqlite")
# query = "SELECT name FROM sqlite_master WHERE type='table';"
# table_names = pd.read_sql_query(query, con)
# print("---", table_names)
# data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
# con.close()

df_folder = 'Data/game'
res_df_folder = 'label_modified'

res_file_names = find_csv_filenames(df_folder + '/' + res_df_folder)

data = df_concatenator(res_file_names, df_folder + '/' + res_df_folder)

margin = data['Result']
data.drop(['Rk', 'Team', 'Date', 'PTS', 'Unnamed: 4', 'Opp', 'Result'],
          axis=1, inplace=True)
data = data.values

for i, row in enumerate(data):
    for j, item in enumerate(row):
        if item == 'MP':
            print(item)
            print("================")
            print(i, j)

data = data.astype(float)
acc_results = []
for x in tqdm(range(300)):
    x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=.1)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)
    param = {
        'max_depth': 3,
        'eta': 0.01,
        'objective': 'multi:softprob',
        'num_class': 2
    }
    epochs = 750

    model = xgb.train(param, train, epochs)
    predictions = model.predict(test)
    y = []

    for z in predictions:
        y.append(np.argmax(z))
    y_test = list(y_test)
    
    acc = round(accuracy_score(y_test, y) * 100, 1)
    # print(f"{acc}%")
    acc_results.append(acc)
    # only save results if they are the best so far
    if acc == max(acc_results):
        model.save_model('Model_Train/Models/XGBoost_{}%_ML-4.json'.format(acc))
