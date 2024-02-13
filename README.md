# NBA Sports Prediction 🏀

A machine learning algorithm used to predict the winners of NBA games. 

## Packages Used

Use Python 3.11. In particular the packages/libraries used are...

* Tensorflow - Machine learning library
* XGBoost - Gradient boosting framework
* Numpy - Package for scientific computing in Python
* Pandas - Data manipulation and analysis
* Tqdm - Progress bars
* Scikit_learn - Machine learning library

## Project Requirements Installation Guide
* Install Virtual Environment
```bash
$ python -m venv venv
```
    
* Activate Virtual Environment
```bash
$ cd venv/scripts
$ activate
```
       
* Install required python packages
```bash
$ pip install -r requirements.txt
```

## Usage

Make sure all packages above are installed.

```bash
$ pip3 install -r requirements.txt
$ python main.py -home BOS -away DEN -xgb -season 23-24
```

## Arguments Description

* home - Home team name
* away - Away team name
* xgb - command to use XGBoost ML model
* nb - command to use Naive Base ML model
* score - command to predict expected output
* season: Match year. Write as type of YY-YY

## Expected Output
1: First team is winner
0: Second team is winner

## Model Description

Used XGBoost that is great fitting model for NBA result prediction.
Trained XGBoost for last 10 years of NBA Match and Epoch is 750.
Selected the best model among trained models and its prediction score is about 95.6 for training data.

## Extensibility
Please remind that this is just a simple model that predict the winners of the NBA game.
We are going to use the advanced models and dataset to increase the accuracy and predict more match details.