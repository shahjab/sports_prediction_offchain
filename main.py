import argparse
from Predict import XGBoost_predict
from Predict.NB_predict import Predictor
# from Predict.Score_Pred.model import NBAModel
from Predict.Score_Pred.model_ import NBAModel

def main():
    if args.xgb:
        # if args.lastdata:
            XGBoost_predict.predictor(args.season, args.home, args.away)
    elif args.nb:
        print("     Selected Naive Base Model\n")
        Predictor(args.season, args.home, args.away)
    elif args.score:
        model = NBAModel()
        # model.get_scores(args.home, args.away)
        # model.stat_scraper()
        model.stat_preprocessor()
        model.soft_impute()
        model.predictor(args.home, args.away)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-score', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-lastdata', action='store_true', help='Run with Last Data')
    parser.add_argument('-from', help='Beginning year')
    parser.add_argument('-to', help='Ending year')
    parser.add_argument('-home', help='name of home team')
    parser.add_argument('-away', help='name of away team')
    parser.add_argument('-season', help='Matching season')
    parser.add_argument('-scrape', help='Scrape season data')
    args = parser.parse_args()

    main()