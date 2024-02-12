import argparse
from Predict import XGBoost_predict

def main():
    if args.xgb:
        # if args.lastdata:
            XGBoost_predict.predictor(args.season, args.home, args.away)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-lastdata', action='store_true', help='Run with Last Data')
    parser.add_argument('-from', help='Beginning year')
    parser.add_argument('-to', help='Ending year')
    parser.add_argument('-home', help='name of home team')
    parser.add_argument('-away', help='name of away team')
    parser.add_argument('-season', help='Matching season')
    args = parser.parse_args()

    main()