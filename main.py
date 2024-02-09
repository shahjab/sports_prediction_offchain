import argparse
from Predict import XGBoost_predict

def main():
    if args.xgb:
        if args.lastdata:
            XGBoost_predict.predictor(args.season, args.team1, args.team2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-lastdata', action='store_true', help='Run with Last Data')
    parser.add_argument('-from', help='Beginning year')
    parser.add_argument('-to', help='Ending year')
    parser.add_argument('-team1', help='Ending year')
    parser.add_argument('-team2', help='Ending year')
    parser.add_argument('-season', help='Ending year')
    args = parser.parse_args()

    main()