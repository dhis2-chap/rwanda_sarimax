import argparse
import os
import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_for_district(district_df):
    features = ['rainfall', 'mean_temperature']
    train_exog = district_df[features]
    Y = district_df['disease_cases']
    Y = Y.fillna(0)  # set NaNs to zero (not a good solution, just for the example to work)
    

    # Handle missing values if any (forward fill as an example)
    train_exog = train_exog.fillna(method='ffill')

    my_order = (0, 1, 0)
    my_seasonal_order = (1, 0, 1, 12)

    # Define SARIMAX model with exogenous variables
    model = SARIMAX(
        Y, 
        exog=train_exog,
        order=my_order, 
        seasonal_order=my_seasonal_order,
        enforce_stationarity=False
    )

    # assert no NaNs in exog or Y
    assert not Y.isnull().any(), "Y contains NaNs"
    assert not train_exog.isnull().any().any(), "train_exog contains NaNs"

    # Fit the model
    #print("")
    #print("______________________________________________")
    #print(Y)
    #print(train_exog)
    # show the whole train_exog, not truncated
    #print(train_exog.to_string())
    model_fit = model.fit()
    return model_fit


def train(csv_fn, model_fn):
    print("Reading data from ", csv_fn)
    print("Current working directory: ", os.getcwd())
    df = pd.read_csv(csv_fn)

    # split df into one df per distinct location
    models = {}
    for district in df['location'].unique():
        print("Training for district: ", district)
        district_df = df[df['location'] == district]
        model = train_for_district(district_df)
        models[district] = model
    
        model_file_name = "model_" + district + ".bin"
        joblib.dump(model, model_file_name)
    
    joblib.dump(model, model_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a minimalist forecasting model.')

    parser.add_argument('csv_fn', type=str, help='Path to the CSV file containing input data.')
    parser.add_argument('model_fn', type=str, help='Path to save the trained model.')
    args = parser.parse_args()
    train(args.csv_fn, args.model_fn)


