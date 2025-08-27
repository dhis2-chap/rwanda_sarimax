import argparse

import joblib
import pandas as pd
import numpy as np


def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    # get all unique districts from historic data
    df = pd.read_csv(future_climatedata_fn)
    districts = pd.read_csv(historic_data_fn)['location'].unique()

    final_predictions_df = pd.DataFrame()

    for district in districts:
        model_file_name = f"{model_fn}_{district}.bin"
        model = joblib.load(model_file_name)

        future_data_for_district = df.loc[df["location"] == district].copy()
        assert len(future_data_for_district) > 0, f"No future data for district {district}"

        print(f"Loaded model for district {district} from {model_file_name}")

        # Forecast
        steps = len(future_data_for_district)
        exog = future_data_for_district[["rainfall", "mean_temperature"]]
        predictions = model.forecast(steps=steps, exog=exog)
        assert not pd.isnull(predictions).any(), f"Predictions contain NaNs for district {district}"

        preds = np.asarray(predictions).reshape(-1)
        if preds.shape[0] != steps:
            raise ValueError(
                f"Prediction length {preds.shape[0]} != future rows {steps} for district {district}"
            )


        future_data_for_district.loc[:, "sample_0"] = preds

        assert len(predictions) == steps, (
            f"Number of predictions {len(predictions)} does not match "
            f"length of future data {steps} for district {district}"
        )

        print("____ predictions for district", district, "________")
        print(future_data_for_district)

        # Collect for later concat (concat once is faster, but this works too)
        final_predictions_df = pd.concat([final_predictions_df, future_data_for_district], ignore_index=False)

    #print(final_predictions_df)
    print("Writing final predictions to ", predictions_fn)
    final_predictions_df.to_csv(predictions_fn, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using the trained model.')

    parser.add_argument('model_fn', type=str, help='Path to the trained model file.')
    parser.add_argument('historic_data_fn', type=str, help='Path to the CSV file historic data (here ignored).')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to the CSV file containing future climate data.')
    parser.add_argument('predictions_fn', type=str, help='Path to save the predictions CSV file.')

    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)
