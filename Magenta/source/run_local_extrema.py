import pandas as pd
from utils.util import my_local_extrema, evaluate_with_tolerance


if __name__ == "__main__":
    res = {}

    y, y_pred, evaluation_dict = my_local_extrema()
    res.update({'loc_ext': evaluation_dict})

    # Example usage
    # Assuming gt and pred are DataFrames with 'Start' and 'End' columns in the ground truth
    # and the predictions are kept as a DataFrame with timestamps as index

    # Set tolerance (you can adjust this according to your needs)
    tolerance_window = pd.Timedelta(minutes=60)

    # Evaluate with tolerance
    #precision, recall, f1, _, _ = evaluate_with_tolerance(y, y_pred, tolerance_window)


