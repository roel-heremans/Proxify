from utils.util import my_xgb


if __name__ == "__main__":
    res = {}
    y, y_pred, evaluation_dict = my_xgb()
    res.update({'loc_ext': evaluation_dict})