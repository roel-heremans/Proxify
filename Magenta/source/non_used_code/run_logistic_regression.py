from utils.util import my_logistic_regression


if __name__ == "__main__":
    res = {}
    y, y_pred, evaluation_dict = my_logistic_regression()
    res.update({'loc_ext': evaluation_dict})


