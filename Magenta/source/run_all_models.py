from utils.util import my_xgb, my_logistic_regression, my_local_extrema, plot_evaluation
import matplotlib.pyplot as plt

if __name__ == "__main__":
    res = {}
    res.update({'xgb': my_xgb()})
    res.update({'log_reg': my_logistic_regression()})
    res.update({'loc_ext': my_local_extrema()})
    plot_evaluation(res)
