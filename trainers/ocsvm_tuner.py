"""
Code from: https://github.com/bzantium/OCSVM-hyperparameter-selection/blob/950f1b372287ec3463cb48688ca95655b7399372/algorithm.py#L19

Code has been adjusted to suit to the needs of the project
"""

from sklearn.model_selection import train_test_split
from pyod.models.ocsvm import OCSVM
import numpy as np

from tqdm import tqdm

from trainers. ocsvm_datashift_algorithm import SelfAdaptiveShifting
from utils.em_bench_high import calculate_emmv_score



def find_best_ocsvm(X,y):
    y_value = np.unique(y)

    f_index = np.where(y == y_value[0])[0]
    s_index = np.where(y == y_value[1])[0]

    target_X, target_y = X[f_index], np.ones(len(f_index))
    outlier_X, outlier_y = X[s_index], -np.ones(len(s_index))
    target_X_train, target_X_test, target_y_train, target_y_test = train_test_split(target_X, target_y, shuffle=True,
                                                                                    random_state=42, test_size=1/3)

    self_adaptive_shifting = SelfAdaptiveShifting(target_X_train)
    self_adaptive_shifting.edge_pattern_detection(0.01)
    pseudo_outlier_X = self_adaptive_shifting.generate_pseudo_outliers()
    pseudo_target_X = self_adaptive_shifting.generate_pseudo_targets()
    pseudo_outlier_y = -np.ones(len(pseudo_outlier_X))
    pseudo_target_y = np.ones(len(pseudo_target_X))

    gamma_candidates = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3, 1/np.size(target_X, -1)]
    nu_candidates = [0.005, 0.01, 0.05, 0.1, 0.5]

    best_err = 1.0
    best_gamma, best_nu = 1/np.size(target_X, -1), 0.5
    for gamma in tqdm(gamma_candidates):
        for nu in tqdm(nu_candidates):
            model = OCSVM(gamma=gamma, nu=nu).fit(target_X_train)
            err_o = 1 - np.mean(model.predict(pseudo_outlier_X) == pseudo_outlier_y)
            err_t = 1 - np.mean(model.predict(pseudo_target_X) == pseudo_target_y)
            err = float((err_o + err_t) / 2)
            if err < best_err:
                best_err = err
                best_gamma = gamma
                best_nu = nu

    best_model = OCSVM(kernel='rbf', gamma=best_gamma, nu=best_nu)
    return best_model
