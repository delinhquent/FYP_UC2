"""
Code from: https://github.com/bzantium/OCSVM-hyperparameter-selection/blob/950f1b372287ec3463cb48688ca95655b7399372/main.py

Code has been adjusted to suit to the needs of the project
"""

from sklearn.model_selection import train_test_split
from pyod.models.ocsvm import OCSVM
import numpy as np

from trainers. ocsvm_datashift_algorithm import SelfAdaptiveShifting
from utils.em_bench_high import calculate_emmv_score

from sklearn.metrics import f1_score, confusion_matrix


def generate_pseudo_datasets(X_train):
    self_adaptive_shifting = SelfAdaptiveShifting(X_train.values)
    self_adaptive_shifting.edge_pattern_detection(0.01)
    pseudo_outlier_X = self_adaptive_shifting.generate_pseudo_outliers()
    pseudo_target_X = self_adaptive_shifting.generate_pseudo_targets()
    pseudo_outlier_y = np.ones(len(pseudo_outlier_X))
    pseudo_target_y = np.zeros(len(pseudo_target_X))

    return pseudo_target_X, pseudo_outlier_X, pseudo_target_y, pseudo_outlier_y


def find_best_ocsvm(X_train):
    print("Tuning One-Class SVM...")

    pseudo_target_X, pseudo_outlier_X, pseudo_target_y, pseudo_outlier_y = generate_pseudo_datasets(X_train)

    kernel_candidates = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma_candidates = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3, 1/np.size(X_train, -1)]
    nu_candidates = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3 , 0.4, 0.5]
    
    # best_err = 1.0
    best_f1 = None
    best_gamma, best_nu,best_kernel = 1/np.size(X_train, -1), 0.5,'rbf'
    for kernel in kernel_candidates:
        for gamma in gamma_candidates:
            for nu in nu_candidates:
                print("Testing kernal = {}, gamma = {} & nu = {}".format(kernel, gamma, nu))
                model = OCSVM(kernel=kernel,gamma=gamma, nu=nu).fit(X_train)
                # err_o = 1 - np.mean(model.predict(pseudo_outlier_X) == pseudo_outlier_y)
                # err_t = 1 - np.mean(model.predict(pseudo_target_X) == pseudo_target_y)
                # err = float((err_o + err_t) / 2)
                # if err < best_err:
                #     best_err = err
                #     best_gamma = gamma
                #     best_nu = nu
                #     best_kernel = kernel
                X_test = np.concatenate((pseudo_outlier_X,pseudo_target_X))
                y_test = np.concatenate((pseudo_outlier_y,pseudo_target_y))
                pred = model.predict(X_test)
                f1 = f1_score(y_test, pred)
                if best_f1 == None or f1 > best_f1:
                    best_f1 = f1
                    best_gamma = gamma
                    best_nu = nu
                    best_kernel = kernel

    best_model = OCSVM(kernel=best_kernel, gamma=best_gamma, nu=best_nu)
    return best_model

# def find_best_ocsvm_adapted(X_train, test_df):
#     print("Tuning One-Class SVM...")
    
#     y_test = evaluate_df['manual_label']
#     X_test = evaluate_df.drop(columns='manual_label')

#     outlier_df = evaluate_df[evaluate_df['manual_label'] == 1]
#     pseudo_outlier_y = outlier_df['manual_label']
#     pseudo_outlier_X = outlier_df.drop(columns='manual_label')

#     target_df = evaluate_df[evaluate_df['manual_label'] == 0]
#     pseudo_target_y = target_df['manual_label']
#     pseudo_target_X = pseudo_target_X.drop(columns='manual_label')

#     kernel_candidates = ['linear', 'poly', 'rbf', 'sigmoid']
#     gamma_candidates = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3, 1/np.size(X_train, -1)]
#     nu_candidates = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3 , 0.4, 0.5]
    
#     best_err = 1.0
#     best_gamma, best_nu,best_kernel = 1/np.size(X_train, -1), 0.5,'rbf'
#     for kernel in kernel_candidates:
#         for gamma in gamma_candidates:
#             for nu in nu_candidates:
#                 for contamination in contaimination_candidates:
#                     print("Testing kernal = {}, gamma = {} & nu = {} = {}".format(kernel, gamma, nu))
#                     model = OCSVM(kernel=kernel,gamma=gamma, nu=nu,contamination=contamination).fit(X_train)
#                     err_o = 1 - np.mean(model.predict(pseudo_outlier_X) == pseudo_outlier_y)
#                     err_t = 1 - np.mean(model.predict(pseudo_target_X) == pseudo_target_y)
#                     err = float((err_o + err_t) / 2)
#                     if err < best_err:
#                         best_err = err
#                         best_gamma = gamma
#                         best_nu = nu
#                         best_kernel = kernel

#     best_model = OCSVM(kernel=best_kernel, gamma=best_gamma, nu=best_nu)
#     return best_model