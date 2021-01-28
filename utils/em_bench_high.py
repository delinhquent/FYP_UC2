"""
Code from: https://github.com/ngoix/EMMV_benchmarks/blob/master/em_bench_high.py

Code has been adjusted to suit to the needs of this project
"""

import numpy as np

from sklearn.utils import shuffle as sh
from utils.em import em, mv


# parameters of the algorithm:
averaging = 50
max_features = 5
n_generated = 100000
alpha_min = 0.9
alpha_max = 0.999
t_max = 0.9
ocsvm_max_train = 10000

np.random.seed(42)


def calculate_emmv_score(X,y, model, novelty_detection=False,ocsvm_model=False):
    # loading and vectorization
    n_samples, n_features = np.shape(X)
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train

    X_train = X.iloc[:n_samples_train, :]
    X_test = X.iloc[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]

    if novelty_detection:
        # training and testing only on normal data:
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]
        X_test = X_test[y_test == 0]
        y_test = y_test[y_test == 0]

    n_samples, n_features = X_test.shape
    em_model, mv_model = 0, 0
    
    nb_exp = 0
    while nb_exp < averaging:
        features = sh(np.arange(n_features))[:max_features]
        X_train_ = X_train.iloc[:, features]
        X_ = X_test.iloc[:, features]

        lim_inf = X_.min(axis=0)
        lim_sup = X_.max(axis=0)
        volume_support = (lim_sup - lim_inf).prod()
        if volume_support > 0:
            nb_exp += 1
            t = np.arange(0, 100 / volume_support, 0.001 / volume_support)
            axis_alpha = np.arange(alpha_min, alpha_max, 0.001)
            unif = np.random.uniform(lim_inf, lim_sup,
                                     size=(n_generated, max_features))

            if ocsvm_model:
                model.fit(X_train_[:min(ocsvm_max_train, n_samples_train - 1)])
                s_X_model = model.decision_function(X_).reshape(1, -1)[0]
                s_unif_model = model.decision_function(unif).reshape(1, -1)[0]
            else:
                model.fit(X_train_)
                s_X_model = model.decision_function(X_)
                s_unif_model = model.decision_function(unif)
            
            em_model += em(t, t_max, volume_support, s_unif_model,
                             s_X_model, n_generated)[0]
            mv_model += mv(axis_alpha, volume_support, s_unif_model,
                             s_X_model, n_generated)[0]

    em_model /= averaging
    mv_model /= averaging

    return em_model, mv_model