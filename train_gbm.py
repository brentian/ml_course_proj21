import glob
import math
import re

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.integration import LightGBMPruningCallback
from sklearn import datasets, linear_model, metrics, model_selection, svm, tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, log_loss, precision_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)


def gbm_objective(trial, X, y):
    # 参数网格
    param_grid = {
        "objective": "binary",
        "is_unbalance": True,
        "metric": "auc",
        "verbosity": -1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-4, 1),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "cat_smooth": trial.suggest_int("cat_smooth", 0, 100),
        "random_state": 2021,
    }
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)
    cv_scores = np.empty(5)

    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # LGBM建模
        model = lgbm.LGBMClassifier(**param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="roc_auc",
            early_stopping_rounds=100,
            callbacks=[LightGBMPruningCallback(trial, "auc")],
        )
        # 模型预测
        preds = model.predict_proba(X_test)[:, 1]
        # 优化指标logloss最小
        cv_scores[idx] = metrics.roc_auc_score(y_test, preds)

    return np.mean(cv_scores)
