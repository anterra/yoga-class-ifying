import math
import pickle
import statistics
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, log_loss, make_scorer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, auc
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from xgboost import XGBClassifier


def average_metrics_pipeline(X, y, model_name, model, oversampler):
    """ For a given dataset and classification model, fits the model to oversampled data
    and returns the weighted average values for all classfication scoring metrics across 
    5 cross-validated folds."""

    # setting up k-folds and dictionaries to hold results
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = []
    cv_precision = []
    cv_recall = []
    cv_f1 = []

    # generating indices for use with k-fold splitting
    X = np.array(X)
    y = np.array(y)

    # fitting each model and k-fold splitting
    # (manually keeping track of which indices are being split into training/valudation sets
    # so I can oversample only to the training set)
    for train_ind, val_ind in kf.split(X, y):

        X_train, y_train = X[train_ind], y[train_ind]
        X_oversampled_train, y_oversampled_train = oversampler.fit_sample(
            X_train, y_train)
        X_val, y_val = X[val_ind], y[val_ind]

        if model_name == "XGBoost":
            eval_set = [
                (X_oversampled_train, y_oversampled_train), (X_val, y_val)]
            model.fit(X_oversampled_train,
                      np.array(y_oversampled_train).ravel(),
                      eval_set=eval_set,
                      eval_metric="merror",
                      early_stopping_rounds=50,
                      verbose=False)
            y_pred = model.predict(
                X_val, ntree_limit=model.best_ntree_limit)

        else:
            model.fit(X_oversampled_train, np.array(
                y_oversampled_train).ravel())
            y_pred = model.predict(X_val)

        # find average scoring metrics across all classes
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="weighted")
        recall = recall_score(y_val, y_pred, average="weighted")
        f1 = f1_score(y_val, y_pred, average="weighted")

        cv_accuracy.append(accuracy)
        cv_precision.append(precision)
        cv_recall.append(recall)
        cv_f1.append(f1)

        #logloss = log_loss(y_val, y_pred)

    # calculate means and standard deviations for each metric
    cv_accuracy = "Accuracy: {}, Variance: {}".format(statistics.mean(
        cv_accuracy), statistics.variance(cv_accuracy))
    cv_precision = "Precision: {}, Variance: {}".format(statistics.mean(
        cv_precision), statistics.variance(cv_precision))
    cv_recall = "Recall: {}, Variance: {}".format(statistics.mean(
        cv_recall), statistics.variance(cv_recall))
    cv_f1 = "F1 Score: {}, Variance: {}".format(
        statistics.mean(cv_f1), statistics.variance(cv_f1))

    print(model_name, ":")
    print(cv_accuracy)
    print(cv_precision)
    print(cv_recall)
    print(cv_f1)


def classification_reports_pipeline(X, y, model_name, model, oversampler):
    """ For a given dataset and classification model, fits the model, predicts classes of
    validation set, and returns the classification report showing precision, recall, f1 and 
    support for each class, as well as the average accuracy score. """

    X = np.array(X)
    y = np.array(y)

    # train-val split and oversample
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.2, random_state=0)
    X_oversampled_train, y_oversampled_train = oversampler.fit_sample(
        X_train, y_train)

    # fit models
    if model_name == "XGBoost":
        eval_set = [(X_oversampled_train, y_oversampled_train), (X_val, y_val)]
        model.fit(X_oversampled_train,
                  np.array(y_oversampled_train).ravel(),
                  eval_set=eval_set,
                  eval_metric="merror",
                  early_stopping_rounds=50,
                  verbose=False)
        y_pred = model.predict(X_val, ntree_limit=model.best_ntree_limit)
    else:
        model.fit(X_oversampled_train, np.array(y_oversampled_train).ravel())
        y_pred = model.predict(X_val)

    # print classification reports
    print(model_name, classification_report(y_val, y_pred))


def roc_curve_pipeline(X, y, model_name, model, oversampler):
    """ For a given multi-class dataset and classification model, fits the model, predicts probability
    of validation set datapoints being in each class, calculates false positive rate and 
    true positive rate for each class, and plots the roc_auc curves for each class on a single plot."""

    # binarizing the output for plotting
    y = label_binarize(y, classes=['Vinyasa', 'Ashtanga', 'Iyengar', 'Power', 'Restorative', 'Hatha',
                                   'Gentle', 'Yin'])
    n_classes = y.shape[1]

    # train-val split and oversample
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.2, random_state=0)
    X_oversampled_train, y_oversampled_train = oversampler.fit_sample(
        X_train, y_train)

    # instantiate One vs. Rest Classifier, which allows for use
    model = OneVsRestClassifier(model)

    # fit model and predict probabilities of validation data being in each class
    model.fit(X_oversampled_train, y_oversampled_train)
    y_pred = model.predict_proba(X_val)

    # calculate fpr, tpr, and roc_auc for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # finding mean fpr/tpr/roc_auc
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # plot roc_auc curves
    plt.figure(figsize=(10, 10))
    plt.plot(fpr["macro"], tpr["macro"], color="#73a7a5", label="macro", lw=2)
    colors = (["#7c003b", "#decae3", "#f5e5ec", "#ebccda",
               "#f7c0d4", "#6d2f77", "#c9b3b9", "#dcd8d9"])
    for i, color in zip(range(n_classes), colors):
        classes = ['Vinyasa', 'Ashtanga', 'Iyengar',
                   'Power', 'Restorative', 'Hatha', 'Gentle', 'Yin']
        plt.plot(fpr[i], tpr[i], color=color,
                 label="{}, area: {}".format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curves for {}'.format(model_name))
    plt.legend()
    plt.show()
