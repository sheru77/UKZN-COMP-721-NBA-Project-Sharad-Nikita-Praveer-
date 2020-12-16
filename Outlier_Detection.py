# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 02:24:14 2020

@author: sheru
"""

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

from sklearn import preprocessing
from sklearn.svm import OneClassSVM

from sklearn.cluster import KMeans

from sklearn.ensemble import IsolationForest

from sklearn.covariance import EllipticEnvelope


def load_dataset():
    data = pd.read_csv("player_regular_season.csv")
    df = data[data['year'] >= 2004]

    # features = ['minutes', 'pts', 'reb', 'asts', 'stl', 'blk', 'turnover', 'pf', 'fga', 'fgm', 'fta', 'ftm', 'tpa',
    # 'tpm']

    X_train = df.iloc[:, [6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
    y_names = df.iloc[:, [0, 1, 2, 3]]

    x = X_train.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    X_train = pd.DataFrame(x_scaled)
    X_train.columns = X_train.iloc[0, :]
    y_names = y_names.values

    return X_train, y_names


def dbScan(X_train, y_names):
    clustering = DBSCAN(eps=0.63, metric="euclidean", min_samples=50, n_jobs=1).fit(X_train)
    outlier_df = pd.DataFrame(X_train)

    y_pred = clustering.labels_

    plot(X_train, y_pred, "DBScan")

    outliers = outlier_df[clustering.labels_ == -1]
    outliers = outliers[:].index.to_numpy()

    #y_names = y_names.values

    print("DBSCAN OUTLIERS:")
    for i in range(0, len(outliers)):
        print(outliers[i], y_names[outliers[i]])
    print("Number of outliers:", len(outliers))
    print("###############################################################################")


def LOF(X_train, y_names):
    clf = LocalOutlierFactor(n_neighbors=30, contamination=.06)
    LOF_pred = pd.Series(clf.fit_predict(X_train)).replace([-1, 1], [1, 0])
    LOF_anomalies = X_train[LOF_pred == 1]

    plt.scatter(X_train.iloc[:, 2], X_train.iloc[:, 1], c='grey', s=20, edgecolor='black')
    plt.scatter(LOF_anomalies.iloc[:, 2], LOF_anomalies.iloc[:, 1], c='blue', edgecolor='grey')
    plt.title('LOF Outlier Detection')
    plt.ylabel('Minutes')
    plt.xlabel('Points')
    plt.show()

    indexes = LOF_anomalies.index
    indexes = list(indexes)

    print("\nLOF OUTLIERS:")
    for i in range(0, len(indexes)):
        print(indexes[i], y_names[indexes[i]])
    print("Number of outliers:", len(indexes), "\n********************END********************")


def one_class_svm(X_train, y_names):
    clf = OneClassSVM(nu=0.05, kernel='rbf', gamma=.001).fit(X_train)
    y_pred = clf.predict(X_train)

    plot(X_train, y_pred, 'ONE-CLASS SVM Outlier Detection')

    outlier_list = []
    for i in range(0, len(y_pred)):
        if y_pred[i] == -1:
            outlier_list.append(i)

    print("ONE-CLASS SVM OUTLIERS:")
    for i in range(0, len(outlier_list)):
        print(outlier_list[i], y_names[outlier_list[i]])
    print("Number of outliers:", len(outlier_list))


def k_means(X_train, y_names):
    x = X_train.values
    wcss = []
    for i in range(1, 16):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0).fit(x)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 16), wcss)
    plt.show()
    kmeans = KMeans(n_clusters=6, init='k-means++', random_state=5)
    y = kmeans.fit_predict(X_train)


    print('###################################################################################')

    plt.scatter(x[y == 0, 0], x[y == 0, 1], s=25, c='indigo', label='Cluster1')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], s=25, c='sienna', label='Cluster2')
    plt.scatter(x[y == 2, 0], x[y == 2, 1], s=25, c='crimson', label='Cluster3')
    plt.scatter(x[y == 3, 0], x[y == 3, 1], s=25, c='black', label='Cluster4')
    plt.scatter(x[y == 4, 0], x[y == 4, 1], s=25, c='orange', label='Cluster5')
    plt.scatter(x[y == 5, 0], x[y == 5, 1], s=25, c='grey', label='Cluster6')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=25, c='yellow', label='centroid')
    plt.legend()
    plt.show()

    outlier_list = []
    for i in range(0, len(y)):
        if y[i] == 3:
            outlier_list.append(i)
    print("K-Means OUTLIERS:")
    for i in range(0, len(outlier_list)):
        print(outlier_list[i], y_names[outlier_list[i]])
    print("Number of outliers:", len(outlier_list))


def isolation_forest(X_train, y_names):
    clf = IsolationForest(contamination=0.05)
    y_pred = clf.fit_predict(X_train)

    plot(X_train, y_pred, "Isolation Forest")

    outlier_list = []
    for i in range(0, len(y_pred)):
        if y_pred[i] == -1:
            outlier_list.append(i)

    print("Isofrest:")
    for i in range(0, len(outlier_list)):
        print(outlier_list[i], y_names[outlier_list[i]])
    print("Number of outliers:", len(outlier_list))


def elliptic_envelope(X_train, y_names):
    cov = EllipticEnvelope(contamination=0.05)
    y_pred = cov.fit_predict(X_train)

    plot(X_train, y_pred, "Covariance")

    outlier_list = []
    for i in range(0, len(y_pred)):
        if y_pred[i] == -1:
            outlier_list.append(i)

    print("Covariance")
    for i in range(0, len(outlier_list)):
        print(outlier_list[i], y_names[outlier_list[i]])
    print("Number of outliers:", len(outlier_list))


def plot(X_train, y_pred, title):
    cmap = cm.get_cmap('bwr')
    plt.scatter(X_train.iloc[:, 2].values, X_train.iloc[:, 1].values, c=y_pred, cmap=cmap, s=20, edgecolor='black')
    plt.title(title)
    plt.show()


def run_outlier_detection():
    X_train, y_names = load_dataset()
    dbScan(X_train, y_names)
    LOF(X_train, y_names)
    one_class_svm(X_train, y_names)
    k_means(X_train, y_names)
    isolation_forest(X_train, y_names)
    elliptic_envelope(X_train, y_names)


run_outlier_detection()
