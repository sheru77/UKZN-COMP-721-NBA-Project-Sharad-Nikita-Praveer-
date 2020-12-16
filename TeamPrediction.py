# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:37:11 2020

@author: sheru
"""

import pandas as pd
from sklearn import preprocessing
import numpy as np

from sklearn.preprocessing import StandardScaler

# LOADING DATA Start
data = pd.read_csv("team_season.csv")
y_names = data.iloc[:, [0, 1]]


data = data[data['year'] >= 1980]

X_train = data.iloc[:len(data) - 30,
          [3, 4, 5, 7, 10, 13, 15, 18, 19, 20, 28, 29, 30]]
X_test = data.iloc[len(data) - 30:, [3, 4, 5, 7, 10, 13, 15, 18, 19, 20, 28, 29, 30]]
y_train = data.iloc[:len(data) - 30, [36]]
y_test = data.iloc[len(data) - 30:, [36]]


# Normalize data
sc_X = StandardScaler()
sc_y = StandardScaler()
sc_Xt = StandardScaler()
X = sc_X.fit_transform(X_train)
y = sc_y.fit_transform(y_train.values)
Xt = sc_Xt.fit_transform(X_test)

print(" This is SVR")
from sklearn.svm import SVR

SVR_model = SVR(kernel='rbf')
SVR_model.fit(X, y)
y_predict_SVR = SVR_model.predict(Xt)
y_predict_SVR = sc_y.inverse_transform(y_predict_SVR)

X_test2 = X_test.copy(deep=True)

X_test2['y_predict_SVR'] = y_predict_SVR
X_test2['team'] = data['team']
print(X_test2[['y_predict_SVR', 'team']].head(30))
Team1 = input("Enter team 1 Abbreviation: ")
Team1 = Team1.upper()
Team2 = input("Enter team 2 Abbreviation: ")
Team2 = Team2.upper()
Team11 = X_test2[X_test2['team'] == Team1]
Team22 = X_test2[X_test2['team'] == Team2]
WinPercentageT1 = Team11['y_predict_SVR'].values
print("A head to head of: " + Team1 + " vs " + Team2)
WinPercentageT2 = Team22['y_predict_SVR'].values
if WinPercentageT1 > WinPercentageT2:
    print(Team1 + " wins")
else:
    print(Team2 + " wins")

print('############################################################################')
from sklearn.linear_model import LinearRegression

print(" This is Linear Regression")
model = LinearRegression().fit(X_train, y_train)
X_test3 = X_test.copy(deep=True)
y_pred = model.predict(X_test)
X_test3['y_pred'] = y_pred

from sklearn.metrics import r2_score

r_squared = r2_score(y_test, y_pred)
print('coefficient of determination:', r_squared)

X_test3['team'] = data['team']


print(X_test3[['y_pred', 'team']].head(30))

Team11 = X_test3[X_test3['team'] == Team1]
Team22 = X_test3[X_test3['team'] == Team2]
WinPercentageT1 = Team11['y_pred'].values
print("A head to head of: " + Team1 + " vs " + Team2)
WinPercentageT2 = Team22['y_pred'].values
if WinPercentageT1 > WinPercentageT2:
    print(Team1 + " wins")
else:
    print(Team2 + " wins")

print('############################################################################################')
from sklearn.ensemble import RandomForestRegressor

print("This is RandomForest")

regressor = RandomForestRegressor(n_estimators=100, random_state=0)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
X_test['y_pred'] = y_pred
X_test['team'] = data['team']
print(X_test[['y_pred', 'team']].head(30))
Team11 = X_test[X_test['team'] == Team1]
Team22 = X_test[X_test['team'] == Team2]
WinPercentageT1 = Team11['y_pred'].values
print("A head to head of: " + Team1 + " vs " + Team2)
WinPercentageT2 = Team22['y_pred'].values
if WinPercentageT1 > WinPercentageT2:
    print(Team1 + " wins")
else:
    print(Team2 + " wins")
print("##########################################################################################")
dataoutlier = pd.read_csv("player_regular_season.csv")
y_names = dataoutlier.iloc[:, [0, 1, 2, 3, 4]]  # team name, year

df = dataoutlier[dataoutlier['year'] >= 2004]


df = df[(df['team'] == Team1) | (df['team'] == Team2)]


print(df[['ilkid', 'gp', 'team']].head(100))

features = ['minutes', 'pts', 'reb', 'asts', 'stl', 'blk', 'turnover', 'pf', 'fga', 'fgm', 'fta', 'ftm', 'tpa', 'tpm']

X_train = df.iloc[:, [6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]

y_names = df.iloc[:, [0, 1, 2, 3, 4, ]]

x = X_train.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

X_train = pd.DataFrame(x_scaled)
X_train.columns = X_train.iloc[0, :]

from sklearn.cluster import DBSCAN

outlier_detection = DBSCAN(eps=0.5, metric="euclidean", min_samples=1.5)
clusters = outlier_detection.fit(X_train)


outlier_df = pd.DataFrame(X_train)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([.1, .1, 1, 1])
colours = clusters.labels_

from matplotlib import cm

cmap = cm.get_cmap('bwr')

ax.scatter(X_train.iloc[:, 2].values, X_train.iloc[:, 1].values, c=colours, s=20, cmap=cmap, edgecolor="black")

ax.set_xlabel("Points")
ax.set_ylabel("Minutes")

plt.title("DBSCAN Outlier Detection")
plt.show()

outliers = outlier_df[clusters.labels_ == -1]
outliers = outliers[:].index.to_numpy()

y_names = y_names.values

print("DBSCAN OUTLIERS:")
for i in range(0, len(outliers)):
    print(outliers[i], "-> ", y_names[outliers[i]])
print("Number of outliers:", len(outliers))

print('#################################################')

from sklearn.cluster import KMeans

x = X_train.values
wcss = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 16), wcss)
plt.show()
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=5)
y = kmeans.fit_predict(X_train)

center = kmeans.cluster_centers_

print('###########')

plt.scatter(x[y == 0, 0], x[y == 0, 1], s=25, c='indigo', label='Cluster0')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s=25, c='sienna', label='Cluster1')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s=25, c='crimson', label='Cluster2')

plt.scatter(x[y == 3, 0], x[y == 3, 1], s=25, c='cyan', label='Cluster3')
'''
plt.scatter(x[y==4,0],x[y==4,1], s=25,c='orange',label='Cluster5')

plt.scatter(x[y==5,0],x[y==5,1], s=25,c='grey',label='Cluster6')

plt.scatter(x[y==6,0],x[y==6,1], s=25,c='royalblue',label='Cluster7')

plt.scatter(x[y==7,0],x[y==7,1], s=25,c='green',label='Cluster8')
plt.scatter(x[y==8,0],x[y==8,1], s=25,c='lightcoral',label='Cluster9')
plt.scatter(x[y==9,0],x[y==9,1], s=25,c='blue',label='Cluster10')
plt.scatter(x[y==10,0],x[y==10,1], s=25,c='lightgrey',label='Cluster11')
plt.scatter(x[y==11,0],x[y==11,1], s=25,c='cyan',label='Cluster12')
plt.scatter(x[y==12,0],x[y==12,1], s=25,c='goldenrod',label='Cluster13')
plt.scatter(x[y==13,0],x[y==13,1], s=25,c='maroon',label='Cluster14')
plt.scatter(x[y==14,0],x[y==14,1], s=25,c='blueviolet',label='Cluster15')
'''
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=25, c='yellow', label='centroid')
plt.legend()
plt.show()
cluster = int(input("Please enter cluster to output"))
cluster2 = int(input("Please enter second cluster to output"))
outlier_list = []
for i in range(0, len(y)):

    if y[i] == cluster:
        outlier_list.append(i)
    elif y[i] == cluster2:
        outlier_list.append(i)
print("K-Means OUTLIERS:")
for i in range(0, len(outlier_list)):
    print(outlier_list[i], "-> ", y_names[outlier_list[i]])
print("Number of outliers:", len(outlier_list))
