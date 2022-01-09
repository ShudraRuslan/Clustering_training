import sklearn.datasets.samples_generator as sdsg
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
from pandas import DataFrame
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds
import sklearn.preprocessing as skprep


def visualizationData2(X, y_true):
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y_true))
    colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'cyan', 5: 'black', 6: 'orange', 7: 'violet'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()


def visualizationData1(X, y_true):
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y_true))
    colors = {1: 'red', 0: 'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()


def visualizationClusters(X, y_pred):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Accent')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc=2)
    plt.show()


def testOPTICSModel(X, y_true, min_samples, min_cluster_size, p=2):
    model = OPTICS(min_samples=min_samples, min_cluster_size=min_cluster_size, p=p)
    y_pred = model.fit_predict(X)
    visualizationClusters(X, y_pred)
    print("Adjusted Rand Index: ", sklm.adjusted_rand_score(y_true, y_pred))
    print("Adjusted Mutual Info: ", sklm.adjusted_mutual_info_score(y_true, y_pred))
    print("Silhouette Coefficient: ", sklm.silhouette_score(X, y_pred))
    print("Amount of clusters: ", len(set(model.labels_)))


def testDBSCANModel(X, y_true, eps, min_samples, min_cluster_size, p=2):
    helpModel = OPTICS(min_samples=min_samples, min_cluster_size=min_cluster_size, p=p)
    helpModel.fit(X)
    y_pred = cluster_optics_dbscan(reachability=helpModel.reachability_,
                                   core_distances=helpModel.core_distances_,
                                   ordering=helpModel.ordering_,
                                   eps=eps)
    visualizationClusters(X, y_pred)
    print("Adjusted Rand Index: ", sklm.adjusted_rand_score(y_true, y_pred))
    print("Adjusted Mutual Info: ", sklm.adjusted_mutual_info_score(y_true, y_pred))
    if len(set(y_pred)) > 1:
        print("Silhouette Coefficient: ", sklm.silhouette_score(X, y_pred))
    print("Amount of clusters: ", len(set(y_pred)))


def testStability(x, y):
    helpArray = np.hstack((x, y.reshape(-1, 1)))
    testOPTICSModel(x, y, 15, 30, 1)
    np.random.shuffle(helpArray)
    x = helpArray[:, 0:2]
    y = helpArray[:, 2]
    testOPTICSModel(x, y, 15, 30, 1)


x1, y1 = sdsg.make_circles(500, factor=.1, noise=.1)
x2_help = np.load('first_var.npy')
y2_help = np.load('targets.npy')
x2 = x2_help[:1000]
y2 = y2_help[:1000]


# visualizationData1(x1, y1)
# visualizationData2(x2, y2)

# testOPTICSModel(x1,y1,2,10)
# testOPTICSModel(x1,y1,10,20)
# testOPTICSModel(x1,y1,15,30)
# testOPTICSModel(x2,y2,2,10)
# testOPTICSModel(x2,y2,10,20)
# testOPTICSModel(x2,y2,15,30)

# testOPTICSModel(x1, y1, 15, 30, 1)
# testOPTICSModel(x2,y2,2,10,1)

# testDBSCANModel(x1, y1, 0.1, 15, 30, 1)
# testDBSCANModel(x1, y1, 0.5, 15, 30, 1)
# testDBSCANModel(x1, y1, 1, 15, 30, 1)

# testDBSCANModel(x2, y2, 0.1, 2, 10)
# testDBSCANModel(x2, y2, 0.5, 2,10)

# normalizer=skprep.Normalizer(norm='l2').fit(x2)
# new_data=normalizer.transform(x2)
# testOPTICSModel(new_data,y2,2,10)
# testOPTICSModel(new_data,y2,15,30)

# scaler=skprep.StandardScaler().fit(x2)
# new_data=scaler.transform(x2)
# testOPTICSModel(new_data,y2,2,10)
# testOPTICSModel(new_data,y2,15,30)

# testStability(x1,y1)