from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


def currentTime():
    return int(round(time.time() * 1000))


# Load data
train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv("Test.csv")

# DATA PREPROCESSING
print("DATA TYPES-----------")
print(train_data.dtypes)
print(test_data.dtypes)

print("\nNULL VALUES--------")
print(train_data.isnull().sum())
print(test_data.isnull().sum())

# DATA ENCODING
print("\nENCODING--------")
le = LabelEncoder()
# change bool type of netgain column to int type with 1 for True and 0 for False
# change string type of money_back_guarantee column to in type with 1 for Yes and 0 for No
train_data["netgain"] = le.fit_transform(train_data["netgain"])
train_data = train_data[train_data.airlocation != "Holand-Netherlands"]
encoded_train_data = pd.get_dummies(train_data)  # change categorical values
print(encoded_train_data.head)
data_x = encoded_train_data.drop(["id", "netgain"], axis=1)  # data as dropped unnecessary columns
data_y = encoded_train_data["netgain"]  # target data as netgain column

# Applying PCA
pca = PCA(n_components=2)
data_x_pca = pca.fit_transform(data_x)
print(data_x_pca)

# Correlation Matrix
corr = train_data.corr()
corr.to_csv("corr.csv")
corr.to_html("corr.html")


def findAccuracy(algo, algo_name, X_train, X_test, y_train, y_test):
    start_time = currentTime()
    algo.fit(X_train, y_train)
    y_pred_train = algo.predict(X_train)
    algo_train_score = accuracy_score(y_train, y_pred_train)  # Accuracy score of train set
    y_pred_test = algo.predict(X_test)
    algo_test_score = accuracy_score(y_test, y_pred_test)  # Accuracy score of test set
    passing_time = currentTime() - start_time
    passing_times.append(passing_time)
    print(algo_name + " \nTrain Accuracy : {0} , Test Accuracy : {1} and Passing Time : {2}ms".format(algo_train_score,
                                                                                                      algo_test_score,
                                                                                                      passing_time))
    return algo_train_score, algo_test_score


# LINEAR CLASSIFICATION TASKS
log_reg = LogisticRegression(max_iter=1000)  # Logistic Regression Process
slp = Perceptron()  # Single Layer Perceptron Process
sgd = SGDClassifier()  # Stochastic Gradient Descent Classifier

# NEURAL NETWORK BASED CLASSIFICATION
mlp = MLPClassifier(hidden_layer_sizes=(16, 8, 4, 2), max_iter=1000)  # Multi Layer Perceptron Classifier

# ENSEMBLE LEARNING BASED CLASSIFICATION
bag = BaggingClassifier()  # Bagging
rfc = RandomForestClassifier()  # Random Forest Classifier
adb = AdaBoostClassifier()  # Adaboost

fig = plt.figure(figsize=(12, 5))
for x, t, j in zip([data_x_pca, data_x], ["PCA", "Normal"], range(1, 3)):
    passing_times = []
    fig.add_subplot(1, 2, j)

    X_train, X_test, y_train, y_test = train_test_split(x, data_y, train_size=0.7)  # split data into train and test
    print("\n----------------" + t + " Results-------------")
    logistic_train_score, logistic_test_score = findAccuracy(log_reg, "LOGISTIC REGRESSION", X_train, X_test, y_train,
                                                             y_test)
    slp_train_score, slp_test_score = findAccuracy(slp, "SINGLE LAYER PERCEPTRON", X_train, X_test, y_train, y_test)
    sgd_train_score, sgd_test_score = findAccuracy(sgd, "STOCHASTIC GRADIENT DESCENT CLASSIFIER", X_train, X_test,
                                                   y_train, y_test)
    mlp_train_score, mlp_test_score = findAccuracy(mlp, "MULTI LAYER PERCEPTRON CLASSIFIER", X_train, X_test, y_train,
                                                   y_test)
    bag_train_score, bag_test_score = findAccuracy(bag, "BAGGING CLASSIFIER", X_train, X_test, y_train, y_test)
    rfc_train_score, rfc_test_score = findAccuracy(rfc, "RANDOM FOREST CLASSIFIER", X_train, X_test, y_train, y_test)
    adb_train_score, adb_test_score = findAccuracy(adb, "ADABOOST CLASSIFIER", X_train, X_test, y_train, y_test)

    algorithms = ["Logistic Regression", "Single Layer Perceptron", "SGD classifier", "MLP classifier",
                  "Bagging Classifier", "Random Forest Classifier", "Adaboost Classifier"]
    train_scores = [logistic_train_score, slp_train_score, sgd_train_score, mlp_train_score, bag_train_score,
                    rfc_train_score, adb_train_score]
    test_scores = [logistic_test_score, slp_test_score, sgd_test_score, mlp_test_score, bag_test_score, rfc_test_score,
                   adb_test_score]
    algos = []
    for a in range(len(algorithms)):
        tmp = algorithms[a] + " \n" + str(passing_times[a]) + "ms"
        algos.append(tmp)

    for i, title in zip([train_scores, test_scores], [" Train", " Test"]):
        plt.plot(algos, i, 'o-', label=(t + title))
        plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel("Algorithms & Passing Times")
    plt.ylabel("accuracy scores")

    print("\nConfusion Matrix of Adaboost Classifier")
    y_pred = adb.predict(X_test)
    print(confusion_matrix(y_test, y_pred))