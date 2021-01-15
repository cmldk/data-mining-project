from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as seb
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

def currentTime():
  return int(round(time.time() * 1000))

start = currentTime()
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
#print(encoded_train_data.head)
data_x = encoded_train_data.drop(["id", "netgain"], axis=1)  # data as dropped unnecessary columns
data_y = encoded_train_data["netgain"]  # target data as netgain column

# Applying PCA
pca = PCA(n_components=2)
data_x_pca = pca.fit_transform(data_x)
#print(data_x_pca)

# Correlation Matrix
corr = train_data.corr()
corr.to_csv("corr.csv")
corr.to_html("corr.html")

#StandardScaler
sc = StandardScaler()
data_y = train_data["netgain"]
dd = train_data.apply(LabelEncoder().fit_transform)
dc = dd[dd.airlocation != "Holand-Netherlands"]
df = dc.drop(["id", "netgain"], axis=1)
scaled_data = sc.fit_transform(df)
#print(scaled_data)


def findAccuracy(algo, algo_name, X_train, X_test, y_train, y_test):
  start_time = currentTime()
  algo.fit(X_train, y_train)
  y_pred_train = algo.predict(X_train)
  algo_train_score = accuracy_score(y_train, y_pred_train)  # Accuracy score of train set
  y_pred_test = algo.predict(X_test)
  algo_test_score = accuracy_score(y_test, y_pred_test)  # Accuracy score of test set
  passing_time = currentTime() - start_time
  passing_times.append(passing_time)
  print(algo_name + " \nTrain Accuracy : {0} , Test Accuracy : {1} and Passing Time : {2}ms".format(algo_train_score, algo_test_score, passing_time))
  return algo_train_score, algo_test_score

# LINEAR CLASSIFICATION TASKS
log_reg = LogisticRegression(max_iter=1000) # Logistic Regression Process
slp = Perceptron()                          # Single Layer Perceptron Process
sgd = SGDClassifier()                       # Stochastic Gradient Descent Classifier

# NEURAL NETWORK BASED CLASSIFICATION
mlp = MLPClassifier(hidden_layer_sizes=(16,8,4,2), max_iter=1000) # Multi Layer Perceptron Classifier

# ENSEMBLE LEARNING BASED CLASSIFICATION
bag = BaggingClassifier()                   # Bagging
rfc = RandomForestClassifier()              # Random Forest Classifier
adb = AdaBoostClassifier()                  # Adaboost

for x,t in zip([data_x_pca, scaled_data, data_x],["PCA", "Scaled", "Normal"]):
  passing_times = []

  X_train, X_test, y_train, y_test = train_test_split(x, data_y, train_size=0.7)  # split data into train and test
  print("\n----------------"+ t + " Results-------------")
  logistic_train_score, logistic_test_score = findAccuracy(log_reg, "LOGISTIC REGRESSION", X_train, X_test, y_train, y_test)
  slp_train_score, slp_test_score = findAccuracy(slp, "SINGLE LAYER PERCEPTRON", X_train, X_test, y_train, y_test)
  sgd_train_score, sgd_test_score = findAccuracy(sgd, "STOCHASTIC GRADIENT DESCENT CLASSIFIER", X_train, X_test, y_train, y_test)
  mlp_train_score, mlp_test_score = findAccuracy(mlp, "MULTI LAYER PERCEPTRON CLASSIFIER", X_train, X_test, y_train, y_test)
  bag_train_score, bag_test_score = findAccuracy(bag, "BAGGING CLASSIFIER", X_train, X_test, y_train, y_test)
  rfc_train_score, rfc_test_score = findAccuracy(rfc, "RANDOM FOREST CLASSIFIER", X_train, X_test, y_train, y_test)
  adb_train_score, adb_test_score = findAccuracy(adb, "ADABOOST CLASSIFIER", X_train, X_test, y_train, y_test)

  algorithms = ["Logistic Regression","Single Layer Perceptron","SGD classifier","MLP classifier","Bagging Classifier","Random Forest Classifier","Adaboost Classifier"]
  algo_to_name = {log_reg: "Logistic Regression",
                  slp: "Single Layer Perceptron",
                  sgd: "SGD classifier",
                  mlp: "MLP classifier",
                  bag: "Bagging Classifier",
                  rfc: "Random Forest Classifier",
                  adb: "Adaboost Classifier"}
  train_scores = [logistic_train_score,slp_train_score,sgd_train_score,mlp_train_score,bag_train_score,rfc_train_score,adb_train_score]
  test_scores = [logistic_test_score,slp_test_score,sgd_test_score,mlp_test_score,bag_test_score,rfc_test_score,adb_test_score]
  algo_to_test = {log_reg: logistic_test_score,
                  slp: slp_test_score,
                  sgd: sgd_test_score,
                  mlp: mlp_test_score,
                  bag: bag_test_score,
                  rfc: rfc_test_score,
                  adb: adb_test_score}

  fig2 = plt.figure(figsize=(20,5))
  ax = fig2.add_subplot(1,2,1)
  true_values = []
  false_values = []
  names = []
  sorted_dict = sorted(algo_to_test, key=algo_to_test.get, reverse=True)
  for i in range(2):
    algo = sorted_dict[i]
    names.append(algo_to_name.get(algo))
    y_pred = algo.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    true_values.append(conf_matrix[0][0] + conf_matrix[1][1])
    false_values.append(conf_matrix[0][1] + conf_matrix[1][0])
 
  ind = np.arange(len(names))
  width = 0.4
  ax.barh(ind, false_values, width, color='red', label='False values')
  ax.barh(ind + width, true_values, width, color='green', label='True values')
  # Remove axes splines 
  for s in ['top', 'bottom', 'left', 'right']: 
      ax.spines[s].set_visible(False) 
  # Remove x, y Ticks 
  ax.xaxis.set_ticks_position('none') 
  ax.yaxis.set_ticks_position('none') 
  # Add padding between axes and labels 
  ax.xaxis.set_tick_params(pad = 5) 
  ax.yaxis.set_tick_params(pad = 10) 
  # Add x, y gridlines 
  ax.grid(b = True, color ='grey', 
          linestyle ='-.', linewidth = 0.5, 
          alpha = 0.2) 
  for i in ax.patches: 
    plt.text(i.get_width()+0.2, i.get_y()+0.18,  
             str(round((i.get_width()), 2)), 
             fontsize = 8, fontweight ='bold')
  ax.set(yticks=ind + 0.2, yticklabels=names, ylim=[width - 1, len(names)])
  title = 'True&False values of 2 Algorithms that have highest test score for {0}'.format(t)
  ax.set_title(title, loc ='left')
  ax.legend()

  fig2.add_subplot(1,2,2)
  colors = ['red', 'green', 'yellow', 'orange', 'purple','blue', '#b4b85e']
  ts = []
  for s in test_scores:
    ts.append("{:.3f}".format(s))
  for i,s in enumerate(test_scores):
    plt.scatter(test_scores[i], passing_times[i], color=colors[i], label=algorithms[i])
    plt.annotate(ts[i], (test_scores[i], passing_times[i]))
  plt.legend(loc='upper left')
  plt.grid()
  plt.ylabel("Passing Times in ms")
  plt.xlabel("Accuracy scores")
  plt.title(t)
  plt.show()


test_data_encoded = pd.get_dummies(test_data).drop(["id"], axis=1)
net_gain_prediction = adb.predict(test_data_encoded)
result = pd.DataFrame({"id": test_data.id, "realtionship_status": test_data.realtionship_status,
                       "industry": test_data.industry,
                       "genre": test_data.genre,
                       "targeted_sex": test_data.targeted_sex,
                       "average_runtime(minutes_per_week)": test_data["average_runtime(minutes_per_week)"],
                       "airtime": test_data.airtime,
                       "airlocation": test_data.airlocation,
                       "ratings": test_data.ratings,
                       "expensive": test_data.expensive,
                       "money_back_guarantee": test_data.money_back_guarantee,
                       "netgain_prediction": net_gain_prediction})
result.to_csv("results.csv")

fig = plt.figure(figsize=(12,5))
fig.add_subplot(1,2,1)
seb.countplot(data=train_data, x="genre", hue="netgain",palette="Set1",
              order=train_data["genre"].value_counts().index)
plt.title("Train data")
fig.add_subplot(1,2,2)
seb.countplot(data=result, x="genre", hue="netgain_prediction",palette="Set1",
              order=result["genre"].value_counts().index)
plt.title("Test data")
plt.savefig("Netgain based comparison for genre.pdf")

end = currentTime() - start
print("Total Runtime: {0}sn".format(end/1000))