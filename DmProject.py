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
encoded_train_data = pd.get_dummies(train_data) # change categorical values
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
