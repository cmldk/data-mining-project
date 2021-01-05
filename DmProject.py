import pandas as pd
import seaborn as seb
from sklearn.model_selection import train_test_split

# Load data
train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv("Test.csv")

# DATA PREPROCESSING
print(train_data.dtypes)
print(test_data.dtypes)

print(train_data.isnull().sum())
print(test_data.isnull().sum())

# change bool type of netgain column to int type with 1 for True and 0 for False

train_data["netgain"] = train_data["netgain"].astype(str)
train_data.netgain[train_data.netgain == "True"] = 1
train_data.netgain[train_data.netgain == "False"] = 0
train_data["netgain"] = train_data["netgain"].astype(int)

# change string type of money_back_guarantee column to in type with 1 for Yes and 0 for No

train_data.money_back_guarantee[train_data.money_back_guarantee == "Yes"] = 1
train_data.money_back_guarantee[train_data.money_back_guarantee == "No"] = 0

print(train_data.head())

encoded_train_data = pd.get_dummies(train_data)
print(encoded_train_data.head())

data_x = encoded_train_data.drop(["id", "netgain"], axis=1)  # data as dropped unnecessary columns
data_y = encoded_train_data["netgain"]                      # target data as netgain column


x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=.7)   # split data into train and test




