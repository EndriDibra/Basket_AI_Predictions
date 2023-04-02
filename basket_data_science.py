# Author: Endri Dibra

# importing the required libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


# data to be processed
data = {"Height": [1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81 ,1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81 ,1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81 ,1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81 ,1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81 ,1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81 ,1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81 ,1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81 ,1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81 ,1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81 ,1.90, np.nan, 1.95, 1.92, 2.02, 1.97, 1.94, np.nan, 1.99, 1.70, 1.90, 1.77, 1.72, 1.92, 1.79, 1.74, 1.94, 1.81],
        "Weight": [92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70 ,92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70 ,92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70 ,92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70 ,92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70 ,92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70 ,92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70 ,92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70 ,92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70 ,92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70 ,92, 103, 97, 92, 103, 97, 92, np.nan, 97, 62, 80, 70, 62, 80, 70, 62, 80, 70],
        "Gender": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Position": ["Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard" ,"Play_Maker", "Small_Forward", "Point_Guard"]}

df = pd.DataFrame(data)

print(df.head())

print("Full dataset:")
print(df.to_string())

df["Height"] = df["Height"].fillna(df["Height"].mean())
df["Weight"] = df["Weight"].fillna(df["Weight"].mean())

print(df)

print(df.shape)

print(df.info())

print(df.describe())

# preparing for model creation
X = df.drop(columns=["Position"])
y = df["Position"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train.values, y_train)

# calculating predictions
predictions = model.predict([[92, 1, 1.90], [62, 0, 1.70]])
print(predictions)

# calculating accuracy of model
predictions_ = model.predict(X_test.values)

score = accuracy_score(y_test, predictions_)

print(score)

# visualizing tree model
tree.export_graphviz(model, out_file="basket.dot", feature_names=["Position", "Gender", "Height"], class_names=sorted(y.unique()),
                     label="all", rounded=True, filled=True)