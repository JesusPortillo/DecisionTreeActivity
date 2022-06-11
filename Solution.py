import pandas as pd
import numpy as np
from sklearn import tree
import pydotplus as pyd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import statsmodels.api as sm


carseats = sm.datasets.get_rdataset("Carseats", "ISLR")
data = carseats.data

data["High_sales"] = np.where(data.Sales > 8, 0, 1)
data = data.drop(columns = "Sales")

shelveLocNormalized = {"Bad":0, "Medium":1, "Good":2}
urbanNormalized = {"Yes":1, "No":0}
usNormalized = {"Yes":1, "No":0}

data["ShelveLoc"] = data["ShelveLoc"].map(shelveLocNormalized)
data["Urban"] = data["Urban"].map(urbanNormalized)
data["US"] = data["US"].map(usNormalized) 

features = ["CompPrice", "Income", "Advertising", "Population", "Price", "ShelveLoc", "Age", "Education", "Urban", "US"]

x = data[features]
y = data["High_sales"]

dtree = DecisionTreeClassifier()
dtree = dtree.fit(x,y)
all_data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pyd.graph_from_dot_data(all_data)
graph.write_png("mydecisiontree.png")

img = pltimg.imread("mydecisiontree.png")
imgplot = plt.imshow(img)
plt.show()



