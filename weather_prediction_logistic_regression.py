#This is basically a machine learning model to predict the weather conditions in seattle based on input values.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("C:/Users/yahqu/Downloads/seattle-weather.csv")
data.info()

#drop unneccessary colums
data = data.drop("date", axis=1)

#Print the first 5 rows of data
print(data.head())

#Plot a pie chart to visualize the count of weather grouped values.
groups = data.groupby('weather').size().plot(kind='pie', autopct='%.2f')