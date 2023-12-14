import pandas as pd

diabetes_data = pd.read_csv("D:/Users/Lenovo/Desktop/archive/diabetes.csv", header=0, sep=",")

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

print (diabetes_data.describe())