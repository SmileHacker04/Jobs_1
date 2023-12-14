import pandas as pd
import numpy as np

diabetes_data = pd.read_csv("D:/Users/Lenovo/Desktop/archive/diabetes.csv", header=0, sep=",")
var_per_column = diabetes_data.var()
print("Әр баған үшін ауытқу:")
print(var_per_column)
