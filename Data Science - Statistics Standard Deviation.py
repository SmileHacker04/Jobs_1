import pandas as pd
import numpy as np

diabetes_data = pd.read_csv("D:/Users/Lenovo/Desktop/archive/diabetes.csv", header=0, sep=",")

std_per_column = diabetes_data.std()
cv_per_column = std_per_column / diabetes_data.mean()

print("Әр баған үшін стандартты ауытқу:")
print(std_per_column)
print("\nӘр баған үшін вариация коэффициенті:")
print(cv_per_column)
