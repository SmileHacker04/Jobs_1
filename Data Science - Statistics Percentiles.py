import pandas as pd
import numpy as np

diabetes_data = pd.read_csv("D:/Users/Lenovo/Desktop/archive/diabetes.csv", header=0, sep=",")

Glucose = diabetes_data["Glucose"]
percentile20 = np.percentile(Glucose, 20)

print(percentile20)
