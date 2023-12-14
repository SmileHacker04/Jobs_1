import sys
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
diabetes_data = pd.read_csv("D:/Users/Lenovo/Desktop/archive/diabetes.csv")
columns_to_plot = ['Glucose', 'BloodPressure']
selected_columns = diabetes_data[columns_to_plot]
selected_columns.plot(x=columns_to_plot[0], y=columns_to_plot[1], kind='scatter')
plt.savefig('scatter_plot.png')
