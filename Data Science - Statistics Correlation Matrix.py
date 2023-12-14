import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

diabetes_data = pd.read_csv("D:/Users/Lenovo/Desktop/archive/diabetes.csv", header=0, sep=",")
correlation_diabetes = diabetes_data.corr()

axis_corr = sns.heatmap(
correlation_diabetes,
vmin=-1, vmax=1, center=0,
cmap=sns.diverging_palette(50, 500, n=500),
square=True
)

plt.show()
