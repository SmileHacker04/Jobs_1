import pandas as pd
import matplotlib.pyplot as plt

diabetes_data = pd.read_csv("D:/Users/Lenovo/Desktop/archive/diabetes.csv")
columns_to_analyze = {'Глюкоза': diabetes_data['Glucose'], 'Қан_қысымы': diabetes_data['BloodPressure']}
columns_to_analyze_df = pd.DataFrame(data=columns_to_analyze)
columns_to_analyze_df.plot(x='Глюкоза', y='Қан_қысымы', kind='scatter')


plt.savefig('scatter_plot.png')

correlation_diabetes = columns_to_analyze_df.corr()
print(correlation_diabetes)
