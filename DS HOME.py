import pandas as pd									
import matplotlib.pyplot as plt							
from scipy import stats									
diabetes_data = pd.read_csv("C:/Users/Lenova/Desktop/archive/diabetes.csv", header=0, sep=",")										
x = diabetes_data["Glucose"]								
y = diabetes_data["BMI"]									
slope, intercept, r, p, std_err = stats.linregress(x, y)				
def myfunc(x):										
    return slope * x + intercept							
mymodel = list(map(myfunc, x))								
plt.scatter(x, y)										
plt.plot(x, mymodel, color="red")  							
plt.xlabel("Glucose")									
plt.ylabel("BMI")										
plt.title("Сызықтық регрессия: BMI-ге байланысты қандағы глюкоза деңгейі")	
plt.show()											
