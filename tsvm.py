import pandas as pd
from sklearn.model_selection import train_test_split
from tslearn.svm import TimeSeriesSVC
from sklearn.metrics import accuracy_score
import time
data = pd.read_csv("C:/Users/Lenova/Desktop/archive/diabetes.csv")
X = data.drop(columns=["Pregnancies"])
y = data["Pregnancies"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = TimeSeriesSVC(kernel="rbf")
start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Точность TSVM:", accuracy)
print("Время обучения модели:", end_time - start_time, "секунд")
