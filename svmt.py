import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
import time
data = pd.read_csv("C:/Users/Lenova/Desktop/archive/diabetes.csv")
X = data.drop(columns=["Pregnancies"])
y = data["Pregnancies"]
y_sparse = csr_matrix(y.values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_sparse, test_size=0.25, random_state=42)
base_classifier = SVC(kernel='linear')
classifier = BinaryRelevance(classifier=base_classifier)
classifier.fit(X_train, y_train)
y_pred_sparse = classifier.predict(X_test)
y_pred = y_pred_sparse.toarray().reshape(-1)
accuracy = accuracy_score(y_test.toarray().reshape(-1), y_pred)
start_time = time.time()
classifier.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time
print("Время обучения модели:", training_time, "секунд")
print("Точность SVMT:", accuracy)
