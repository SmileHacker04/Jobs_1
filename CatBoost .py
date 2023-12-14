# Импортируем необходимые библиотеки
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Загружаем данные
data = pd.read_csv('C:/Users/Lenova/Desktop/archive/diabetes.csv')  # Замените 'C:/Users/Lenova/Desktop/archive/diabetes.csv' на путь к вашему файлу с данными

# Определяем признаки (X) и целевую переменную (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель CatBoost
model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, loss_function='Logloss', random_seed=42)
model.fit(X_train, y_train)

# Делаем предсказания на тестовом наборе данных
predictions = model.predict(X_test)

# Вычисляем точность модели
accuracy = accuracy_score(y_test, predictions)
print(f'Точность модели: {accuracy:.2f}')

# Строим график важности признаков
feature_importance = model.get_feature_importance(type='FeatureImportance')
feature_names = X.columns
sorted_idx = feature_importance.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), feature_importance[sorted_idx], align='center')
plt.xticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx], rotation=90)
plt.xlabel('Признаки')
plt.ylabel('Важность')
plt.title('Важность признаков')
plt.show()
