from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = np.random.rand(100, 1) * 2 - 1
y = 2 * (X > 0).astype(int) - 1
# Создание и обучение GentleBoost модели
gentleboost = GradientBoostingClassifier(loss='exponential', n_estimators=100, learning_rate=1.0, random_state=0)
gentleboost.fit(X, y)
# Построение графика
plt.figure(figsize=(12, 6))
plt.scatter(X, y, c='gray', marker='o', label='Точки данных')
x_vals = np.linspace(-1, 1, 400).reshape(-1, 1)
y_pred = np.array([gentleboost.predict(x_val.reshape(1, -1))[0] for x_val in x_vals])
plt.plot(x_vals, y_pred, c='red', linewidth=2, label='Предсказание GentleBoost')
plt.xlabel('Признак')
plt.ylabel('Целевая переменная')
plt.legend()
plt.show()
