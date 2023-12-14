from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt

# Генерация синтетических данных для примера
np.random.seed(0)
X = np.random.rand(100, 1) * 2 - 1
y = 2 * (X > 0).astype(int) - 1

# Создание и обучение BrownBoost модели
brownboost = GradientBoostingClassifier(loss='exponential', n_estimators=100, learning_rate=1.0, random_state=0)
brownboost.fit(X, y)

# Построение графика
plt.figure(figsize=(12, 6))
plt.scatter(X, y, c='gray', marker='o', label='Точки данных')
x_vals = np.linspace(-1, 1, 400).reshape(-1, 1)
y_prob = np.exp(-1 * brownboost.decision_function(x_vals))
plt.plot(x_vals, y_prob, c='green', linewidth=2, label='Вероятность BrownBoost')
plt.xlabel('Признак')
plt.ylabel('Вероятность')
plt.legend()
plt.show()
