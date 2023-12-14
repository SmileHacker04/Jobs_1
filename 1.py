import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

# Предположим, что у нас есть две матрицы: реальные рейтинги (actual) и предсказанные рейтинги (predicted)
# Эти матрицы могут быть созданы на основе вашего алгоритма рекомендаций

# Пример данных (замените их реальными данными)
actual = np.array([4, 5, 3, 2, 4, 5, 1, 3, 2, 4])
predicted = np.array([4.2, 4.8, 3.2, 2.5, 4.0, 4.9, 1.1, 3.4, 2.1, 4.3])

# Выберите значение K для оценки точности и полноты в топ-K рекомендациях
K = 5

# Вычислите точность, полноту и F-меру для топ-K рекомендаций
precision = precision_score(actual >= 4, predicted >= 4, average='binary')
recall = recall_score(actual >= 4, predicted >= 4, average='binary')
f1 = f1_score(actual >= 4, predicted >= 4, average='binary')

# Вычислите среднюю абсолютную ошибку (MAE) и среднеквадратичную ошибку (MSE)
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)

# Выведите результаты оценки
print(f"Precision at {K}: {precision:.2f}")
print(f"Recall at {K}: {recall:.2f}")
print(f"F1 Score at {K}: {f1:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
