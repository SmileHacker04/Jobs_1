import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
# Создание синтетических данных
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)
# Обучение модели LogitBoost
n_estimators = 100
learning_rate = 1.0
clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, loss='exponential')
clf.fit(X, y)
# График решающей функции модели
plot_step = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), cmap=plt.cm.RdBu, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k', cmap=plt.cm.RdBu, s=50)
plt.xlabel('Особенность 1')
plt.ylabel('Особенность 2')
plt.title('Граница принятия решения LogitBoost')
plt.show()
