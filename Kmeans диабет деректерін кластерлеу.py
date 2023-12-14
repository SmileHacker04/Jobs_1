import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
file_path = 'C:/Users/Lenova/Desktop/archive/diabetes.csv'
data = pd.read_csv(file_path)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[features]
num_clusters = 3  
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
kmeans.fit(X)
data['Кластер'] = kmeans.labels_
for cluster_num in range(num_clusters):
    cluster_data = data[data['Кластер'] == cluster_num]
    plt.scatter(cluster_data['Glucose'], cluster_data['Insulin'], label=f'Кластер {cluster_num}')

plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 4], marker='*', s=200, c='black', label='Центроидтар')
plt.xlabel('Глюкоза')
plt.ylabel('Инсулин')
plt.title('KMeans әдісі бойынша қант диабеті туралы деректерді кластерлеу.')
plt.legend()
plt.show()
