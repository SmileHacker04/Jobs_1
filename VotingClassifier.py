import pandas as pd #1
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier #2
from sklearn.linear_model import LogisticRegression #3
from sklearn.model_selection import train_test_split #4
from sklearn.metrics import accuracy_score #5
from sklearn.preprocessing import StandardScaler #6
df = pd.read_csv('C:/Users/Lenova/Desktop/archive/diabetes.csv') #7
X = df.drop('Outcome', axis=1) #8
y = df['Outcome'] #9
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #10
scaler = StandardScaler() #11
X_train_scaled = scaler.fit_transform(X_train) #12
X_test_scaled = scaler.transform(X_test) #13
rf_classifier = RandomForestClassifier(random_state=42) #14
lr_classifier = LogisticRegression(random_state=42) #15
gb_classifier = GradientBoostingClassifier(random_state=42) #16
voting_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('lr', lr_classifier),
    ('gb', gb_classifier)
], voting='soft') #17
voting_classifier.fit(X_train_scaled, y_train) #18
y_pred = voting_classifier.predict(X_test_scaled) #19
accuracy = accuracy_score(y_test, y_pred) #20
print(f'Модельдің дәлдігі: {accuracy:.0%}') #21
