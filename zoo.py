import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Cargar los conjuntos de datos
zoo2_df = pd.read_csv('zoo2.csv')
zoo3_df = pd.read_csv('zoo3.csv')

# Dividir los conjuntos de datos en características (X) y etiquetas (y)
X_zoo2 = zoo2_df.drop('class_type', axis=1)
y_zoo2 = zoo2_df['class_type']

X_zoo3 = zoo3_df.drop('class_type', axis=1)
y_zoo3 = zoo3_df['class_type']

# Codificación one-hot de características categóricas
X_zoo2_encoded = pd.get_dummies(X_zoo2)
X_zoo3_encoded = pd.get_dummies(X_zoo3)

# Dividir los conjuntos de datos en conjuntos de entrenamiento y prueba
X_train_zoo2, X_test_zoo2, y_train_zoo2, y_test_zoo2 = train_test_split(X_zoo2_encoded, y_zoo2, test_size=0.2, random_state=42)
X_train_zoo3, X_test_zoo3, y_train_zoo3, y_test_zoo3 = train_test_split(X_zoo3_encoded, y_zoo3, test_size=0.2, random_state=42)

# Entrenar modelos y calcular la precisión
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machines': SVC(),
    'Naive Bayes': GaussianNB()
}

accuracies_zoo2 = []
accuracies_zoo3 = []

for name, model in models.items():
    model.fit(X_train_zoo2, y_train_zoo2)
    y_pred = model.predict(X_test_zoo2)
    accuracy_zoo2 = accuracy_score(y_test_zoo2, y_pred)
    accuracies_zoo2.append(accuracy_zoo2)

    model.fit(X_train_zoo3, y_train_zoo3)
    y_pred = model.predict(X_test_zoo3)
    accuracy_zoo3 = accuracy_score(y_test_zoo3, y_pred)
    accuracies_zoo3.append(accuracy_zoo3)

# Visualización de la cantidad de animales por clase
plt.figure(figsize=(10, 6))
sns.countplot(x='class_type', data=pd.concat([zoo2_df, zoo3_df]))
plt.title('Cantidad de animales por clase')
plt.xlabel('Clase de animal')
plt.ylabel('Cantidad de animales')
plt.show()

# Graficar la precisión de cada modelo
plt.figure(figsize=(10, 6))
plt.bar(models.keys(), accuracies_zoo2, alpha=0.7, label='zoo2.csv')
plt.bar(models.keys(), accuracies_zoo3, alpha=0.7, label='zoo3.csv')
plt.title('Precisión de modelos en zoo2.csv y zoo3.csv')
plt.xlabel('Modelo')
plt.ylabel('Precisión')
plt.legend()
plt.ylim(0, 1)
plt.show()

# Imprimir resultados de la segunda figura en la consola
print("Resultados de la segunda figura:")
for model, acc_zoo2, acc_zoo3 in zip(models.keys(), accuracies_zoo2, accuracies_zoo3):
    print(f"Modelo: {model}, Precisión en zoo2.csv: {acc_zoo2}, Precisión en zoo3.csv: {acc_zoo3}")
