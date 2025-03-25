# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Charger les données Airbnb
df = pd.read_csv("airbnb_data.csv")

# Sélection des colonnes utiles
df = df[['name', 'neighbourhood', 'room_type', 'price', 'minimum_nights', 'number_of_reviews']]

# Nettoyage des données
df = df.drop_duplicates()
df = df.dropna()
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# Analyse rapide
print(df.describe())
sns.histplot(df['price'], bins=50, kde=True)
plt.show()

# Préparation des données pour l'IA
X = df[['minimum_nights', 'number_of_reviews']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle de régression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Résultat
print("Erreur moyenne absolue :", mean_absolute_error(y_test, y_pred))
