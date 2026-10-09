import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("covid_19.csv")

print(df.head())
print(df.shape)
 

print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.isnull().sum().sum())


df["continent"] = df["continent"].fillna(df["continent"].mode()[0])


num_cols = df.select_dtypes(include=np.number).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print("Missing values after cleaning:")
print(df.isnull().sum())

print(df.head())


X = df.drop(columns=["country", "time"], errors="ignore")


X = pd.get_dummies(X, drop_first=True, dtype=int)

print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("reduced data:")
print( X_pca[:5])

print("Explained variance ratio:",pca.explained_variance_ratio_)

print("Total explained variance:")
print(pca.explained_variance_ratio_.sum())

print("Principal components:")
print(pca.components_)

print("Original data shape:", X.shape)
print("Reduced data shape:", X_pca.shape)


plt.scatter(X_pca[:, 0], X_pca[:, 1])

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on COVID-19 Dataset")

plt.grid(True)
plt.show()

plt.bar(
    ["PC1", "PC2"],
    pca.explained_variance_ratio_
)

plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance")

plt.show()


top_cases = df.nlargest(7, "Cases")

plt.figure(figsize=(10, 5))
plt.bar(top_cases["country"], top_cases["Cases"])
plt.xlabel("Country")
plt.ylabel("Total Cases")
plt.title("Top 7 Countries with Highest COVID-19 Cases")
plt.show()



top_deaths = df.nlargest(7, "Deaths")

plt.figure(figsize=(10, 5))
plt.bar(top_deaths["country"], top_deaths["Deaths"])
plt.xlabel("Country")
plt.ylabel("Total Deaths")
plt.title("Top 7 Countries with Highest COVID-19 Deaths")
plt.show()

