import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# ============================================================
# Carga del archivo PARQUET
# (ajusta el nombre si es distinto)
# ============================================================
df = pd.read_parquet("data/application_.parquet")

print("Primeras columnas:", df.columns.tolist()[:20])
print("Dimensiones del dataset:", df.shape)

# ============================================================
# Variables numéricas utilizadas en el clustering
# ============================================================
columnas_numericas = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]

col_target = "TARGET"   # 0 = buen pagador, 1 = default

# Filtro para conservar solo filas completas
df_cluster = df[columnas_numericas + [col_target]].dropna().copy()

X = df_cluster[columnas_numericas].values

# ============================================================
# Escalamiento de los datos
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# Evaluación de K (Codo + Silhouette)
# ============================================================
inertia_values = []
silhouette_values = []
K_values = range(2, 8)

for k in K_values:
    modelo = KMeans(n_clusters=k, random_state=42, n_init="auto")
    etiquetas = modelo.fit_predict(X_scaled)

    inertia_values.append(modelo.inertia_)
    silhouette_values.append(silhouette_score(X_scaled, etiquetas))

    print(f"K={k}  Inercia={modelo.inertia_:.0f}  Silhouette={silhouette_values[-1]:.4f}")

# Activar modo interactivo para abrir varias ventanas a la vez
plt.ion()

# --------- Gráfico del método del codo ---------
plt.figure()
plt.plot(K_values, inertia_values, marker="o")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia")
plt.title("Método del Codo")
plt.grid()
plt.show(block=False)   # NO bloquea
plt.pause(0.001)        # Mantiene ventana abierta

# --------- Gráfico del índice de Silhouette ---------
plt.figure()
plt.plot(K_values, silhouette_values, marker="o")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Silhouette")
plt.title("Índice de Silhouette")
plt.grid()
plt.show(block=False)   # NO bloquea
plt.pause(0.001)        # Mantiene ventana abierta

# Mantener todo abierto hasta que tú cierres manualmente
input("\nPresiona ENTER cuando ya hayas cerrado las ventanas...\n")

# ============================================================
# Entrenamiento del modelo final
# ============================================================
K_OPTIMO = 3  # definido por el análisis anterior

kmeans_final = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init="auto")
df_cluster["CLUSTER"] = kmeans_final.fit_predict(X_scaled)

# ============================================================
# Análisis de los clusters resultantes
# ============================================================
print("\nTasa de default (TARGET) por cluster:")
print(df_cluster.groupby("CLUSTER")[col_target].mean())

print("\nPromedio de variables numéricas por cluster:")
print(df_cluster.groupby("CLUSTER")[columnas_numericas].mean())
