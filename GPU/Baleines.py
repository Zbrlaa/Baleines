import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd

# 1. Chargement et préparation des données
data_raw = np.load("/scratch/Shawn/tdoa_results_s70.npy")
# Structure supposée : [nom_fichier, clic_index, canal_id, delta_t]
df = pd.DataFrame(data_raw, columns=['file', 'clic_idx', 'channel', 'dt'])
df[['clic_idx', 'channel', 'dt']] = df[['clic_idx', 'channel', 'dt']].apply(pd.to_numeric)

# 2. Pivotage pour avoir 1 ligne par clic avec ses 3 TDOA
# On regroupe par fichier et par index de clic
df_pivot = df.pivot_table(index=['file', 'clic_idx'], columns='channel', values='dt').dropna().reset_index()

# 3. Calcul du temps absolu (approximatif) pour l'axe X
# On considère que chaque fichier suit le précédent (environ 120s par fichier)
file_list = sorted(df_pivot['file'].unique())
file_map = {name: i * 120 for i, name in enumerate(file_list)}
df_pivot['time'] = df_pivot['file'].map(file_map) + (df_pivot['clic_idx'] / 384000)

# 4. Clustering avec DBSCAN (Déduit le nombre de baleines K)
# On cluster sur les 3 TDOA (canaux 1, 2, 3)
# eps est la distance maximale entre deux clics pour qu'ils soient de la même baleine
X = df_pivot[[1, 2, 3]].values
# db = DBSCAN(eps=8, min_samples=60).fit(X) # Ajuste eps selon tes résultats
db = DBSCAN(eps=10, min_samples=20).fit(X)
df_pivot['whale_id'] = db.labels_

# 5. Visualisation
plt.figure(figsize=(15, 7))
# On ne colorie que les clusters (whale_id >= 0), le bruit est en gris (-1)
scatter = plt.scatter(df_pivot['time'], df_pivot[1], c=df_pivot['whale_id'], cmap='tab20', s=1, alpha=0.5)
plt.colorbar(scatter, label='ID de la Baleine (Cluster)')
plt.xlabel("Temps (secondes)")
plt.ylabel("Délai TDOA (échantillons)")
plt.title(f"Détection des trajectoires : {len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)} baleines estimées")
plt.grid(True, alpha=0.3)
plt.savefig("Trajectoires_Baleines_s70.png")

n_whales = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"Nombre de baleines estimées par clustering : {n_whales}")