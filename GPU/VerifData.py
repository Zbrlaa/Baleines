import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob
import os

# Configuration
data_dir = "/scratch/Shawn/data_processed/"
sample_rate_target = 384000
npy_files = sorted(glob.glob(os.path.join(data_dir, "data_S70_*.npy")))[:10]

plt.figure(figsize=(12, 6))

print(f"Début de la vérification sur {len(npy_files)} fichiers...")

for i, npy_path in enumerate(npy_files):
	file_name = os.path.basename(npy_path).replace("data_ONECAT_", "").replace(".npy", "")
	
	# Chargement (on ne charge que le canal 5 pour économiser la RAM)
	# mmap_mode='r' permet de ne lire que ce dont on a besoin sans tout charger en RAM
	data = np.load(npy_path, mmap_mode='r')
	canal5 = np.array(data[:, 4]) 
	
	# Détection des PPS (Pics d'amplitude sur le canal 5 corrigé)
	# On cherche des pics espacés d'environ 1 seconde (384 000 samples)
	# peaks, _ = find_peaks(np.abs(canal5), height=0.7, distance=300000)
	seuil_verif = 10 * np.median(np.abs(canal5))
	peaks, _ = find_peaks(np.abs(canal5), height=seuil_verif, distance=300000)
	
	intervalles = np.diff(peaks)
	moyenne = np.mean(intervalles)
	std = np.std(intervalles)
	
	# Ajout au graphique
	plt.scatter(range(len(intervalles)), intervalles, marker='x', alpha=0.6, label=f"{file_name} (avg:{moyenne:.1f})")
	
	print(f"Fichier {i+1}/10 : {file_name} -> Moyenne: {moyenne:.2f} (std: {std:.2f})")

# Mise en forme du graphique
plt.axhline(y=sample_rate_target, color='black', linestyle='--', linewidth=1.5, label='Cible 384 000')
plt.title("Vérification de la Synchronisation sur 10 fichiers (Option B)")
plt.xlabel("Index de l'intervalle PPS")
plt.ylabel("Nombre d'échantillons")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()

# Sauvegarde
graph_path = "/scratch/Shawn/Verif_Globale_Resampling.png"
plt.savefig(graph_path)
plt.close()

print(f"\nTerminé ! Le graphique de synthèse est ici : {graph_path}")