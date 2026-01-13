import cupy as cp
from cupyx.scipy.signal import butter, filtfilt, correlate as gpucorrelate
import numpy as np
import glob
import os

def highpass_filter_gpu(sig, sr, cutoff=10000, order=5):
	from scipy.signal import butter as cpu_butter
	b, a = cpu_butter(order, cutoff/(sr/2), btype='high', analog=False)
	return filtfilt(cp.asarray(b), cp.asarray(a), sig)

# Configuration
SR = 384000
data_dir = "/scratch/Shawn/data_processed/"
npy_files = sorted(glob.glob(os.path.join(data_dir, "data_ONECAT_*.npy")))

all_results = []

for npy_path in npy_files:
	print(f"Analyse des clics : {os.path.basename(npy_path)}")
	data = cp.load(npy_path)
	
	# Filtrage canal 0
	filt0 = highpass_filter_gpu(data[:, 0], SR)
	sig_abs = cp.abs(filt0)
	
	# --- SEUIL ADAPTATIF ---
	# On calcule la médiane du signal absolu (très robuste au bruit)
	# Un clic est typiquement 10 à 15 fois plus fort que le bruit médian
	bruit_median = cp.median(sig_abs)
	seuil_adaptatif = 12 * bruit_median # Ajuste le multiplicateur (12) si besoin
	
	indices = cp.where(sig_abs > seuil_adaptatif)[0].get()
	
	# Tri des clics
	clics = []
	prev = -9999
	for idx in indices:
		if idx - prev > int(0.002 * SR):
			clics.append(idx)
		prev = idx

	print(f"   -> {len(clics)} clics détectés")

	# Calcul des TDOA (Inter-corrélation)
	for clic in clics:
		if clic < 1000 or clic > len(filt0) - 2000: continue
		
		start, end = clic-100, clic+500
		motif = filt0[start:end]
		
		for i in range(1, 4):
			sig_chan = data[clic-1000:clic+2000, i]
			corr = gpucorrelate(sig_chan, motif, mode='valid')
			delta_t = int(cp.argmax(cp.abs(corr)).get()) - 1000
			all_results.append([npy_path, clic, i, delta_t])

# Sauvegarde des résultats finaux sur le scratch
results_arr = np.array(all_results)
np.save("/scratch/Shawn/tdoa_results.npy", results_arr)
print("Analyse terminée. Résultats sauvegardés dans /scratch/Shawn/tdoa_results.npy")