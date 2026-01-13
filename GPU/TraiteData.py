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
# On cible bien les fichiers S70
npy_files = sorted(glob.glob(os.path.join(data_dir, "data_S70_*.npy")))

# --- PARAMÈTRES "HAUTE SENSIBILITÉ" POUR S70 ---
SEUIL_MULTIPLICATEUR = 7   # On divise par deux pour capter les clics faibles
SCORE_CORR_MIN = 0.30       # Un peu plus tolérant sur la forme du clic
# ----------------------------------------------

all_results = []

for npy_path in npy_files:
	print(f"Analyse sensible de {os.path.basename(npy_path)}...")
	data_raw = cp.load(npy_path)
	
	# Filtrage de tous les canaux
	data_filt = cp.zeros_like(data_raw)
	for ch in range(4):
		data_filt[:, ch] = highpass_filter_gpu(data_raw[:, ch], SR)
	
	# Détection sur le canal 0
	sig_abs0 = cp.abs(data_filt[:, 0])
	bruit_median = cp.median(sig_abs0)
	seuil_adaptatif = SEUIL_MULTIPLICATEUR * bruit_median
	
	indices = cp.where(sig_abs0 > seuil_adaptatif)[0].get()
	
	# Tri des clics (2ms)
	clics_candidats = []
	prev = -9999
	for idx in indices:
		if idx - prev > int(0.002 * SR):
			clics_candidats.append(idx)
		prev = idx

	clics_valides = 0
	scores_clics = []

	for clic in clics_candidats:
		if clic < 1000 or clic > len(data_filt) - 2000: continue
		
		motif = data_filt[clic-100:clic+500, 0]
		motif_norm = motif / cp.sqrt(cp.sum(motif**2))
		
		clic_data_tmp = []
		qualite_clic = True
		scores_canaux = []
		
		for i in range(1, 4):
			sig_chan = data_filt[clic-1000:clic+2000, i]
			sig_chan_norm = sig_chan / cp.sqrt(cp.sum(sig_chan**2))
			
			corr = gpucorrelate(sig_chan_norm, motif_norm, mode='valid')
			score = float(cp.max(cp.abs(corr)).get())
			scores_canaux.append(score)
			
			if score < SCORE_CORR_MIN:
				qualite_clic = False
				break
			
			delta_t = int(cp.argmax(cp.abs(corr)).get()) - 1000
			clic_data_tmp.append([clic, i, delta_t])
		
		if qualite_clic:
			for item in clic_data_tmp:
				all_results.append([npy_path, item[0], item[1], item[2]])
			clics_valides += 1
			scores_clics.append(np.mean(scores_canaux))

	avg_score = np.mean(scores_clics) if scores_clics else 0
	print(f"   -> {clics_valides} clics retenus / {len(clics_candidats)} candidats (Score moy: {avg_score:.2f})")

# Sauvegarde spécifique pour S70
results_arr = np.array(all_results)
np.save("/scratch/Shawn/tdoa_results_s70.npy", results_arr)
print(f"\nTerminé. Résultats S70 sauvegardés dans tdoa_results_s70.npy")