import cupy as cp
from cupyx.scipy.signal import correlate as gpucorrelate
from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
import glob
import os
import gc

input_dir = "/home/partage/M2IPA/meute/"
output_dir = "/scratch/Shawn/data_processed/"
os.makedirs(output_dir, exist_ok=True)

# On cible les fichiers S70
files = sorted(glob.glob(os.path.join(input_dir, "S70_*.wav")))[:10]
sample_rate_nominal = 384000

for wav_path in files:
	audio_id = os.path.basename(wav_path).replace(".wav", "")
	output_path = os.path.join(output_dir, f"data_{audio_id}.npy")
	
	if os.path.exists(output_path):
		print(f"Saut de {audio_id} (déjà traité)")
		continue

	print(f"--- Traitement de {audio_id} ---")
	sr, data = wavfile.read(wav_path)
	
	# 1. Détection PPS robuste sur GPU
	canal5_raw = cp.asarray(data[:, 4].astype(np.float32))
	seg = canal5_raw[:800000]
	
	# Seuil initial basé sur le bruit ambiant (médiane)
	signal_abs = cp.abs(seg)
	seuil_pps = 15 * cp.median(signal_abs) # On cherche 15x le bruit de fond

	indices_seuil = cp.where(seg > seuil_pps)[0]
	
	if len(indices_seuil) == 0:
		print(f"   /!\\ Aucun PPS trouvé dans le segment initial de {audio_id}. Abandon.")
		continue

	upi = int(indices_seuil[0])
	atome = seg[upi-10:upi+10]
	atome -= cp.mean(atome)
	atome /= cp.max(cp.abs(atome))

	# Corrélation sur tout le canal 5
	corr = gpucorrelate(canal5_raw, atome, mode='valid')
	corr_abs = cp.abs(corr)
	
	# SEUIL DE CORRÉLATION ROBUSTE : On ne prend plus le MAX
	seuil_corr = 15 * cp.median(corr_abs) 
	
	indices_pps, _ = find_peaks(corr_abs.get(), height=float(seuil_corr), distance=300000)
	
	del canal5_raw, seg, corr, corr_abs
	
	# 2. Vérification avant calcul
	if len(indices_pps) < 2:
		print(f"   /!\\ Pas assez de PPS trouvés ({len(indices_pps)}) pour {audio_id}.")
		continue

	# 3. Calcul du facteur (Synchronisation)
	mean_inter = np.mean(np.diff(indices_pps))
	factor = sample_rate_nominal / mean_inter
	
	# Sécurité anti-aberration (le facteur doit être proche de 1.0)
	if not (0.9 < factor < 1.1):
		print(f"   /!\\ Facteur aberrant détecté ({factor:.4f}). Problème de PPS sur {audio_id}.")
		continue

	n_target = int(np.round(len(data) * factor))
	print(f"   Sync OK (facteur: {factor:.6f}) -> Resampling...")

	# 4. Resampling Canal par Canal (Option B : Interpolation)
	data_resampled = cp.zeros((n_target, 5), dtype=cp.float32)
	x_old = cp.arange(len(data))
	x_new = cp.linspace(0, len(data)-1, n_target)

	for i in range(5):
		print(f"      Canal {i}...")
		chan_gpu = cp.asarray(data[:, i].astype(np.float32))
		data_resampled[:, i] = cp.interp(x_new, x_old, chan_gpu)
		del chan_gpu

	# 5. Normalisation et Sauvegarde
	data_resampled -= cp.mean(data_resampled, axis=0)
	data_resampled /= cp.max(cp.abs(data_resampled), axis=0)
	
	np.save(output_path, data_resampled.get())
	
	# Nettoyage
	del data_resampled, x_old, x_new
	gc.collect()
	cp.get_default_memory_pool().free_all_blocks()
	
	print(f"Sauvegardé : {output_path}")

print("\nTraitement terminé.")