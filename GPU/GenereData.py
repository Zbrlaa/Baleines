import cupy as cp
from cupyx.scipy.signal import correlate as gpucorrelate
from cupyx.scipy.signal import resample as gpuresample
from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
import glob
import os
import gc

# Configuration des chemins
input_dir = "/home/partage/M2IPA/meute/"
output_dir = "/scratch/Shawn/data_processed/"
os.makedirs(output_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(input_dir, "ONECAT_*.wav")))[:10]
sample_rate_nominal = 384000

for wav_path in files:
	audio_id = os.path.basename(wav_path).replace(".wav", "")
	output_path = os.path.join(output_dir, f"data_{audio_id}.npy")
	
	if os.path.exists(output_path):
		print(f"Saut de {audio_id} (déjà traité)")
		continue

	print(f"--- Traitement de {audio_id} ---")
	sr, data = wavfile.read(wav_path)
	
	# 1. Détection PPS sur GPU
	canal5_raw = cp.asarray(data[:, 4].astype(np.float32))
	seg = canal5_raw[:800000]
	seuil_pps = 0.7 * cp.max(cp.abs(seg))
	upi = int(cp.where(seg > seuil_pps)[0][0])
	atome = seg[upi-10:upi+10]
	atome -= cp.mean(atome)
	atome /= cp.max(cp.abs(atome))

	corr = gpucorrelate(canal5_raw, atome, mode='valid')
	indices_pps, _ = find_peaks(corr.get(), height=float(0.8 * cp.max(cp.abs(corr))))
	
	# Libération mémoire après PPS
	del canal5_raw, seg, corr
	
	# 2. Calcul du facteur
	mean_inter = np.mean(np.diff(indices_pps))
	factor = sample_rate_nominal / mean_inter
	n_target = int(np.round(len(data) * factor))
	
	# 3. Resampling CANAL PAR CANAL (Moins de VRAM utilisée)
	print(f"   Resampling vers {n_target} échantillons...")
	data_resampled = cp.zeros((n_target, 5), dtype=cp.float32)
	
	for i in range(5):
		print(f"      Canal {i} en cours...")
		chan_gpu = cp.asarray(data[:, i].astype(np.float32))
		
		# Option A : Resample FFT (Original)
		# data_resampled[:, i] = gpuresample(chan_gpu, n_target)
		
		# Option B : Si ça plante encore, remplace la ligne au dessus par :
		x_old = cp.arange(len(chan_gpu))
		x_new = cp.linspace(0, len(chan_gpu)-1, n_target)
		data_resampled[:, i] = cp.interp(x_new, x_old, chan_gpu)
		
		del chan_gpu # Nettoyage après chaque canal

	# 4. Normalisation finale
	data_resampled -= cp.mean(data_resampled, axis=0)
	data_resampled /= cp.max(cp.abs(data_resampled), axis=0)
	
	# 5. Sauvegarde et Nettoyage complet
	np.save(output_path, data_resampled.get())
	del data_resampled
	gc.collect() # Nettoyage CPU
	cp.get_default_memory_pool().free_all_blocks() # Nettoyage GPU
	
	print(f"Sauvegardé : {output_path}")

print("\nTraitement terminé avec succès.")