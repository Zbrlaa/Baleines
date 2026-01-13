import cupy as cp
from cupyx.scipy.signal import correlate as gpucorrelate
from cupyx.scipy.signal import resample as gpuresample
from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
import glob
import os

# Configuration
input_dir = "/home/partage/M2IPA/meute/"
files = sorted(glob.glob(os.path.join(input_dir, "ONECAT_*.wav")))[:10] # Les 10 premiers
sample_rate_nominal = 384000

for wav_path in files:
	audio_id = os.path.basename(wav_path).replace(".wav", "")
	print(f"--- Traitement de {audio_id} ---")
	
	sr, data = wavfile.read(wav_path)
	# On prend les 800k premiers échantillons pour trouver l'atome PPS
	seg = cp.asarray(data[:800000, 4].astype(np.float32))
	
	# Détection du motif PPS (Atome)
	seuil = 0.7 * cp.max(cp.abs(seg))
	upi = int(cp.where(seg > seuil)[0][0])
	atome = seg[upi-10:upi+10]
	atome -= cp.mean(atome)
	atome /= cp.max(cp.abs(atome))

	# Corrélation sur tout le canal 5 (GPU)
	canal5 = cp.asarray(data[:, 4].astype(np.float32))
	corr = gpucorrelate(canal5, atome, mode='valid')
	indices_pps, _ = find_peaks(corr.get(), height=float(0.8 * cp.max(cp.abs(corr))))
	
	# Calcul du facteur de correction
	mean_inter = np.mean(np.diff(indices_pps))
	factor = sample_rate_nominal / mean_inter
	n_target = int(np.round(len(data) * factor))
	
	# Resampling de tous les canaux (GPU)
	data_gpu = cp.asarray(data.astype(np.float32))
	data_resampled = gpuresample(data_gpu, n_target, axis=0)
	
	# Normalisation et sauvegarde
	data_resampled -= cp.mean(data_resampled, axis=0)
	data_resampled /= cp.max(cp.abs(data_resampled), axis=0)
	
	np.save(f"data_{audio_id}.npy", data_resampled.get())
	print(f"Sauvegardé : data_{audio_id}.npy")