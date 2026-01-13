import cupy as cp
from cupyx.scipy.signal import butter, filtfilt, correlate as gpucorrelate
import numpy as np
import glob

def highpass_filter_gpu(sig, sr, cutoff=10000, order=5):
	from scipy.signal import butter as cpu_butter
	b, a = cpu_butter(order, cutoff/(sr/SR_HALF), btype='high', analog=False)
	return filtfilt(cp.asarray(b), cp.asarray(a), sig)

SR = 384000
SR_HALF = SR / 2
npy_files = sorted(glob.glob("data_ONECAT_*.npy"))[:10]

all_results = []

for npy_path in npy_files:
	print(f"Analyse des clics : {npy_path}")
	data = cp.load(npy_path)
	
	# Filtrage canal 0 pour détection
	filt0 = highpass_filter_gpu(data[:, 0], SR)
	sig_abs = cp.abs(filt0)
	indices = cp.where(sig_abs > 0.3)[0].get()
	
	# Tri des clics (2ms d'écart minimum)
	clics = []
	prev = -9999
	for idx in indices:
		if idx - prev > int(0.002 * SR):
			clics.append(idx)
		prev = idx

	# Calcul des TDOA (Delta T) sur GPU
	for clic in clics:
		start, end = max(0, clic-100), min(len(filt0), clic+500)
		motif = filt0[start:end]
		
		# Corrélation avec canaux 1, 2, 3
		for i in range(1, 4):
			sig_chan = data[clic-1000:clic+2000, i]
			corr = gpucorrelate(sig_chan, motif, mode='valid')
			delta_t = int(cp.argmax(cp.abs(corr)).get()) - 1000
			all_results.append({"file": npy_path, "clic_idx": clic, "channel": i, "dt": delta_t})

# Ici, all_results contient toutes les données pour ton clustering final