from scipy.signal import butter, filtfilt, correlate, find_peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def highpass_filter(sig, sr, cutoff=10000, order=5) :
	b, a = butter(order, cutoff/(sr/2), btype='high', analog=False)
	return filtfilt(b, a, sig)

datas = np.load("data_ONECAT_20200114_150201_247.npy")
audio = "ONECAT_20200114_150201_247"

for data in [datas] :
	sample_rate = 384000

	canal0 = data[:,0].astype(float)

	filt0 = highpass_filter(canal0, sample_rate)
	filt0 -= np.mean(filt0)
	filt0 /= np.max(np.abs(filt0))

	sig = np.abs(filt0)
	seuil = 0.3 * np.max(np.abs(sig))
	# energy = filt0**2
	# seuil = np.mean(energy) + 6 * np.std(energy)
	print(f"seuil choisi : {seuil}")

	indices = np.where(sig > seuil)[0]

	clics = []
	min_sep = int(0.002 * sample_rate)

	prev = -999999
	for idx in indices:
		if idx - prev > min_sep:
			clics.append(idx)
		prev = idx

	print(f"Nombre de clics détectés : {len(clics)}")

	plt.figure(figsize=(12,4))
	plt.plot(filt0)
	plt.axhline(y=seuil, color='green', linestyle='--', label=f'Seuil={seuil}')
	plt.axhline(y=-seuil, color='green', linestyle='--')
	# for c in clics:
	# 	plt.axvline(c, linestyle='--')
	plt.title("Détection des clics sur canal 0")
	plt.legend()
	plt.savefig(f"Clics C0 : {audio}")
	plt.close()

	window = int(0.001 * sample_rate)
	filtered = []
	for i in range(3) :
		canal = data[:, i+1].astype(float)
		canal -= np.mean(canal)
		canal /= np.max(np.abs(canal))

		filt = highpass_filter(canal, sample_rate)
		filt -= np.mean(filt)
		filt /= np.max(np.abs(filt))

		filtered.append(filt)

	ic = 0
	delta_ts = {ch+1 : [] for ch in range(3)}

	for clic in clics :
		print(f"Clic {ic} : {clic}")
		start = max(0, clic-100)
		end = min(len(filt0), clic+500)

		motif = filt0[start : end]
		motif -= np.mean(motif)
		motif /= np.max(np.abs(motif))

		plt.figure(figsize=(10,4))
		plt.plot(motif)
		plt.title(f"Clic {ic} - {audio}")
		plt.xlabel("Échantillons")
		plt.ylabel("Amplitude")
		plt.tight_layout()
		plt.savefig(f"Motif Clic {ic} : {audio}")
		plt.close()

		channels_to_check = [1, 2, 3]

		plt.figure(figsize=(14, 8))  # Figure plus grande pour 3 subplots
		plt.suptitle(f"Corrélations du clic {ic} : {audio}", fontsize=16)

		for i, filt in enumerate(filtered, start=1):
			start = max(0, clic-1000)
			end = min(len(filt0), clic+2000)
			sig = filt[start:end]

			corr = correlate(sig, motif, mode='valid')
			corr /= np.max(np.abs(corr))
			
			imax = np.argmax(corr)
			peak = imax + start
			delta_ts[i].append(peak - clic)

			plt.subplot(len(channels_to_check), 1, i)
			plt.plot(corr)
			plt.plot(imax, corr[imax], "rx")
			plt.title(f"Canal {i}")
			plt.xlabel("Échantillons")
			plt.ylabel("Corrélation")

		plt.tight_layout(rect=[0, 0, 1, 0.95])
		plt.savefig(f"Correlation clic {ic} : {audio}")
		plt.close()
	
		ic += 1

	print(f"DeltaT : {delta_ts}")