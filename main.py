from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import resample


audios = ["ONECAT_20200114_150201_247", "S70_20200114_150211_982"]

intervalles = []
intervalles_resampled = []

for audio in audios :
	sample_rate, data = wavfile.read("/home/partage/M2IPA/meute/" + audio + ".wav")

	print("Fréquence d’échantillonnage :", sample_rate)
	print("Type des données :", data.dtype)
	print("Forme des données :", data.shape)

	n_samples, n_channels = data.shape

	plt.figure(figsize=(12, 8))
	for i in range(n_channels):
		plt.subplot(n_channels, 1, i + 1)
		plt.plot(data[:, i])
		plt.title(f"Canal {i+1}")
		plt.ylabel("Amplitude")
		if i == n_channels - 1:
			plt.xlabel("Echantillons")
	plt.tight_layout()
	name = "Audio complet : " + audio
	plt.savefig(name)

	seg = data[:800000]

	canal5 = seg[:, 4].astype(np.float32)

	signal_abs = np.abs(canal5)

	seuil = 0.7 * np.max(signal_abs)
	ups = np.where(seg > seuil)[0]
	downs = np.where(seg < -seuil)[0]
	upi = ups[0]
	if downs[0] < upi :
		downi = downs[1]
	else :
		downi = downs[0]
	pps = canal5[upi:downi+1]
	plt.figure(figsize=(12, 4))
	plt.plot(pps)
	plt.title("Impulsion PPS détectée (canal 5)")
	plt.xlabel("Échantillons")
	plt.ylabel("Amplitude")
	name = "PPS : " + audio
	plt.savefig(name)

	up = canal5[upi-10:upi+10]
	plt.figure(figsize=(12, 4))
	plt.plot(up)
	plt.title("Impulsion PPS détectée (canal 5)")
	plt.xlabel("Échantillons")
	plt.ylabel("Amplitude")
	name = "Up : " + audio
	plt.savefig(name)

	atome = up - np.mean(up)
	atome /= np.max(np.abs(atome))

	canal = data[:,4].astype(float)  # 6ème canal si data[5]
	canal -= np.mean(canal)
	canal /= np.max(np.abs(canal))


	corr = correlate(atome, canal, mode='valid')
	corr /= np.max(np.abs(corr))
	seuil_corr = 0.8
	indices_pps, _ = find_peaks(corr, height=seuil_corr)


	# --- Affichage ---
	plt.figure(figsize=(12,6))
	plt.plot(corr, label="Corrélation")
	plt.scatter(indices_pps, corr[indices_pps], color='red', label="PPS détectés")
	plt.xlabel("Échantillons")
	plt.ylabel("Normalisée")
	plt.title("Corrélation")
	name = "All PPS : " + audio
	plt.savefig(name)

	intervalle = np.diff(indices_pps)
	intervalles.append(intervalle)

	mean_inter = np.mean(intervalle)

	factor = sample_rate / mean_inter
	print("Facteur de resampling (simple):", factor)

	n_target = int(np.round(n_samples * factor))
	resampled = resample(canal, n_target)

	corr = correlate(atome, resampled, mode='valid')
	corr /= np.max(np.abs(corr))
	indices_pps_resampled, _ = find_peaks(corr, height=seuil_corr)

	intervalles_resampled.append(np.diff(indices_pps_resampled))


plt.figure(figsize=(10,4))
plt.scatter(range(1, len(intervalles[0])+1), intervalles[0], marker='.', color='r', label=audios[0])
plt.scatter(range(1, len(intervalles[1])+1), intervalles[1], marker='.', color='b', label=audios[1])
plt.scatter(range(1, len(intervalles_resampled[0])+1), intervalles_resampled[0], marker='x', color='r', label="resample "+audios[0])
plt.scatter(range(1, len(intervalles_resampled[1])+1), intervalles_resampled[1], marker='x', color='b', label="resample "+audios[0])


plt.axhline(y=sample_rate, color='green', linestyle='--', label=f'sample_rate={sample_rate}')
plt.xlabel("Intervalle n°")
plt.ylabel("Nombre d'échantillons")
plt.title("Intervalle entre chaque PPS")
plt.ylim(bottom=350000, top=400000)
plt.legend()
plt.savefig("Intervalles")