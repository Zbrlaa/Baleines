from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import resample

#S70_20200114_150211_982
audios = ["ONECAT_20200114_150201_247"]

intervalles = []
intervalles_resampled = []

for audio in audios :
	sample_rate, data = wavfile.read("/home/partage/M2IPA/meute/" + audio + ".wav")

	print("Fréquence d’échantillonnage :", sample_rate)
	print("Type des données :", data.dtype)
	print("Forme des données :", data.shape)

	n_samples, n_channels = data.shape

	# plt.figure(figsize=(12, 8))
	# for i in range(n_channels):
	# 	plt.subplot(n_channels, 1, i + 1)
	# 	plt.plot(data[:, i])
	# 	plt.title(f"Canal {i+1}")
	# 	plt.ylabel("Amplitude")
	# 	if i == n_channels - 1:
	# 		plt.xlabel("Echantillons")
	# plt.tight_layout()
	# name = "Audio complet : " + audio
	# plt.savefig(name)
	# plt.close()

	seg = data[:800000]

	canal5 = seg[:, 4].astype(np.float32)

	signal_abs = np.abs(canal5)

	seuil = 0.7 * np.max(signal_abs)
	ups = np.where(canal5 > seuil)[0]
	downs = np.where(canal5 < -seuil)[0]
	upi = ups[0]
	
	if downs[0] < upi :
		downi = downs[1]
	else :
		downi = downs[0]

	pps = canal5[upi:downi+1]
	# plt.figure(figsize=(12, 4))
	# plt.plot(pps)
	# plt.title("Impulsion PPS détectée (canal 5)")
	# plt.xlabel("Échantillons")
	# plt.ylabel("Amplitude")
	# name = "PPS : " + audio
	# plt.savefig(name)
	# plt.close()

	up = canal5[upi-10:upi+10]
	atome = up - np.mean(up)
	atome /= np.max(np.abs(atome))
	# plt.figure(figsize=(12, 4))
	# plt.plot(atome)
	# plt.title("Impulsion PPS détectée (canal 5)")
	# plt.xlabel("Échantillons")
	# plt.ylabel("Amplitude")
	# name = "Atome : " + audio
	# plt.savefig(name)
	# plt.close()

	ecart = (upi-200)
	data = data[ecart:, :].astype(float)
	mean_data = np.mean(data, axis=0)
	max_data = np.max(np.abs(data), axis=0)
	data -= mean_data
	data /= max_data

	canal = data[:,4]

	corr = correlate(atome, canal, mode='valid')
	seuil_corr = 0.8 * np.max(np.abs(corr))
	indices_pps, _ = find_peaks(corr, height=seuil_corr)
	# print(f"Nb de pps avant resample : {len(indices_pps)}")

	# plt.figure(figsize=(12,6))
	# plt.plot(corr, label="Corrélation")
	# plt.scatter(indices_pps, corr[indices_pps], color='red', label="PPS détectés")
	# plt.xlabel("Échantillons")
	# plt.ylabel("Normalisée")
	# plt.title("Corrélation")
	# name = "All PPS : " + audio
	# plt.savefig(name)
	# plt.close()

	intervalle = np.diff(indices_pps)
	intervalles.append(intervalle)
	mean_inter = np.mean(intervalle)

	

	factor = sample_rate / mean_inter
	print("Facteur de resampling : ", factor)

	n_target = int(np.round((n_samples-ecart) * factor))

	data = resample(data, n_target, axis=0)
	mean_data = np.mean(data, axis=0)
	max_data = np.max(np.abs(data), axis=0)
	data -= mean_data
	data /= max_data
	np.save(f"data_{audio}.npy", data)
	# resampled = data[:, 4]

	# corr = correlate(atome, resampled, mode='valid')
	# corr /= np.max(np.abs(corr))
	# indices_pps_resampled, _ = find_peaks(corr, height=seuil_corr)
	# intervalle_resampled = np.diff(indices_pps_resampled)
	# # print(f"Nb de pps après resample : {len(indices_pps_resampled)}")

	# intervalles_resampled.append(intervalle_resampled)


# colors = ['r', 'b']
# if intervalles_resampled :
# 	plt.figure(figsize=(10,4))
# 	for i in range(len(intervalles_resampled)) :
# 		plt.scatter(range(1, len(intervalles[i])+1), intervalles[i], marker='.', color=colors[i], label=audios[i])
# 		plt.scatter(range(1, len(intervalles_resampled[i])+1), intervalles_resampled[i], marker='x', color=colors[i], label="resample "+audios[i])
# 	plt.axhline(y=sample_rate, color='green', linestyle='--', label=f'sample_rate={sample_rate}')
# 	plt.xlabel("Intervalle n°")
# 	plt.ylabel("Nombre d'échantillons")
# 	plt.title("Intervalle entre chaque PPS")
# 	plt.ylim(bottom=350000, top=400000)
# 	plt.legend()
# 	plt.savefig("Intervalles")
# 	plt.close()