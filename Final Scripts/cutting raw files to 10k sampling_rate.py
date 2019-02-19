# Cutting Files to 10000 sampling rate
# 0.4535 sec.

import numpy as np
import tensorflow as tf
import librosa
import glob
import os
import matplotlib.pyplot as plt


dirname = 'D:\AMD\DrumSample'
destination_dirname = 'D:\AMD\DrumSample_Final_10k'
classes = ['Clhat', 'Crash', 'Kick', 'Ride', 'Snare', 'Tom']

for classe in classes:
	file_lists_wav = glob.glob(os.path.join(dirname, classe, '*.wav'))
	file_lists_mp3 = glob.glob(os.path.join(dirname, classe, '*.mp3'))
	file_lists = file_lists_mp3 + file_lists_wav
	for filenumber, file_list in enumerate(file_lists):
		y, sr = librosa.load(file_list, sr=44100)
		# print("the size of the file is : ", np.size(y))
		onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
		# print("onset frames are: ", onset_frames)
		onset_times = librosa.frames_to_time(onset_frames, sr=sr)
		# print("onset times are: ", onset_times)
		onset_samples = librosa.frames_to_samples(onset_frames, hop_length=512, n_fft=1024)
		# print("onset samples are: ", onset_samples)
		first_onset = onset_samples[0]
		## is this the right algorithm to choose first onset?
		## or should I choose the biggest amplitude?
		# print("first onset sample is : ", first_onset)
		y = y[first_onset:]
		# print("y is : ", y)
		if np.size(y) < 10000:
			y = np.lib.pad(y, (0, 10000-np.size(y)), 'constant', constant_values = (0,0))
		else:
			y = y[:10000]
		
		nor_factor = 1/np.max(np.abs(y))
		y = y * nor_factor

		# mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
		# log_mel_spec = librosa.power_to_db(mel_spec, ref = np.max)
		# plt.figure(figsize = (10,4))
		# librosa.display.specshow(log_mel_spec, y_axis='mel', x_axis='time', fmax=8000)

		# np.save(os.path.join(destination_dirname, classe, '%s_10k_%s' % (classe, filenumber)), log_mel_spec)
		librosa.output.write_wav(os.path.join(destination_dirname, classe, '%s_10k_%s.wav' % (classe, filenumber)), y, sr)

####### All files are now in DrumSample_10k folder #########