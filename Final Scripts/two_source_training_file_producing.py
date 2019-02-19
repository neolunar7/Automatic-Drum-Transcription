# making two_source training files

import librosa
import tensorflow as tf
import numpy as np
import glob
import os

### DOUBLE: Clhat+Kick, Crash+Kick, Ride+Kick, Snare+Kick, Tom+Kick, Clhat+Snare, Crash+Snare, Ride+Snare, Tom+Snare (9)
### TRIPLE: Clhat+Snare+Kick, Crash+Snare+Kick, Ride+Snare+Kick, Tom+Snare+Kick (4)

dirname = 'D:\AMD\DrumSample_Final_10k'
other_classes = ['Clhat_Kick', 'Crash_Kick', 'Ride_Kick', 'Snare_Kick', 'Tom_Kick', 'Clhat_Snare', 'Crash_Snare', 'Ride_Snare', 'Tom_Snare', 'Clhat_Snare_Kick', 'Crash_Snare_Kick', 'Ride_Snare_Kick', 'Tom_Snare_Kick']


# Making two folders 'testing' and 'training'
for classe in other_classes:
	for folder in folders:
		path = os.path.join(dirname, classe, folder)
		try:
			os.mkdir(path)
		except OSError:
			print("Creation of the directory %s failed" % path)
		else:
			print("Succesfully created the directory %s" % path)



def training_file_lists_num(class_name):
	file_lists = glob.glob(os.path.join(dirname, class_name, 'Training', '*.wav'))
	return len(file_lists)

def randomfile_training_extraction(class_name, extraction_number):
	file_lists = glob.glob(os.path.join(dirname, class_name, 'Training', '*.wav'))
	idx = np.arange(0, len(file_lists))
	np.random.shuffle(idx)
	idx = idx[:extraction_number]
	temp = []
	for i in range(extraction_number):
		temp.append(file_lists[idx[i]])
	return temp

def twofile_training_comb(first_class_name, second_class_name):
	first_class_file_number = training_file_lists_num(first_class_name)
	second_class_file_number = training_file_lists_num(second_class_name)
	if first_class_file_number > 100:
		first_class_file_number = 100
	if second_class_file_number > 100:
		second_class_file_number = 100

	first_class_files = randomfile_training_extraction(first_class_name, first_class_file_number)
	second_class_files = randomfile_training_extraction(second_class_name, second_class_file_number)

	saving_dir = os.path.join(dirname, '%s_%s' % (first_class_name, second_class_name), 'Training')
	for i, first_class_file in enumerate(first_class_files):
		for j, second_class_file in enumerate(second_class_files):
			y1, sr = librosa.load(first_class_file, sr=44100)
			y2, sr = librosa.load(second_class_file, sr=44100)
			y3 = y1+y2
			nor_factor = 1/np.max(np.abs(y3))
			y3 = y3*nor_factor

			librosa.output.write_wav(os.path.join(saving_dir, '%s_%s_%s_%s.wav' % (first_class_name, second_class_name, i, j)), y3, sr)


twofile_training_comb('Clhat', 'Kick')
twofile_training_comb('Clhat', 'Snare')
twofile_training_comb('Crash', 'Kick')
twofile_training_comb('Crash', 'Snare')
twofile_training_comb('Ride', 'Kick')
twofile_training_comb('Ride', 'Snare')
twofile_training_comb('Snare', 'Kick')
twofile_training_comb('Tom', 'Kick')
twofile_training_comb('Tom', 'Snare')