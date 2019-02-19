# three_source testing file producing

import librosa
import tensorflow as tf
import numpy as np
import glob
import os

### DOUBLE: Clhat+Kick, Crash+Kick, Ride+Kick, Snare+Kick, Tom+Kick, Clhat+Snare, Crash+Snare, Ride+Snare, Tom+Snare (9)
### TRIPLE: Clhat+Snare+Kick, Crash+Snare+Kick, Ride+Snare+Kick, Tom+Snare+Kick (4)

dirname = 'D:\AMD\DrumSample_Final_10k'

def testing_file_lists_num(class_name):
	file_lists = glob.glob(os.path.join(dirname, class_name, 'Testing', '*.wav'))
	return len(file_lists)

def randomfile_testing_extraction(class_name, extraction_number):
	file_lists = glob.glob(os.path.join(dirname, class_name, 'Testing', '*.wav'))
	idx = np.arange(0, len(file_lists))
	np.random.shuffle(idx)
	idx = idx[:extraction_number]
	temp = []
	for i in range(extraction_number):
		temp.append(file_lists[idx[i]])
	return temp

def threefile_training_comb(first_class_name, second_class_name, third_class_name):
	first_class_file_number = testing_file_lists_num(first_class_name)
	second_class_file_number = testing_file_lists_num(second_class_name)
	third_class_file_number = testing_file_lists_num(third_class_name)
	if first_class_file_number > 2:
		first_class_file_number = 2
	if second_class_file_number > 3:
		second_class_file_number = 3
	if third_class_file_number > 2:
		third_class_file_number = 2

	first_class_files = randomfile_testing_extraction(first_class_name, first_class_file_number)
	second_class_files = randomfile_testing_extraction(second_class_name, second_class_file_number)
	third_class_files = randomfile_testing_extraction(third_class_name, third_class_file_number)

	saving_dir = os.path.join(dirname, '%s_%s_%s' % (first_class_name, second_class_name, third_class_name), 'Testing')
	for i, first_class_file in enumerate(first_class_files):
		for j, second_class_file in enumerate(second_class_files):
			for k, third_class_file in enumerate(third_class_files):
				y1, sr = librosa.load(first_class_file, sr=44100)
				y2, sr = librosa.load(second_class_file, sr=44100)
				y3, sr = librosa.load(third_class_file, sr=44100)
				nor_factor = 1/np.max(np.abs(y4))
				y4 = y4*nor_factor

				librosa.output.write_wav(os.path.join(saving_dir, '%s_%s_%s_%s_%s_%s.wav' % (first_class_name, second_class_name, third_class_name, i, j, k)), y4, sr)
