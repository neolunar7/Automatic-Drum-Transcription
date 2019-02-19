#file duplication

import numpy as np
import librosa
import os
import glob
import shutil

dirname = 'D:\AMD\DrumSample_10k'
classes = ['Clhat', 'Crash', 'Kick', 'Ride', 'Snare', 'Tom']

# making each training datasets to 10000 number of datasets
for classe in classes:
	file_lists = glob.glob(os.path.join(dirname, classe, 'training', '*.wav'))
	file_num = len(file_lists)
	duplication_number = 10000 // file_num
	for i, file in enumerate(file_lists):
		for j in range(duplication_number):
			shutil.copy(file, os.path.join(dirname, classe, '%s_times_duplicated_%s.wav' % (j,i)))

# Next, use file_division to move the new files to the training folder


# Move all the leftover files to the 'training' folder
def move_to_training_folder(classe):
	file_lists = glob.glob(os.path.join(dirname, classe, '*.wav'))
	destination_dirname = os.path.join(dirname, classe, 'Training')

	for _, file_list in enumerate(file_lists):
		head, tail = os.path.split(file_list)
		shutil.move(file_list, os.path.join(destination_dirname, tail))

for classe in classes:
	move_to_training_folder(classe = classe)