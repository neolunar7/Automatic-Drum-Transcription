# dividing the files into training and testing set
import os
import tensorflow as tf
import glob
import numpy as np
import librosa
import shutil

dirname = 'D:\AMD\DrumSample_10k'
classes = ['Clhat', 'Crash', 'Kick', 'Ride', 'Snare', 'Tom', 'Clhat_Kick', 'Crash_Kick', 'Ride_Kick', 'Snare_Kick', 'Tom_Kick', 'Clhat_Snare', 'Crash_Snare', 'Ride_Snare', 'Tom_Snare', 'Clhat_Snare_Kick', 'Crash_Snare_Kick', 'Ride_Snare_Kick', 'Tom_Snare_Kick']
folders = ['training', 'testing']

# Making two folders 'testing' and 'training'
for classe in classes:
	for folder in folders:
		path = os.path.join(dirname, classe, folder)
		try:
			os.mkdir(path)
		except OSError:
			print("Creation of the directory %s failed" % path)
		else:
			print("Succesfully created the directory %s" % path)

# Randomly move 10% of the total files to the 'testing' folder
## BEWARE NOT TO IMPLEMENT THIS FUNCTION TWICE !!!!!!!!!!!!!!!!!!!!!! ##
def move_to_testing_folder(classe):
	file_lists = glob.glob(os.path.join(dirname, classe, '*.wav'))
	destination_dirname = os.path.join(dirname, classe, 'Testing')
	set_num = len(file_lists) // 10
	idx = np.arange(0, len(file_lists))
	np.random.shuffle(idx)
	idx = idx[:set_num]

	for _, random_num in enumerate(idx):
		head, tail = os.path.split(file_lists[random_num])
		shutil.move(file_lists[random_num], os.path.join(destination_dirname, tail))

# Move all the leftover files to the 'training' folder
def move_to_training_folder(classe):
	file_lists = glob.glob(os.path.join(dirname, classe, '*.wav'))
	destination_dirname = os.path.join(dirname, classe, 'Training')

	for _, file_list in enumerate(file_lists):
		head, tail = os.path.split(file_list)
		shutil.move(file_list, os.path.join(destination_dirname, tail))

for classe in classes:
	move_to_testing_folder(classe = classe)

for classe in classes:
	move_to_training_folder(classe = classe)