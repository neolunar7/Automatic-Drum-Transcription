# single source file duplication

import numpy as np
import librosa
import os
import glob
import shutil

dirname = 'D:\AMD\DrumSample_Final_10k'
classes = ['Clhat', 'Crash', 'Kick', 'Ride', 'Snare', 'Tom']

# making each training datasets to 10000 number of datasets
for classe in classes:
	file_lists = glob.glob(os.path.join(dirname, classe, 'Training', '*.wav'))
	file_num = len(file_lists)
	duplication_number = 10000 // file_num
	for i, file in enumerate(file_lists):
		for j in range(duplication_number):
			shutil.copy(file, os.path.join(dirname, classe, 'Training', '%s_times_duplicated_%s.wav' % (j,i)))