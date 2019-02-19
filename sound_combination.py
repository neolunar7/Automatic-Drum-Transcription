# Combining two sounds into one file
# Already normalized, so no need to normalize it again.

import librosa
import tensorflow as tf
import numpy as np
import glob
import os

### DOUBLE: Clhat+Kick, Crash+Kick, Ride+Kick, Snare+Kick, Tom+Kick, Clhat+Snare, Crash+Snare, Ride+Snare, Tom+Snare (9)
### TRIPLE: Clhat+Snare+Kick, Crash+Snare+Kick, Ride+Snare+Kick, Tom+Snare+Kick (4)

########################## Produced not enough data.... 30*30 is 1000 not 10000!!!!!!!!!!!
########### random 100 files needed, (Using ONLY originial files !!!) if there are less than 100 files, just use max number.
# Do it again.

# returns random files
def randomfile_extraction(dirname, class_name, extraction_number):
	file_lists = glob.glob(os.path.join(dirname, class_name, '*.wav'))
	idx = np.arange(0, len(file_lists))
	np.random.shuffle(idx)
	idx = idx[:extraction_number]
	temp = []
	for i in range(extraction_number):
		temp.append(file_lists[idx[i]])
	return temp

def twofile_combination(first_class_name, second_class_name, first_class_files, second_class_files):
	saving_dir = os.path.join(dirname, "%s_%s" %(first_class_name, second_class_name))
	for i, first_class_file in enumerate(first_class_files):
		for j, second_class_file in enumerate(second_class_files):
			y1, sr = librosa.load(first_class_file, sr=44100)
			y2, sr = librosa.load(second_class_file, sr=44100)
			y3 = y1 + y2
			nor_factor = 1/np.max(np.abs(y3))
			y3 = y3 * nor_factor

			librosa.output.write_wav(os.path.join(saving_dir, '%s_%s_%s_%s.wav' % (first_class_name, second_class_name, i, j)), y3, sr)

def threefile_combination(first_class_name, second_class_name, third_class_name, first_class_files, second_class_files, third_class_files):
	saving_dir = os.path.join(dirname, "%s_%s_%s" %(first_class_name, second_class_name, third_class_name))
	for i, first_class_file in enumerate(first_class_files):
		for j, second_class_file in enumerate(second_class_files):
			for k, third_class_file in enumerate(third_class_files):
				y1, sr = librosa.load(first_class_file, sr=44100)
				y2, sr = librosa.load(second_class_file, sr=44100)
				y3, sr = librosa.load(third_class_file, sr=44100)
				y4 = y1+y2+y3
				nor_factor = 1/np.max(np.abs(y4))
				y4 = y4 * nor_factor

				librosa.output.write_wav(os.path.join(saving_dir, '%s_%s_%s_%s_%s_%s.wav' % (first_class_name, second_class_name, third_class_name, i, j, k)), y4, sr)




### TWO FILE COMBINATION ###
dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 30)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 30)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 30)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 30)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 30)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 30)
twofile_combination('Clhat', 'Kick', Clhat_file_lists, Kick_file_lists)


dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 30)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 30)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 30)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 30)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 30)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 30)
twofile_combination('Clhat', 'Snare', Clhat_file_lists, Snare_file_lists)

dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 30)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 30)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 30)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 30)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 30)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 30)
twofile_combination('Crash', 'Kick', Crash_file_lists, Kick_file_lists)

dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 30)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 30)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 30)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 30)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 30)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 30)
twofile_combination('Crash', 'Snare', Crash_file_lists, Snare_file_lists)

dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 30)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 30)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 30)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 30)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 30)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 30)
twofile_combination('Ride', 'Kick', Ride_file_lists, Kick_file_lists)

dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 30)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 30)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 30)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 30)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 30)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 30)
twofile_combination('Ride', 'Snare', Ride_file_lists, Snare_file_lists)

dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 30)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 30)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 30)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 30)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 30)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 30)
twofile_combination('Snare', 'Kick', Snare_file_lists, Kick_file_lists)

dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 30)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 30)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 30)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 30)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 30)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 30)
twofile_combination('Tom', 'Kick', Tom_file_lists, Kick_file_lists)

dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 30)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 30)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 30)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 30)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 30)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 30)
twofile_combination('Tom', 'Snare', Tom_file_lists, Kick_file_lists)


### THREE FILE COMBINATION ###
dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 10)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 10)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 10)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 10)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 10)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 10)
threefile_combination('Clhat', 'Snare', 'Kick', Clhat_file_lists, Snare_file_lists, Kick_file_lists)

dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 10)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 10)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 10)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 10)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 10)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 10)
threefile_combination('Crash', 'Snare', 'Kick', Crash_file_lists, Snare_file_lists, Kick_file_lists)

dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 10)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 10)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 10)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 10)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 10)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 10)
threefile_combination('Ride', 'Snare', 'Kick', Ride_file_lists, Snare_file_lists, Kick_file_lists)

dirname = 'D:\AMD\DrumSample_10k'
Clhat_file_lists = randomfile_extraction(dirname, 'Clhat', 10)
Crash_file_lists = randomfile_extraction(dirname, 'Crash', 10)
Kick_file_lists = randomfile_extraction(dirname, 'Kick', 10)
Ride_file_lists = randomfile_extraction(dirname, 'Ride', 10)
Snare_file_lists = randomfile_extraction(dirname, 'Snare', 10)
Tom_file_lists = randomfile_extraction(dirname, 'Tom', 10)
threefile_combination('Tom', 'Snare', 'Kick', Tom_file_lists, Snare_file_lists, Kick_file_lists)

