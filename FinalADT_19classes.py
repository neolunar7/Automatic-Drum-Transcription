########### 19 features ADT ###########
### SINGLE: Clhat, Crash, Kick, Ride, Snare, Tom (6)
### DOUBLE: Clhat+Kick, Crash+Kick, Ride+Kick, Snare+Kick, Tom+Kick, Clhat+Snare, Crash+Snare, Ride+Snare, Tom+Snare (9)
### TRIPLE: Clhat+Snare+Kick, Crash+Snare+Kick, Ride+Snare+Kick, Tom+Snare+Kick (4)

# Currently not considering high-level techniques such as flam, drag, paradiddles, so on.

import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import librosa
import os
import librosa.display

dirname = 'D:\AMD\DrumSample_10k'
classes = ['Clhat', 'Crash', 'Kick', 'Ride', 'Snare', 'Tom', 'Clhat_Kick', 'Crash_Kick', 'Ride_Kick', 'Snare_Kick', 'Tom_Kick', 'Clhat_Snare', 'Crash_Snare', 'Ride_Snare', 'Tom_Snare', 'Clhat_Snare_Kick', 'Crash_Snare_Kick', 'Ride_Snare_Kick', 'Tom_Snare_Kick']
folders = ['training', 'testing']
dict_class = {'Clhat': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Crash': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Kick' : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Ride' : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Snare': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Tom'  : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Clhat_Kick' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Crash_Kick' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Ride_Kick'  : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Snare_Kick' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Tom_Kick'   : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              'Clhat_Snare': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              'Crash_Snare': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              'Ride_Snare' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              'Tom_Snare'  : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              'Clhat_Snare_Kick' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              'Crash_Snare_Kick' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              'Ride_Snare_Kick'  : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              'Tom_Snare_Kick'   : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
			} # as one-hot
sr = 44100

for classe in classes:
	for folder in folders:
		file_lists = glob.glob(os.path.join(dirname, classe, folder, '*.wav'))
		log_spectrograms = []
		labels = []
		for file_list in file_lists:
			y, sr = librosa.load(file_list, sr)
			if np.size(y) < 10000:
				y = np.lib.pad(y, (0, 10000-np.size(y)), 'constant', constant_values = (0,0))
			else:
				y = y[:10000]

			nor_factor = 1/np.max(np.abs(y))
			y = y * nor_factor

			mel_spec = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
			log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

			log_spectrograms.append(log_mel_spec)
			label = dict_class[classe]
			labels.append(label)

		np.save(os.path.join(dirname, '%s_%s_features.npy' % (classe, folder)), log_spectrograms)
		np.save(os.path.join(dirname, '%s_%s_labels.npy' % (classe, folder)), labels)


## Concatenting all the same type files ##
training_npy_file_lists = glob.glob(os.path.join(dirname, '*training_features.npy'))
training_features = np.ndarray((0,128,20))
for _, training_npy_file in enumerate(training_npy_file_lists):
    training_features = np.append(training_features,np.load(training_npy_file),axis=0)
print(np.shape(training_features))

training_label_npy_file_lists = glob.glob(os.path.join(dirname, '*training_labels.npy'))
training_labels = np.ndarray((0,19))
for _, training_label_npy_file in enumerate(training_label_npy_file_lists):
	training_labels = np.append(training_labels, np.load(training_label_npy_file), axis=0)
print(np.shape(training_labels))

testing_npy_file_lists = glob.glob(os.path.join(dirname, '*testing_features.npy'))
testing_features = np.ndarray((0,128,20))
for _, testing_npy_file in enumerate(testing_npy_file_lists):
	testing_features = np.append(testing_features, np.load(testing_npy_file), axis=0)
print(np.shape(testing_features))

testing_label_npy_file_lists = glob.glob(os.path.join(dirname, '*testing_labels.npy'))
testing_labels = np.ndarray((0,19))
for _, testing_label_npy_file in enumerate(testing_label_npy_file_lists):
	testing_labels = np.append(testing_labels, np.load(testing_label_npy_file), axis=0)
print(np.shape(testing_labels))

training_features = np.expand_dims(training_features, axis=3)
testing_features = np.expand_dims(testing_features, axis=3)


def next_batch(num, data, labels):
	idx = np.arange(0, len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[i] for i in idx]
	labels_shuffle = [labels[i] for i in idx]

	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def answer_matrix_into_source_answer(x):
	answer_matrix = x
	source_answer_matrix = []
	for i in range(len(x)):
		if answer_matrix[i] == 0:
			source_answer_matrix.append('Clhat')
		elif answer_matrix[i] == 1:
			source_answer_matrix.append('Crash')
		elif answer_matrix[i] == 2:
			source_answer_matrix.append('Kick')
		elif answer_matrix[i] == 3:
			source_answer_matrix.append('Ride')
		elif answer_matrix[i] == 4:
			source_answer_matrix.append('Snare')
		elif answer_matrix[i] == 5:
			source_answer_matrix.append('Tom')
		elif answer_matrix[i] == 6:
			source_answer_matrix.append('Clhat_Kick')
		elif answer_matrix[i] == 7:
			source_answer_matrix.append('Crash_Kick')
		elif answer_matrix[i] == 8:
			source_answer_matrix.append('Ride_Kick')
		elif answer_matrix[i] == 9:
			source_answer_matrix.append('Snare_Kick')
		elif answer_matrix[i] == 10:
			source_answer_matrix.append('Tom_Kick')
		elif answer_matrix[i] == 11:
			source_answer_matrix.append('Clhat_Snare')
		elif answer_matrix[i] == 12:
			source_answer_matrix.append('Crash_Snare')
		elif answer_matrix[i] == 13:
			source_answer_matrix.append('Ride_Snare')
		elif answer_matrix[i] == 14:
			source_answer_matrix.append('Tom_Snare')
		elif answer_matrix[i] == 15:
			source_answer_matrix.append('Clhat_Snare_Kick')
		elif answer_matrix[i] == 16:
			source_answer_matrix.append('Crash_Snare_Kick')
		elif answer_matrix[i] == 17:
			source_answer_matrix.append('Ride_Snare_Kick')
		else:
			source_answer_matrix.append('Tom_Snare_Kick')
	return source_answer_matrix


tf.reset_default_graph()

def CNN_classifier(x):
	input_sound_as_image = x

	##First Convolutional Layer##
	W_conv1 = tf.get_variable('c1w', shape = [5,5,1,64], initializer = tf.contrib.layers.xavier_initializer())
	b_conv1 = tf.get_variable('c1b', shape = [64], initializer = tf.contrib.layers.xavier_initializer())
	h_conv1 = tf.nn.relu(tf.nn.conv2d(input_sound_as_image, W_conv1, strides = [1,1,1,1], padding = 'SAME') + b_conv1)
	##First Pooling Layer##
	h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	##Second Convolutional Layer##
	W_conv2 = tf.get_variable('c2w', shape = [3,3,64,128], initializer = tf.contrib.layers.xavier_initializer())
	b_conv2 = tf.get_variable('c2b', shape = [128], initializer = tf.contrib.layers.xavier_initializer())
	h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides = [1,1,1,1], padding = 'SAME') + b_conv2)
	##Second Pooling Layer##
	h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)

    ##Fully Connected1##
	W_fc1 = tf.get_variable('f1w', shape = [32*5*128, 1024], initializer = tf.contrib.layers.xavier_initializer()) #With three max-pooling, the size becomes 16*5, and 128 is the previous filter num
	b_fc1 = tf.get_variable('f1b', shape = [1024], initializer = tf.contrib.layers.xavier_initializer())
	h_pool3_flat = tf.reshape(h_pool2_drop, [-1, 32*5*128])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ##Fully Connected2##
	W_fc2 = tf.get_variable('f2w', shape = [1024, 19], initializer = tf.contrib.layers.xavier_initializer())
	b_fc2 = tf.get_variable('f2b', shape = [19], initializer = tf.contrib.layers.xavier_initializer())
	logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_pred = tf.nn.softmax(logits)

	return y_pred, logits


x = tf.placeholder(tf.float32, shape = [None, 128, 20, 1])
y = tf.placeholder(tf.float32, shape = [None, 19])
keep_prob = tf.placeholder(tf.float32)

y_pred, logits = CNN_classifier(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# saver = tf.train.Saver()
my_prediction = tf.argmax(y_pred, 1) # When labels don't exist.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        batch = next_batch(1000, training_features, training_labels)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y: batch[1], keep_prob: 1.0})
            loss_print = loss.eval(feed_dict = {x: batch[0], y: batch[1], keep_prob: 1.0})
            print("Epoch: %d, Training Set Accuracy: %f, Loss: %f" % (i, train_accuracy, loss_print))
        sess.run(optimizer, feed_dict = {x: batch[0], y: batch[1], keep_prob:0.8})

    test_accuracy = 0
    for i in range(10):
       test_batch = next_batch(10, testing_features, testing_labels)
       test_accuracy = test_accuracy + accuracy.eval(feed_dict = {x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
       test_accuracy = test_accuracy / 10
    print("Test accuracy: %f" %test_accuracy)

    finalized_answer_matrix1 = sess.run(my_prediction, feed_dict = {x: bpm100_single_track1, keep_prob: 0.8})
    finalized_answer_matrix2 = sess.run(my_prediction, feed_dict = {x: bpm100_single_track2, keep_prob: 0.8})
    finalized_answer_matrix3 = sess.run(my_prediction, feed_dict = {x: bpm100_multi_track1, keep_prob: 0.8})

#######################################################################################

### CHECKING WITH REAL DRUM TRACK PRODUCED BY MUSESCORE2 ###
### Run this block before running session ###

track_dirname = 'D:\AMD\Produced MIDI'
tracklist = glob.glob(os.path.join(track_dirname, '*.wav'))
total_onset_times = []

for track in tracklist:
    y, sr = librosa.load(track)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length = 512, n_fft = 1024)
    onset_times = librosa.frames_to_time(onset_frames, sr = sr)
    temp = []
    for i in range(len(onset_samples)):
        temp.append(y[onset_samples[i]:onset_samples[i]+10000])

    log_spectrogram = []
    for j in range(len(temp)):
        normalization_factor = 1 / np.max(np.abs(temp[j]))
        temp[j] = temp[j] * normalization_factor
        mel_spectrogram = librosa.feature.melspectrogram(temp[j], sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        log_spectrogram.append(log_mel_spectrogram)
    
    np.save(os.path.join(dirname, '%s.npy' % track), log_spectrogram)
    total_onset_times.append(onset_times)

bpm100_single_track1 = np.load(os.path.join(track_dirname, 'bpm100 single track1.wav.npy'))
bpm100_single_track2 = np.load(os.path.join(track_dirname, 'bpm100 single track2.wav.npy'))
bpm100_multi_track1 = np.load(os.path.join(track_dirname, 'bpm100 multi track1.wav.npy'))

bpm100_single_track1 = np.expand_dims(bpm100_single_track1, axis = 3)
bpm100_single_track2 = np.expand_dims(bpm100_single_track2, axis = 3)
bpm100_multi_track1 = np.expand_dims(bpm100_multi_track1, axis = 3)

#######################################################################################

single_track1_sourcesheet = answer_matrix_into_source_answer(finalized_answer_matrix1)
single_track2_sourcesheet = answer_matrix_into_source_answer(finalized_answer_matrix2)
multi_track1_sourcesheet = answer_matrix_into_source_answer(finalized_answer_matrix3)

finalized_answer_track1 = []
finalized_answer_track2 = []
finalized_answer_track3 = []
for k in range(len(single_track1_sourcesheet)):
    finalized_answer_track1.append([single_track1_sourcesheet[k], total_onset_times[0][k]])
for k in range(len(single_track2_sourcesheet)):
    finalized_answer_track2.append([single_track2_sourcesheet[k], total_onset_times[1][k]])
for k in range(len(multi_track1_sourcesheet)):
	finalized_answer_track3.append([multi_track1_sourcesheet[k], total_onset_times[2][k]])

print(finalized_answer_track1)
print(finalized_answer_track2)
print(finalized_answer_track3)