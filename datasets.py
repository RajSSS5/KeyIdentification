# A Wavenet For Source Separation - Francesc Lluis - 25.10.2018
# Datasets.py

import util
import os
import random
import numpy as np
import jams
import logging
import scipy

jam = jams.JAMS()
encoded_key_vals = np.eye(24)
encoded_key_vals = [tuple(arr) for arr in encoded_key_vals]

class GiantStepsDataset():

    def __init__(self, mode):
        self.mode = mode
        self.activeSequences = 0
        self.jams_directory = r"dataset\annotations\jams"
        self.key_vals = {
            'A:minor': encoded_key_vals[0],
            'A:major':encoded_key_vals[1],
            'Ab:minor':encoded_key_vals[2],
            'Ab:major':encoded_key_vals[3],
            'B:minor':encoded_key_vals[4],
            'B:major':encoded_key_vals[5],
            'Bb:minor':encoded_key_vals[6],
            'Bb:major':encoded_key_vals[7],
            'C:minor':encoded_key_vals[8],
            'C:major':encoded_key_vals[9],
            'D:minor':encoded_key_vals[10],
            'D:major':encoded_key_vals[11],
            'Db:minor':encoded_key_vals[12],
            'Db:major':encoded_key_vals[13],
            'E:minor':encoded_key_vals[14],
            'E:major':encoded_key_vals[15],
            'Eb:minor':encoded_key_vals[16],
            'Eb:major':encoded_key_vals[17],
            'F:major':encoded_key_vals[18],
            'F:minor':encoded_key_vals[19],
            'G:minor':encoded_key_vals[20],
            'G:major':encoded_key_vals[21],
            'Gb:minor':encoded_key_vals[22],
            'Gb:major':encoded_key_vals[23],
            
        }

        if(mode == "test"):
            self.directory = r"dataset\wav\test"
        else:
            self.directory = r"dataset\wav\train"

        
    

    def load_dataset_generator(self):
        print('Loading dataset...')
        print(self.directory)
        
        
        
        input_length = 500000
        
        while True:
            filenames = os.listdir(self.directory)
            random.shuffle(filenames)
            sequences = []
            letters = []
            key_types = []

            for i in range(3):
                if filenames[i].endswith(".wav"): 
                    sequence_path = os.path.join(self.directory, filenames[i])
                    sequence = util.load_wav(sequence_path,16000)
                    jams_name = filenames[i][:-4] + ".jams"
                    jams_path = os.path.join(self.jams_directory, jams_name)
                    jamsObj = jams.load(jams_path)
                    if(jamsObj):

                        data_val = jamsObj.annotations[0].data[0].value

                        sequence = sequence[:input_length]
                        f, t, Sxx = scipy.signal.spectrogram(sequence, 16000)
                        sequence = np.array(Sxx)
                        sequences.append(sequence)
                        key_types.append(self.key_vals[data_val])
                else:
                    continue
            
            sequences = np.array(sequences)
            sequences = np.expand_dims(sequences, axis=-1)
            key_types = np.array(key_types)
            batch = {'data_input': sequences}, {'data_output': key_types}

            yield(batch)

