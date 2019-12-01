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


class GiantStepsDataset():

    def __init__(self, mode):
        self.mode = mode
        self.activeSequences = 0
        self.jams_directory = r"dataset\annotations\jams"
        self.key_vals = {
            'A:minor': 0,
            'A:major':1,
            'Ab:minor':2,
            'Ab:major':3,
            'B:minor':4,
            'B:major':5,
            'Bb:minor':6,
            'Bb:major':7,
            'C:minor':8,
            'C:major':9,
            'D:minor':10,
            'D:major':11,
            'Db:minor':12,
            'Db:major':13,
            'E:minor':14,
            'E:major':15,
            'Eb:minor':16,
            'Eb:major':17,
            'F:major':18,
            'F:minor':19,
            'G:minor':20,
            'G:major':21,
            'Gb:minor':22,
            'Gb:major':23,
            
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
                        #print([sequence,letter,key_type])
                        # sequence = np.array(sequence)
                        # print("Reading: ", os.path.join(self.directory, filenames[i]))  
                        data_val = jamsObj.annotations[0].data[0].value

                        sequence = sequence[:input_length]
                        f, t, Sxx = scipy.signal.spectrogram(sequence, 16000)
                        sequence = np.array(Sxx)
                        
                        # sequence.reshape(129,2232,1)
                        sequences.append(sequence)

                        key_types.append(self.key_vals[data_val])
                else:
                    continue
            
            sequences = np.array(sequences)
            sequences = np.expand_dims(sequences, axis=-1)
            # sequences.reshape(50,129,2232,1)

            key_types = np.array(key_types)
            # print(sequences.shape)

            # print(sequences[0].shape)

            batch = {'data_input': sequences}, {'data_output': key_types}

            
            yield(batch)

                                # print(sequence.shape)
                    # if(return_type == 'letter'):
                    #     sample = {'data_input': sequence, 'key_types': key_type}, {'letters': letter}
                    # elif(return_type == 'key_type'):
                    #     sample = {'data_input': sequence}, {'key_types': key_type} 
                    
                    # yield(sample)

