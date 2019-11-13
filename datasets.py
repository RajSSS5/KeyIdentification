# A Wavenet For Source Separation - Francesc Lluis - 25.10.2018
# Datasets.py

import util
import os
import random
import numpy as np
import jams
import logging

jam = jams.JAMS()

class GiantStepsDataset():

    def __init__(self, mode):
        self.mode = mode
        self.activeSequences = 0
        self.jams_directory = r"dataset\annotations\jams"

        if(mode == "test"):
            self.directory = r"dataset\wav\test"
        else:
            self.directory = r"dataset\wav\train"

        
    

    def load_dataset_generator(self):
        print('Loading dataset...')
        print(self.directory)
        filenames = os.listdir(self.directory)
        random.shuffle(filenames)
        sequences = []
        # Load max of 100 songs

        

        for i in range(100):
            if filenames[i].endswith(".wav"): 
                sequence_path = os.path.join(self.directory, filenames[i])
                sequence = util.load_wav(sequence_path,16000)
                jams_name = filenames[i][:-4] + ".jams"
                jams_path = os.path.join(self.jams_directory, jams_name)
                jamsObj = jams.load(jams_path)
                if(jamsObj):
                    print("Reading: ", os.path.join(self.directory, filenames[i]))  
                    data_val = jamsObj.annotations[0].data[0].value
                    letter, key_type = data_val.split(":")
                    #print([sequence,letter,key_type])
                    sequences.append([sequence,letter,key_type])
            else:
                continue
        yield(np.array(sequences))



trainer = GiantStepsDataset("train")

print(next(trainer.load_dataset_generator())[0])