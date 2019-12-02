# A Wavenet For Source Separation - Francesc Lluis - 25.10.2018
# Extended in 2019 - Sukhraj Sekhon
# Main.py

import sys
import logging
import optparse
import json
import os
import models
import jams
import scipy
import numpy as np

from datasets import GiantStepsDataset
import util


encoded_key_vals = np.eye(24)
encoded_key_vals = [tuple(arr) for arr in encoded_key_vals]

key_vals = {
    encoded_key_vals[0]:'A:minor',
    encoded_key_vals[1]:'A:major',
    encoded_key_vals[2]:'Ab:minor',
    encoded_key_vals[3]:'Ab:major',
    encoded_key_vals[4]:'B:minor',
    encoded_key_vals[5]:'B:major',
    encoded_key_vals[6]:'Bb:minor',
    encoded_key_vals[7]:'Bb:major',
    encoded_key_vals[8]:'C:minor',
    encoded_key_vals[9]:'C:major',
    encoded_key_vals[10]:'D:minor',
    encoded_key_vals[11]:'D:major',
    encoded_key_vals[12]:'Db:minor',
    encoded_key_vals[13]:'Db:major',
    encoded_key_vals[14]:'E:minor',
    encoded_key_vals[15]:'E:major',
    encoded_key_vals[16]:'Eb:minor',
    encoded_key_vals[17]:'Eb:major',
    encoded_key_vals[18]:'F:major',
    encoded_key_vals[19]:'F:minor',
    encoded_key_vals[20]:'G:minor',
    encoded_key_vals[21]:'G:major',
    encoded_key_vals[22]:'Gb:minor',
    encoded_key_vals[23]:'Gb:major'
    
}
        
trainer = GiantStepsDataset("train")
tester = GiantStepsDataset("test")


def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        logging.error('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)
input_len = 500000

def training(config_letter, config_key_type, cla):

    #Instantiate model
    
    # model_letter = models.WavenetClassifier(config_letter, load_checkpoint=cla.load_checkpoint, input_length=input_length)
    model_key_type = models.WavenetClassifier(config_key_type, load_checkpoint=cla.load_checkpoint, input_length=input_len)
    
    num_steps_train_key_type = config_key_type['training']['num_steps_train']
    num_steps_test_key_type = config_key_type['training']['num_steps_test']

    batch_train_key_type = trainer.load_dataset_generator()
    batch_test_key_type = tester.load_dataset_generator()

    # batch = {'data_input': sequences}, {'letters': letters, 'key_types': batch_outputs_2}

    model_key_type.fit_model(batch_train_key_type, num_steps_train_key_type, batch_test_key_type, num_steps_test_key_type, config_key_type['training']['num_epochs'])
    # model_letter.fit_model(batch_train_letter, num_steps_train_letter, batch_test_letter, num_steps_test_letter, config_letter['training']['num_epochs'])

def get_command_line_arguments():
    parser = optparse.OptionParser()
    parser.set_defaults(mode='training')
    parser.set_defaults(load_checkpoint=None)
    parser.set_defaults(source=None)

    parser.add_option('--mode', dest='mode')
    parser.add_option('--load_checkpoint', dest='load_checkpoint')

    parser.add_option('--source', dest='source')

    (options, args) = parser.parse_args()

    return options



def inference(config_letter, config_key_type, cla):
    model_key_type = models.WavenetClassifier(config_key_type, load_checkpoint=cla.load_checkpoint, input_length=input_len)

    if cla.source.endswith('.wav'):
        filenames = [cla.source.rsplit('/', 1)[-1]]
        cla.source = cla.source.rsplit('/', 1)[0] + '/'

    else:
        if not cla.source.endswith('/'):
            cla.source += '/'
        filenames = [filename for filename in os.listdir(cla.source) if filename.endswith('.wav')]


    jams_directory = r"dataset\annotations\jams"

    for filename in filenames:
        inp = util.load_wav(cla.source + filename, 16000)
        inp = inp[:input_len]
        f, t, Sxx = scipy.signal.spectrogram(inp, 16000)
        Sxx = np.expand_dims(Sxx, axis=-1)
        Sxx = np.expand_dims(Sxx, axis=0)
        

        jams_name = filename[:-4] + ".jams"
        jams_path = os.path.join(jams_directory, jams_name)
        jamsObj = jams.load(jams_path)
        data_val = jamsObj.annotations[0].data[0].value

        print(model_key_type.predict(Sxx))
        result = key_vals[tuple(model_key_type.predict(Sxx)[0].astype(int))]
        print('File: ', filename, 'Prediction: ', result, 'Actual: ', data_val)


def set_system_settings():
    sys.setrecursionlimit(50000)
    logging.getLogger().setLevel(logging.INFO)

def get_valid_output_folder_path(outputs_folder_path):
    j = 1
    while True:
        output_folder_name = 'samples_%d' % j
        output_folder_path = os.path.join(outputs_folder_path, output_folder_name)
        if not os.path.isdir(output_folder_path):
            os.mkdir(output_folder_path)
            break
        j += 1
    return output_folder_path

def main():
    set_system_settings()
    cla = get_command_line_arguments()

    config_letter = load_config('letter_config.json')
    config_key_type = load_config('key_type_config.json')

    if cla.mode == 'training':
        training(config_letter, config_key_type, cla)
    elif cla.mode == 'inference':
        inference(config_letter, config_key_type, cla)


if __name__ == "__main__":
    main()

