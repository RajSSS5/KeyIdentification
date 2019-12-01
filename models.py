# architecture modified from Wavenet For Source Separation - Francesc Lluis - 25.10.2018

import keras
import util
import os
import numpy as np
import layers
import math
import logging


#Singing Voice Separation Wavenet Model

class WavenetClassifier():

    def __init__(self, config, load_checkpoint=None,  print_model_summary=False, input_length=0):

        self.config = config
        self.optimizer = self.get_optimizer()
        self.metrics = self.get_metrics()
        self.epoch_num = 0
        self.checkpoints_path = ''
        self.samples_path = ''
        self.history_filename = ''
        self.input_length = input_length
        self.input_shape = (129, 2232,1)


        self.model = self.setup_model(load_checkpoint, print_model_summary)

    def setup_model(self, load_checkpoint=None, print_model_summary=False):

        self.checkpoints_path = os.path.join(self.config['training']['path'], 'checkpoints')
        self.samples_path = os.path.join(self.config['training']['path'], 'samples')
        self.history_filename = 'history_' + self.config['training']['path'][
                                             self.config['training']['path'].rindex('/') + 1:] + '.csv'

        model = self.build_model()

        if os.path.exists(self.checkpoints_path) and util.dir_contains_files(self.checkpoints_path):

            if load_checkpoint is not None:
                last_checkpoint_path = load_checkpoint
                self.epoch_num = 0
            else:
                checkpoints = os.listdir(self.checkpoints_path)
                checkpoints.sort(key=lambda x: os.stat(os.path.join(self.checkpoints_path, x)).st_mtime)
                last_checkpoint = checkpoints[-1]
                last_checkpoint_path = os.path.join(self.checkpoints_path, last_checkpoint)
                self.epoch_num = int(last_checkpoint[11:16])
            print('Loading model from epoch: %d' % self.epoch_num)
            model.load_weights(last_checkpoint_path)

        else:
            print('Building new model...')

            if not os.path.exists(self.config['training']['path']):
                os.mkdir(self.config['training']['path'])

            if not os.path.exists(self.checkpoints_path):
                os.mkdir(self.checkpoints_path)

            self.epoch_num = 0

        if not os.path.exists(self.samples_path):
            os.mkdir(self.samples_path)

        if print_model_summary:
            model.summary()

        model.compile(optimizer=self.optimizer,
                      loss='binary_crossentropy', metrics=self.metrics)
        self.config['model']['num_params'] = model.count_params()

        config_path = os.path.join(self.config['training']['path'], 'config.json')
        if not os.path.exists(config_path):
            util.pretty_json_dump(self.config, config_path)

        if print_model_summary:
            util.pretty_json_dump(self.config)
        return model

    def get_optimizer(self):

        return keras.optimizers.Adam(lr=self.config['optimizer']['lr'], decay=self.config['optimizer']['decay'],
                                     epsilon=self.config['optimizer']['epsilon'])

    def get_callbacks(self):

        return [
            # keras.callbacks.EarlyStopping(patience=self.config['training']['early_stopping_patience'], verbose=1,
            #                               monitor='loss'),
            keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoints_path,
                                                         'checkpoint.{epoch:05d}-{val_loss:.3f}.hdf5'), period=50),
            keras.callbacks.CSVLogger(os.path.join(self.config['training']['path'], self.history_filename), append=True)
        ]
    
    def get_metrics(self):

        return [
            keras.metrics.mean_absolute_error,
            self.valid_mean_absolute_error
        ]
    
    def valid_mean_absolute_error(self, y_true, y_pred):
        return keras.backend.mean(
            keras.backend.abs(y_true[:, 1:-2] - y_pred[:, 1:-2]))


    def fit_model(self, train_set_generator, num_steps_train, test_set_generator, num_steps_test, num_epochs):

        print('Fitting model with %d training num steps and %d test num steps...' % (num_steps_train, num_steps_test))

        self.model.fit_generator(train_set_generator,
                                 num_steps_train,
                                 epochs=num_epochs,
                                 validation_data=test_set_generator,
                                 validation_steps=num_steps_test,
                                 callbacks=self.get_callbacks(),
                                 verbose=1,
                                 initial_epoch=self.epoch_num,
                                 
                                 )

    def predict_on_batch(self, inputs):
        return self.model.predict_on_batch(inputs)
    
    def predict(self,input):
        return self.model.predict(input)

    def build_model(self):
        # modified from https://github.com/drscotthawley/audio-classifier-keras-cnn/blob/master/eval_network.py
        nb_filters = 32  # number of convolutional filters to use
        pool_size = (2, 2)  # size of pooling area for max pooling
        kernel_size = (3, 3)  # convolution kernel size
        nb_layers = 4
        sr = 16000

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(nb_filters, kernel_size,
                            border_mode='valid',input_shape=self.input_shape, name='data') )
        model.add(keras.layers.BatchNormalization(axis=1))
        model.add(keras.layers.Activation('relu'))

        for layer in range(nb_layers-1):
            model.add(keras.layers.Conv2D(nb_filters, kernel_size))
            model.add(keras.layers.BatchNormalization(axis=1))
            model.add(keras.layers.ELU(alpha=1.0))  
            model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
            model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(24))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1,activation='softmax', name='data_output'))
      
        return model