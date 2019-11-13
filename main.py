# A Wavenet For Source Separation - Francesc Lluis - 25.10.2018
# Extended in 2019 - Sukhraj Sekhon
# Main.py

import sys
import logging
import optparse
import json
import os

from datasets import GiantStepsDataset
import util


trainer = GiantStepsDataset("train")
tester = GiantStepsDataset("test")



def training(config, cla):

    #Instantiate model
    # model = 
    
    num_steps_test =  500
    num_steps_train =  2000
    num_steps_train = 100
    num_steps_val = 100
    num_epochs = 100


    #model.fit_model(trainer.load_dataset(), num_steps_train, tester, num_steps_test, num_epochs)




def inference():
    return

def main():
    return


if __name__ == "__main__":
    main()
