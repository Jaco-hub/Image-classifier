# This function retries the inputs from the user using Argparse module for Python.
# To ensure that the process runs, default values will be used in case any inputs are missing.
# for useful info on argparse module, go to:https://www.golinuxcloud.com/python-argparse/
# import packages
import argparse

from torchvision import models
from model_functions import *

# def get_input_args():
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default = 'flowers', dest = 'data_dir', help= 'Path to the folder with data')
parser.add_argument('--save_dir', default = 'checkpoint.pth', help = 'Where to save the checkpoint')
parser.add_argument('--arch', default = 'vgg11', dest = 'arch', help = 'Choose the model to be used, options are "vgg11", "vgg16" and "vgg19"')
parser.add_argument('--learning_rate', default = 0.001, type = float, dest= 'learning_rate', help = 'Set the learning rate of the model')
parser.add_argument('--epochs', default = 1, type = int, dest = 'epochs', help = 'Set the number of epochs to train the model')
parser.add_argument('--hidden_units', default = 512, type = int, dest = 'hidden_units', help = 'Set the number of hidden units in the classifier')
parser.add_argument('--gpu', type = bool, default = True, help = 'Use the GPU or cpu?')
parser.add_argument('--chkp_location', default = 'checkpoint.pth', help = 'Where will the checkpoint be saved?')
args = parser.parse_args()



# run the training by executing the functions in model_functions
# load the data and create the datasets
train_data, trainloader, validloader, testloader = data_load(args.data_dir)
# using the GPU?
device = gpu_cpu(args.gpu)
# selecting and loading the model parameters
model = load_network(args.arch, args.hidden_units)
# function to generate the loss optimizer and criterion for the training
criterion, optimizer = loss_opt(model, args.learning_rate)
# returns the trained model
t_model = training(model, device, args.epochs, trainloader, optimizer, criterion, validloader, testloader)
# save the trained model in the checkpoint
save_chkp(t_model, train_data, args.epochs, args.learning_rate, optimizer, criterion, args.chkp_location)

