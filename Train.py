# This function retries the inputs from the user using Argparse module for Python.
# To ensure that the process runs, default values will be used in case any inputs are missing.
# for useful info on argparse module, go to: https://www.golinuxcloud.com/python-argparse/
# import packages
import argparse

from torchvision import models
from model_functions import data_load, gpu_cpu, load_network, loss_opt, training

# def get_input_args():
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default = 'flowers', dest = 'data_dir', help= 'Path to the folder with data')
    #parser.add_argument('--save_dir', default = opzoeken in de boom help = 'Where to save the checkpoint')
parser.add_argument('--cat_name', default = 'cat_to_name.json', dest = 'cat_name', help = 'Name of the file mapping the category numbers to names')
parser.add_argument('--arch', default = 'vgg13', dest = 'arch', help = 'Chose the model to be used, options are vgg11, vgg13, vgg16 and vgg19')
parser.add_argument('--learning_rate', default = 0.01, type = float, dest= 'learning_rate', help = 'Set the learning rate of the model')
parser.add_argument('--epochs', default = 2, type = int, dest = 'epochs', help = 'Set the number of epochs to train the model')
parser.add_argument('--hidden_units', default = 512, type = int, dest = 'hidden_units', help = 'Set the number of hidden units in the classifier')
parser.add_argument('--gpu', type = bool, default = True, help = 'Use the GPU or cpu?')
parser.add_argument('--chkp_location', default = 'checkpoint.pth', help = 'Where will the checkpoint be saved?')
args = parser.parse_args()


# run the training by executing the functions in model_functions
# load the data and create the datasets
trainloader, validloader, testloader = data_load(args.data_dir)
# using the GPU?
device = gpu_cpu(args.gpu)
# 
model = load_network(args.hidden_units)
# function to generate the loss optimizer and criterion for the training
criterion, optimizer = loss_opt(model, args.learning_rate)
# returns the trained model
t_model = training(model, device, args.epochs, trainloader, optimizer, criterion, validloader, testloader)

# savechkp(t_model)