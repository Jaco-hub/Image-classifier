# All functions needed to train the model are here.

# Import the packages needed
from os import truncate
import numpy as np
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict

def data_load(data_dir):
    # Loads the data from the indicated location (data_dir) and then applies the transformation to the images.
    # Outputs are the dataloaders

    # Directories where the data is stored
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(25),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    # image_datasets = 
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    return trainloader, validloader, testloader

def load_mapping(cat_name):
    # Loads the mapping of the category numbers to category names
    # file name is taken from argparse
    with open('cat_name', 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def gpu_cpu(gpu):
    # Using the GPU (Cuda) or CPU to train
    if gpu == True:
        device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    return device


def load_network(hidden_units):
    # load a pre-trained netwerk selected
    model = models.vgg11(pretrained=True)

    # Freeze parameters so we don't backprop through them,
    for param in model.parameters():
        param.requires_grad = False

    # Define new classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units, bias=True)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier

    return model

def optim(model, learning_rate):
    #
    # Setting up the training configurations and definitions
    # # Loss with the Negative Log LikeLihood
    criterion = nn.NLLLoss()
    # learning_rate = 0.001
    # Adam is the chosen Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)   
    
    return criterion, optimizer


def training(model, device, epochs, trainloader, optimizer, criterion, validloader, testloader):

    # Put the model into the GPU
    model.to(device)

    # Define the training parameters
    steps = 0
    print_every = 50

    # Let's start training!
    for ii in range(epochs):
        train_loss = 0
        for images, labels in trainloader:
            steps += 1

            # Move image and label tensors to the GPU
            images, labels = images.to(device), labels.to(device)    
    
            #Zeroing the gradients
            optimizer.zero_grad()

            #Forward pass
            output = model.forward(images)
            loss = criterion(output, labels)
    
            # Backward propagation
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
        
            #validation
            if steps % print_every ==0:
                accuracy = 0
                valid_loss = 0
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for images, labels in validloader:
                        #move the data to the gpu
                        images, labels = images.to(device), labels.to(device)

                        #forward pass with the validation data      
                        output = model.forward(images)
                        valid_loss += criterion(output, labels).item()

                        #accuracy
                        ps = torch.exp(output)
                        top_prob, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += equality.type(torch.FloatTensor).mean()
        
                print("Epoch: {}/{}.. ".format(ii+1, epochs),
                      "Training Loss: {:.3f}.. ".format(train_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
        
                train_loss = 0
                # Make sure training is back on to allow drop out
                model.train()
    t_model = model
    return t_model


def savechkp(t_model):
    # Saves the checkpoint 
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'number_of_epochs': epochs,
                  'learning_rate': learning_rate,
                  'optimizer_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'criterion': criterion}

    torch.save(checkpoint, chkp_location)