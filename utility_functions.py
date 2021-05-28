# Utility functions to run the predict.py file
# Import packages
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
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
import time
from PIL import Image
import seaborn as sns

# Defining the utility functions to run predict.py

def load_mapping(cat_name):
    # Loads the mapping of the category numbers to category names
    # file name is taken from argparse
    with open(cat_name, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(chkp_location):
    model = torch.load(chkp_location)    
    #model.classifier = checkpoint['classifier']
    class_to_idx = model['class_to_idx']
    #model.epochs = checkpoint['number_of_epochs']
    #learning_rate = model['learning_rate']
    #optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate) 
    #optimizer.load_state_dict(model['optimizer_dict'])
    #model.load_state_dict(model['state_dict'])
    
    return model, class_to_idx

def gpu_cpu(gpu):
    # Using the GPU (Cuda) or CPU to train
    if gpu == True:
        device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    return device

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)
    # Using the same transformation as before for the validation set,
    # so no need to apply ndarry.transpose. Also normalizes the color channes, so also not needed.
    transform = transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                             [0.229, 0.224, 0.225])])
    
    image = transform(pil_image) 
    
    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# imshow(process_image(image), None, 'surprise flower')

def predict(model, image_path, device, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()    
    # Implement the code to predict the class from an image file
    image = process_image(image_path).type(torch.FloatTensor).unsqueeze(0).to(device)
    
    # load_checkpoint(chkp_location)
    # is the eval mode needed? should be, to avoid drop-out....?

    
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    ps_topk_class = []
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        
        #take top 5 probability and index from output
        ps_topk = ps.cpu().topk(top_k)[0].numpy()[0]
        ps_topk_idx = ps.cpu().topk(top_k)[1].numpy()[0]
        
        # Loop through class_to_idx dict to reverse key, values in idx_to_class dict
        for key, value in model.class_to_idx.items():
            model.idx_to_class[value] = key
        # Loop through index to retrieve class from idx_to_class dict
        for item in ps_topk_idx:
            ps_topk_class.append(model.idx_to_class[item])
    
    results = np.column_stack((ps_topk_class, ps_topk))
    #print(ps_topk)
    #print(ps_topk_class)
    #return ps_topk, ps_topk_class
    return results

# Display an image along with the top 5 classes
def plot_solution(image, ps_topk, ps_topk_class, model):
    
    #number_classes = 5
    
    tensor_image = process_image(image)
    #ps_topk, ps_topk_class = predict(image, model, topk=5)
    ps_topk, ps_topk_class
    labels = [cat_to_name[c1] for c1 in ps_topk_class]
    np_image = tensor_image.numpy().squeeze()
    img_tensor = torch.from_numpy(np.asarray(np_image))
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
   
    flower_num = image.split('/')[2]
    title = cat_to_name[flower_num]
    ax.set_title(title)
    imshow(img_tensor, ax);
    class_idx=model.class_to_idx
    plt.subplot(2,1,2)
  #  sns.barplot(x=probs*100, y=labels, color=sns.color_palette()[0]);
    sns.barplot(x=ps_topk, y=labels, color=sns.color_palette()[0]);
    plt.show()    
    plt.tight_layout