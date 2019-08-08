#!bin/usr/env python3
#
# PROGRAMMER: Martin Chileshe
# DATE CREATED: 8/8/2019
# REVISED DATE: 8/8/2019
# PURPOSE: contains utility functions that facilitate the program operations including
#          getting the input arguments from the terminal, loading the datasets, processing and showing the image
#
# module imports
#
import torch
import numpy as np
from torchvision import datasets, transforms, models
import argparse

def get_input_args():
    """
    Retrieves and parses the command line arguments provided at programme launch.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create the command line arguments
    parser.add_argument('--data_dir', type = str, default = 'flowers', help='path to the image datasets to train the model with')
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help='path to the folder to save the checkpoint')
    parser.add_argument('--arch', type = str, default = 'densenet201', help='the model architecture to use for the training')
    parser.add_argument('--learning_rate', type = int, default = 0.001, help='the learning rate to use when training the model')
    parser.add_argument('--hidden_units', type = int, default = 256, help='number of hidden units to use for training the model')
    parser.add_argument('--epochs', type = int, default = 5, help='number of epochs to train the model')
    parser.add_argument('--gpu', type = str, default = 'gpu', help='wether to use the gpu or not')
    parser.add_argument('--top_k', type = int, default = 1, help='the number of most likely classes to return')
    parser.add_argument('--input', type= str, default = 'flowers/valid/100/image_07895.jpg', help='path to the image to calssifier')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help='the file name to use for idx to name mapping')
 
    # return the arguments
    return parser.parse_args()

def load_datasets(image_path):
    """
    load the datasets from disk and applies transfaomations for prepare for model training
    Parameters:
        image_path - the path to the image datsets
    Returns:
        train_loader - the loader for the training datasets
        test_loader - the loader for testing datasets
        valid_loader - the loader for the validation datasets
        train_datasets - the training datasets without a loder
    """
    data_dir = image_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, shuffle=True, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    
    return train_loader, test_loader, valid_loader, train_datasets
                        
def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model
    Parameters:
        image - the image to process before prediction
    Returns:
        npt_image - a Numpy array of the image that was given
    '''
    
    # here, we resize to make the shoter side be 256
    width, height = image.size # get dimensions
    if height > width:
        h = (height/width)*256
        image.thumbnail((h,256))
    elif height < width:
        w = (width/height)*256
        image.thumbnail((256,w))
        
    # center crop
    left = (width-224)/2
    top = (height-224)/2
    right = (width+224)/2
    bottom = (height+224)/2
    
    croped_image = image.crop((left,top,right, bottom))
    
    
    # convert image to floats between 0,1
    np_image = np.array(croped_image)
    np_image.astype(float)
    np_image = np_image/255
    
    # normalise image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-mean)/std
    
    # transpose image
    npt_image = np_image.transpose()
    return torch.Tensor(npt_image)
