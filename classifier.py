#!usr/bin/env python3
#
# PORGAMMER: Martin Chileshe
# DATE CREATED: 8/8/2019
# REVISED DATE: 8/8/2019
# PURPOSE: This module contains all the initialisation code for the model
# Loading and saving the the trained model is also implemented in this module.
#
# modulem imports
#
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
import numpy as np
from utils import get_input_args, load_datasets
from collections import OrderedDict
from os import path

class Classifier:
    
    # define the model parameters
    args = None
    model = None
    optimizer = None
    criterion = None
    device = None
    epochs = None
    learning_rate = None
    arch = None
    hidden_units = None
    
    def __init__(self, args, device):
        self.args = args
        self.hidden_units = args.hidden_units
        self.arch = args.arch
        self.learning_rate = args.learning_rate
        self.device = device
        self.epochs = args.epochs
        self.criterion = nn.NLLLoss()
  
        self.set_model(models) # set the model
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate) # set the optimizer and the learning rate
        
    def set_model(self, models):
        
        """
        sets the model
        parameters:
            model_name - the name of model to initilize with. Given as --arch
            hidden_units - the number of hidden unist to use in the network
        return:
        None - function does not retunr anything
        """
    
         # define and set the model
        resnet18 = models.resnet18(pretrained=True)
        alexnet = models.alexnet(pretrained=True)
        vgg16 = models.vgg16(pretrained=True)
        densenet201 = models.densenet201(pretrained=True)

        models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16, 'densenet201': densenet201}
    
        # apply model
        self.model = models[self.arch]
    
        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False
        
        # set the classifier to match our datasets
        classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(1920, 1000)),
                ('ReLU', nn.ReLU()),
                ('fc2', nn.Linear(1000, self.hidden_units)),
                ('ReLU', nn.ReLU()),
                ('Dropout', nn.Dropout(0.7)),
                ('fc3', nn.Linear(self.hidden_units, 102)),
                ('output', nn.LogSoftmax(dim=1))
        ]))
        self.model.classifier = classifier
       
    def load_checkpoint(self, save_dir):
        """
        loads the saved model
        parameters:
            save_dir - the path to the directory where the model is saved
        return:
        None - function does not retunr anything
        """
        checkpoint = None
        if path.exists(save_dir):
            # load the checkpoint file
            if self.device == 'cpu':
                checkpoint = torch.load(save_dir,map_location=lambda storage, location: storage)
            else:
                checkpoint = torch.load(save_dir)
    
            # load the hyperparameter states form the checkpoint
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.class_to_idx = checkpoint['class_to_idx']
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.epochs = checkpoint['epochs']
            self.learning_rate = checkpoint['learning_rate']
            self.arch = checkpoint['arch']
        else:
            # do nothing if there is nothing to load
            pass
 
    def save_checkpoint(self, save_dir, train_datasets):
        """
        saves the trained model and other parameters to disk
        parameters:
            save_dir - the directory where the model should be saved
            train_datasets - the datasets that the model was trained on. 
                             this param is being used for getting the idx to class mappings
        """
        self.model.class_to_idx = train_datasets.class_to_idx
    
        # crete custom dictionary to save additional params
        checkpoint = {'epochs': self.epochs,
              'classifier': self.model.classifier,
              'learning_rate': self.learning_rate,
              'arch': self.arch,
              'class_to_idx': self.model.class_to_idx,
              'optimizer_state': self.optimizer.state_dict(),
             'state_dict': self.model.state_dict()}

        torch.save(checkpoint, save_dir)
