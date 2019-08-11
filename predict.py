#!usr/bin/env python3
#
# PROGRAMMER: Martin
# DATE CREATED: 06/08/2019
# REVISED DATE: 11/08/2019
# PURPOSE: To predict the class of flowers the given input belongs to. 
#          It uses the model trained from the train.py module
#
#       Example call:
#           python predict.py --input file.jpg
#
# Import python modules here

import torch
from PIL import Image
from utils import process_image, get_input_args
from classifier import Classifier
import json

def main():
    
    args = get_input_args() # get the commndline arguements
    cat_to_name = None
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # check if gpu arg is present and train with gpu else train with cpu
    device = None
    if args.gpu == 'gpu':
        if torch.cuda.is_available():
            device = torch.device("cuda") 
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        
    classifier = Classifier(args, device)
    classifier.load_checkpoint(args.save_dir)

    probs, classes = predict(args.input, classifier.model)
    labels = []
    labels += [cat_to_name.get(x) for x in classes]
    
    show_results(labels, probs, args.top_k) # show the results
  

def show_results(labels, probs, topk=1):
    
    probs = probs.squeeze()
    print("................................\n"
          "The top_{} predicted classes with their probabilities are:".format(topk))

    for idx, label in enumerate(labels):
        print("{}: {:.3f}..".format(label, probs[idx]))
    print("................................")

def predict(image_path, model, topk=5):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    arguments:
     image_path - path to the image being predicted
     model - the trained model to use for the prediction
     topk - the number of top probabilties that the input matched the most
    return:
     top_p - the list values for topk probalities of input matches
     classes - the clases to which the input has been matched
    '''
    
    # open and process the given image
    image = Image.open(image_path)
    img = process_image(image)
    
    # peform the model classification
    top_p = None
    top_class = None
    with torch.no_grad():
        model.eval()
        logs = model(img.unsqueeze_(0))
        ps = torch.exp(logs)
    
        top_p, top_class = ps.topk(topk, dim=1) #get the top probalities
    
    classes_idx = []
    # map the predicted classes to the class names using category_names
    top_class = top_class.squeeze()
    for i, x in model.class_to_idx.items():
        if x in top_class:
            classes_idx.append(i)
            
    return top_p, classes_idx

if __name__=="__main__":
    main()