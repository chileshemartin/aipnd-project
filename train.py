import torch
from workspace_utils import active_session
import numpy as np
from utils import get_input_args, load_datasets
from classifier import Classifier

def start_training(train_loader, test_loader, valid_loader, model, epochs, device, criterion, optimizer, learning_rate, arch):
    model.to(device)
    steps = 0
    print_every = 5

    # start the trainig and keep an active session
    # print a statement to show the start of the training with the number of epochs
    print("............................\nstatinng the training with:\n",
          "epochs: {}\ndevice: {}\nlearning rate: {}\narchitecture: {}\n".
          format(epochs, device, learning_rate, arch),
          "............................")
    with active_session():
        for e in range(epochs):
            steps+=1
            running_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                log_d = model(inputs)
                loss = criterion(log_d, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
    
                # print the progress every 5 epochs
            if steps%print_every == 0:
                test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    model.eval()         
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log_d = model(inputs)
                        loss = criterion(log_d, labels)
                        test_loss += loss
                
                        ps = torch.exp(log_d)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
            
                    print('Epoch: {}/{}..'.format(e+1, epochs),
                            'Running loss: {:.3f}..'.format(running_loss/len(train_loader)),
                            'Test loss: {:.3f}..'.format(test_loss/len(test_loader)),
                            'accuracy: {:.2f}%'.format(accuracy.item()/len(test_loader)*100))
            
                    # set the model to traing mode
                    model.train()  
                
def main():
    
    args = get_input_args() # get the commndline arguements
    # check if gpu arg is present and train with gpu else train with cpu
    device = None
    if args.gpu == 'gpu':
        if torch.cuda.is_available():
            device = torch.device("cuda") 
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        
    classifier = Classifier(args, device) # initilize the train class with params   
    # attempt to load saved stated
    classifier.load_checkpoint(args.save_dir) # load a saved model
    train_loader, test_loader, valid_loader, train_datasets = load_datasets(args.data_dir) # prepare and load the datasets
    start_training(
        train_loader, 
        test_loader, 
        valid_loader, 
        classifier.model, 
        classifier.epochs, 
        classifier.device, 
        classifier.criterion, 
        classifier.optimizer,
        classifier.learning_rate,
        args.arch) # this starts the training process and prints the results
    classifier.save_checkpoint(args.save_dir, train_datasets) # save the trained state
                
if __name__ == '__main__':
    main()
