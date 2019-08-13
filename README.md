# AI Programming with Python Project by Udacity

Final project files for creating own image classifer for 102 different flowers.
This is part of the requirements for completing the Udacity's AI Programming with Python Nanodegree program. 
In this project, I build an image classifier using PyTorch, then convert it into a command line application.
The model can be saved to disk and loaded for usage at a later stage using the pytorch library.
The user has the option to train the model using the GPU or the CPU just by passing in the arguement of choice for example starting the program with:
```
    python train.py --gpu --epochs 10 --save_dir dir
```
Starts the training program using the gpu if it exists on the host. 10 iterations are made and dir is used as the save and load directory for the model. While starting the program with:
```
    python predict --input file.jpg --catergory_names category_names.json
```
Performs a prediction on the given input. It uses the trained model on disk to detemine what class this image belongs to.
The catergory names file conatins a mapping of the indexes to the catergory names.
