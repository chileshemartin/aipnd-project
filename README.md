# AI Programming with Python Project by Udacity

Final project files for creating own image classifer for 102 different flowers.
This is part of the requirements for completing the Udacity's AI Programming with Python Nanodegree program. 
In this project, I build an image classifier using PyTorch, then convert it into a command line application.
The model can be saved to disk and loaded for usage at a later stage using the pytorch library.
The user has the option to train the model using the GPU or the CPU just by passing in the arguement of choice for example statring th programming with:
```
    python train.py --gpu --epochs 10 --save_dir dir
```
Starts the training program using the gpu is it exists on the host. 10 iterations are made and dir is used as the save and load directory for th model and it's parameters.
