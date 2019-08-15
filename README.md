# AI Programming with Python Project by Udacity

Final project files for creating an image classifer for 102 different flowers.
This is part of the requirements for completing the Udacity's AI Programming with Python Nanodegree program.
This project, I build an image classifier using PyTorch, then convert it into a command line application.
The model can be saved to disk and loaded for usage at a later stage using the pytorch library.

## Requirements
This program requires the following modules to execute:
1. python 3 and later
2. pytorch 1.2
3. numpy
4. pandas
5. IMAGE PIL
6. torchvision

Altenatively, you can use [anconda](https://www.anaconda.com/distribution/) to manage your AI or data science environment.

## Quckstart
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

## LICENSE

This repository and the associated files is available under the below MIT License:

    Copyright(c) 2018 Udacity

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.