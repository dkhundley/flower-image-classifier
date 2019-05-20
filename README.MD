# Image Classifier Project
In this project, we'll be creating a deep learning network to classify flowers per the labels provided. This project was established by Udacity and performed within Udacity's GPU enabled workspace, so unfortunately the source files for this project are not included. The project also utilizes transfer learning to import already trained classifiers from the PyTorch package while modifying the classifier attribute of each package.

## Project Breakdown
The files work through the project in the following manners:
 - **Creating the Datasets**: Utilizing the images provided by Udacity, the first part of the project looks to import the data while applying proper transforms and segmenting them into respective training, validation, and testing datasets
 - **Creating the Architecture**: Utilizing the pre-trained models from PyTorch's torchvision package, we establish different classifier paramaters to fit our datasets as well as establishing an NLL Loss criterion and Adam optimizer
 - **Training the Model**: With help from PyTorch and Udacity's GPU-enabled platform, we train our model across our training and validation datasets to create an ideal model for classifying the flowers.
 - **Saving / Loading the Model**: To practice utilizing the model in other platforms, we export the model to a 'checkpoint.pth' file and re-load / rebuild it in another file.
 - **Class Prediction**: Finally, we use our newly trained model to make a prediction of a flower given a testing input image.

## Files Included
These are the files included as part of the project and what each contains:
 - **Image Classifier Project.ipynb**: This is the Jupyter notebook where I conducted all my activities, including a little more than what is included in the predict.py and train.py files.
 - **Image Classifier Project.html**: Same as the file above, except in HTML form.
 - **train.py**: This file accepts inputs from the command line prompt and takes the work from the Jupyter notebook for the following activities:
  - Creating the Datasets
  - Creating the Architecture
  - Training the model
  - Saving the Model

- **predict.py**: This file accepts inputs from the command line prompt and takes the work from the Jupyter notebook for the following activities
  - Loading the Model
  - Class Prediction
