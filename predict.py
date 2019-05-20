#Imports necessary tools
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
import argparse



#Brings in arguments from CLI
parser = argparse.ArgumentParser()

parser.add_argument('--json_file', type=str, default='cat_to_name.json', help='Allows user to enter custom JSON file for category names.')
parser.add_argument('--test_file', type=str, default='flowers/train/43/image_02364.jpg', help='Allows user to run prediction on a given image.')
parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pth', help='Allows user to input a checkpoint file to load/build model from.')
parser.add_argument('--topk', type=int, default=5, help='Allows user to enter the top "k" predictions suggested by the model.')
parser.add_argument('--gpu', default='gpu', type=str, help='Determines where to run model: CPU vs. GPU')


#Maps parser arguments to variables for ease of use later
cl_inputs = parser.parse_args()

json_file = cl_inputs.json_file
test_file = cl_inputs.test_file
checkpoint_file = cl_inputs.checkpoint_file
topk = cl_inputs.topk
gpu = cl_inputs.gpu



#Imports inputted JSON file
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)



#Defines function to model from loaded checkpoint file
def load_model(checkpoint_file=checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    arch = checkpoint['arch']
    lr = checkpoint['lr']
    hidden_layer = checkpoint['hidden_layer']
    gpu = checkpoint['gpu']
    epochs = checkpoint['epochs']
    dropout = checkpoint['dropout']
    classifier = checkpoint['classifier']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)

    #model = Define_Model(arch, dropout, hidden_layer)
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False

    return model


#Builds model using Load_Model function and loaded checkpoint file
loaded_model = load_model()



#Processes the test_file image
def process_image(image=test_file):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Numpy array'''
    picture = Image.open(image)

    transformation = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])

    np_array = transformation(picture).float()

    return np_array



#Defines function to predict what the inputted image may represent according to the model
def predict(image_path=test_file, model=loaded_model, topk=topk, gpu=gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    #Processing image
    image = process_image(image_path)
    image = image.float().unsqueeze_(0)

    #Moving to GPU if user prompted
    if gpu == 'gpu':
        model.to('cuda:0')

    #Creating prediction score
    with torch.no_grad():
        if gpu == 'gpu':
            image = image.to('cuda')

        output = model.forward(image)

    prediction = F.softmax(output.data, dim = 1)

    probs, indices = prediction.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]

    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]

    return probs, classes



#Runs the "predict" function defined above on the test_file image (and prints out the outcome)
probs, classes = predict(test_file, loaded_model, topk)
print(probs)
print(classes)
