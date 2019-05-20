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
import argparse



#Imports cat_to_name.json file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



#Brings in arguments from CLI
parser = argparse.ArgumentParser()

#UPDATE 4: Added data_dir and save_dir as input to the ArgumentParser
parser.add_argument('--data_dir', type=str, default="./flowers/", help='Determines which directory to pull information from')
parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', help='Enables user to choose directory for saving')
parser.add_argument('--arch', type=str, default='vgg16', help='Determines which architecture you choose to utilize')
parser.add_argument('--learning_rate', type=float, default=.001, help='Dictates the rate at which the model does its learning')
parser.add_argument('--hidden_layer', type=int, default=1024, help='Dictates the hidden units for the hidden layer')
parser.add_argument('--gpu', default='gpu', type=str, help='Determines where to run model: CPU vs. GPU')
parser.add_argument('--epochs', type=int, default=3, help='Determines number of cycles to train the model')
parser.add_argument('--dropout', type=float, default=0.5, help='Determines probability rate for dropouts')



#Maps parser arguments to variables for ease of use later
cl_inputs = parser.parse_args()

data_dir = cl_inputs.data_dir
save_dir = cl_inputs.save_dir
arch = cl_inputs.arch
lr = cl_inputs.learning_rate
hidden_layer = cl_inputs.hidden_layer
gpu = cl_inputs.gpu
epochs = cl_inputs.epochs
dropout = cl_inputs.dropout



#Pulls in and transforms data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


valtest_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=valtest_transforms)
testing_data = datasets.ImageFolder(test_dir, transform=valtest_transforms)

trainloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testing_data, batch_size=32, shuffle=True)



#Defines the model
def Classifier(arch='vgg16', dropout=0.5, hidden_layer=1024):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216

    for param in model.parameters():
        param.requires_grad = False

    my_classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('fc1', nn.Linear(input_size, hidden_layer)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_layer, 256)),
            ('output', nn.Linear(256, 102)),
            ('softmax', nn.LogSoftmax(dim = 1))]))

    model.classifier = my_classifier

    return model



#Establishs the model, criterion, and optimizer
model = Classifier(arch, dropout, hidden_layer)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

#Defines the function to train the defined model
def train_model(model=model, trainloader=trainloader, criterion=criterion, optimizer=optimizer, epochs=epochs, gpu=gpu):
    steps = 0
    print_every = 40

    if gpu == 'gpu':
        model.to('cuda')

    for e in range(epochs):
        running_loss = 0

        for images, labels in trainloader:
            steps += 1

            #Moving images / labels to GPU per user input
            if gpu == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')

            #Zero-ing out the optimizer's gradient
            optimizer.zero_grad()

            #Calculating loss and updating weights
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #Adjusting running loss
            running_loss += loss.item()

            #Calculating validation loss & accuracy every 20 steps
            if steps % print_every == 0:
                #Moving model to eval mode
                model.eval()

                val_loss = 0
                accuracy = 0

                #Calculating validation loss & accuracy
                for val_images, val_labels in validloader:

                    #Moving to GPU per user input
                    if gpu == 'gpu':
                        val_images, val_labels = val_images.to('cuda'), val_labels.to('cuda')

                    #Turning off gradient for validation purposes
                    with torch.no_grad():
                        val_outputs = model.forward(val_images)
                        val_loss = criterion(val_outputs, val_labels)

                        #Calculating probability from the val_outputs
                        ps = torch.exp(val_outputs)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == val_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                #Printing out our metrics along the way
                print('Epoch: {} / {}..'.format(e + 1, epochs),
                      'Training Loss: {:.3f}..'.format(running_loss / print_every),
                      'Validation Loss: {:.3f}..'.format(val_loss / len(validloader)),
                      'Validation Accuracy: {:.3f}..'.format(accuracy / len(validloader)))

                running_loss = 0
                #Turning the dropouts back on
                model.train()



#Trains the model
train_model(model, trainloader, criterion, optimizer, epochs, gpu)



#Defines accuracy checking function
def test_checker(testloader=testloader, gpu=gpu):
correct = 0
total = 0

#Moving to GPU based on user input
if gpu == 'gpu':
    model.to('cuda')

#Testing with the gradients turned off
with torch.no_grad():
    for images, labels in testloader:
        if gpu == 'gpu':
            images, labels = images.to('cuda'), labels.to('cuda')

        #Calculating accuracy
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test data set: {}%'.format((correct / total) * 100))



#Runs accuracy checker functional
test_checker(testloader)



#Saving the checpoint to the save_dir path
model.class_to_idx = training_data.class_to_idx
checkpoint = {'arch': arch,
              'lr': lr,
              'hidden_layer': hidden_layer,
              'gpu': gpu,
              'epochs': epochs,
              'dropout': dropout,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir)
