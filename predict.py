import sys
import argparse
import os

parser = argparse.ArgumentParser(description= "Command line trainer")
parser.add_argument('data_dir', metavar = 'path', type = str, help= "Path to image")
parser.add_argument('checkpoint', metavar = 'checkpoint', type = str,  help= "Directory to load check point")
parser.add_argument('--model_name', metavar = 'model_name', type = str,  help = 'specify network architecture used in chkpoint. Default is vgg16')
parser.add_argument('--gpu', action = 'store_true', dest = 'gpu',  help= "use gpu if available")
parser.add_argument('--topk', action = 'store', dest = 'topk', type = int, help= "select number of top probabilities" )

parser.add_argument('--category_names', action = 'store', dest = 'cat_names',  help= "file for category names for each label" )
args = parser.parse_args()

image = args.data_dir
if os.path.isfile(image) == False:
    print('Image file not found')
    exit(0)


chkpoint_pth = args.checkpoint
if os.path.isfile(chkpoint_pth) == False:
    print('Checkpoint file not found')
    exit(0)

model_name = args.model_name
if model_name == None:
    model_name = 'vgg16'
elif model_name not in ['vgg16', 'densenet']:
    print('choose between vgg16 or densenet model')
    exit(0)

category_to_name = args.cat_names
if category_to_name != None:
    if os.path.isfile(category_to_name) == False:
        print('Categorical file not found')
        exit(0)



if args.topk == None:
    topk = 5
else:
    topk = args.topk

import torch
import torchvision
import numpy as np

import json
import PIL
from PIL import Image

if args.gpu:
    if torch.cuda.is_available() == False:
      print('No gpu detected , using cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
else:
    device = torch.device("cpu")


def get_model(name , pretrained = True):
    if pretrained == False:
        if name == 'vgg16':
            model = torchvision.models.vgg16()
            model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 12544),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.Linear(12544,3136),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout (0.5),
                                       torch.nn.Linear(3136, 784),
                                       torch.nn.Linear(784, 102),
                                       torch.nn.LogSoftmax(dim=1))
        elif name == 'densenet':
            model = torchvision.models.densenet121()
            model.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 512),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.Linear(512,256),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout (0.5),
                                       torch.nn.Linear(256, 102),
                                       torch.nn.LogSoftmax(dim=1))
    else:
        if name == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 12544),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.Linear(12544,3136),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout (0.5),
                                       torch.nn.Linear(3136, 784),
                                       torch.nn.Linear(784, 102),
                                       torch.nn.LogSoftmax(dim=1))
        elif name == 'densenet':
            model = torchvision.models.densenet121(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 512),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.Linear(512,256),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout (0.5),
                                       torch.nn.Linear(256, 102),
                                       torch.nn.LogSoftmax(dim=1))
    model.accuracy = 0
    return model

def create_optimizer(model, lr = 0.001):
    trainable_params = []
    for parameter in model.parameters():
        if parameter.requires_grad == True:
                trainable_params.append(parameter)
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    return optimizer

def load(path):
    checkpoint = torch.load(path, map_location= 'cpu')
    model = get_model(model_name, pretrained=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    return model

model = load(chkpoint_pth)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img.thumbnail((255,255), Image.ANTIALIAS)
    t = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    img = t(img)
    np_image = np.array(img)
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        prediction = model.forward(image)
        
    predicted_prob = torch.exp(prediction)
    
    probs, inds = predicted_prob.topk(topk)
    probs = list(probs.to('cpu').numpy()[0])
    inds = list(inds.to('cpu').numpy()[0])
    inv_dict = {}
    for key, val in model.class_to_idx.items():
        inv_dict[val] = key
    
    classes = [inv_dict[item] for item in inds]
    
    return probs, classes

prob, classes = predict(image, model, topk)
names = {}

if category_to_name != None:
    with open(category_to_name, 'r') as f:
        cat_to_name = json.load(f)

    for i in range(len(prob)):
        names[cat_to_name[classes[i]]] = prob[i]

    for key in list(names.keys()):
        print('Class = ', key, ' probability = ', names[key])
else:
    print('Top k Labels: ', classes, '.. Probs: ', prob)
