import sys
import argparse
import os

parser = argparse.ArgumentParser(description= "Command line trainer.")
parser.add_argument('data_dir', metavar = 'data_directory', type = str, help= "Data Folder.")
parser.add_argument('--save_dir', action = 'store', dest = 'save_dir', type = str,  help= "Directory to save check point.")
parser.add_argument('--arch', type = str, default = 'densenet121', help = 'densenet121 or vgg13.')
parser.add_argument('--gpu', action = 'store_true', dest = 'gpu', help= "Use gpu if available.")
parser.add_argument('--epoch', action = 'store', dest = 'epoch', type = int, help= "Specify number of epochs.")
parser.add_argument('--lr', action = 'store', dest = 'lr', type = float, help= "Learning Rate.")
parser.add_argument('--model', action = 'store', dest = 'model_name', type = str, help= "Choose model architeture, vgg16 or densenet. Default is vgg16")
args = parser.parse_args()

cwd = os.getcwd()  # Get the current working directory (cwd)

data_dir = args.data_dir
if os.path.isdir(data_dir) == False:
    print('Datafolder not found')
    exit(0)

model_name = args.model_name
if model_name == None:
    model_name = 'vgg16'
elif model_name not in ['vgg16', 'densenet']:
    print('choose between vgg16 or densenet model')
    exit(0)

epoch = args.epoch
if epoch == None:
    epoch = 15

if args.save_dir == None:
    save_dir = data_dir
else:
    save_dir = args.save_dir
    if os.path.isdir(save_dir) == False:
        print('save folder not found')
        exit(0) 

lr = args.lr
if lr == None:
    lr = 0.001

import torch
import time
import torchvision

if args.gpu:
    if torch.cuda.is_available() == False:
      print('No gpu detected , using cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
else:
    device = torch.device("cpu")


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
num_cat = len(os.listdir(train_dir)) 

train_data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(244),
    torchvision.transforms.RandomRotation(45),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(244),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_datasets = torchvision.datasets.ImageFolder(train_dir, transform = train_data_transforms)
valid_datasets = torchvision.datasets.ImageFolder(valid_dir, transform = valid_data_transforms)

train_data_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
valid_data_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)

dataloaders = {'train': train_data_loader,
              'valid' : valid_data_loader}

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

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_model(model, dataloader, criterion , optimizer, n_epochs = 15, val_steps = 5):
    start = time.time()
    validation_accuracy_history = []
    
    best_model_weights = model.state_dict()
    best_accuracy = model.accuracy
    
    for epoch in range(n_epochs):
        print(f'Epoch{epoch + 1}/{n_epochs}')
        print("-" * 10)
        steps = 0
        model.train()
                
        r_loss = 0.0
        r_corrects = 0
        total = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            steps += 1
            total += labels.size(0)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = model(inputs)
                loss = criterion(output, labels)
                _,preds = torch.max(output, 1)
                loss.backward()
                optimizer.step()
                r_loss += loss.item()
                r_corrects += (preds == labels.data).sum().item()
                if steps % val_steps == 0:
                    val_loss = 0
                    val_accuracy = 0
                    model.eval()
                    with torch.set_grad_enabled(False):
                        for v_inputs, v_labels in dataloaders['valid']:
                            v_inputs = v_inputs.to(device)
                            v_labels = v_labels.to(device)
                            val_out = model(v_inputs)
                            validation_loss = criterion(val_out, v_labels)
                            val_loss += validation_loss.item() * v_inputs.size(0)
                            _, v_preds = torch.max(val_out, 1)
                            val_accuracy += (v_preds == v_labels.data).sum().item()
                        val_accuracy = val_accuracy/ len(dataloaders['valid'].dataset)
                        val_loss /= len(dataloaders['valid'].dataset)
                        validation_accuracy_history.append(val_accuracy)
                    r_loss /= val_steps
                    r_accuracy = r_corrects / total
                    stop_v = time.time()
                    period_v = stop_v - start
                    print(f'Train Loss: {r_loss :.4f} ..Train Acc: {r_accuracy:.4f}..',
                    f'Validation Loss: {val_loss:.4f}.. val Acc: {val_accuracy:.4f}..',
                    f'{steps // val_steps} val complete in {period_v//60:.0f}m {period_v%60:.0f}s ')
                    if val_accuracy > model.accuracy:
                        model.accuracy = val_accuracy
                        best_model_weights = model.state_dict()
                    model.train()
                         
                
    stop = time.time()
    period = stop - start
    print('Training complete in {:.0f}m {:.0f}s '.format(period//60, period%60))
        
    model.load_state_dict(best_model_weights)
    return model, validation_accuracy_history

def create_optimizer(model, lr = 0.001):
    trainable_params = []
    for parameter in model.parameters():
        if parameter.requires_grad == True:
                trainable_params.append(parameter)
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    return optimizer


def save_model(save_dir = os.getcwd()):
    model.class_to_idx = train_datasets.class_to_idx
    model.opt_lr = lr
    checkpoint = {'class_to_idx': model.class_to_idx, 
              'model_state_dict': model.state_dict()}
    
    torch.save(checkpoint, os.path.join(save_dir ,'checkpoint.pth'))

model = get_model(model_name, pretrained = True)
model.to(device)
optimizer = create_optimizer(model, lr)
criterion = torch.nn.NLLLoss()
train_model(model, dataloaders, criterion , optimizer, n_epochs = epoch)
save_model(save_dir)

