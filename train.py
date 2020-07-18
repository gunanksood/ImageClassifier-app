import click
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models



# this method transforms training dataset
def data_transformer(dir_path):
    data_type = dir_path.split('/')[-1]
    if data_type == 'train':
        data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
    else:
        
        data_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])        
        

    transformed_data = datasets.ImageFolder(dir_path, transform=data_transforms)
    return transformed_data
    

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

def gpu_availability(gpu_arg):

    if not gpu_arg:
        print('Using default: ' + str(torch.device("cpu")))
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    
    print('Using: ' + str(device))
    return device

def model_selector(architecture="vgg16"):
    # Load Defaults if none specified
    model = eval("models.{}(pretrained=True)".format(architecture))
    model.name = architecture
    
    print('Selected architecture: ' + str(model.name))
    for param in model.parameters():
        param.requires_grad = False 
    return model


def classifier_initilization(model, hidden_units):
    if type(hidden_units) == type(None): 
        hidden_units = 4096
    
    # Find Input Layers
    input_features = model.classifier[0].in_features
    
    # Define Classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    print('classifier initialized')
    return classifier


def model_validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()   
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy


def training_network(model, trainloader, testloader, device, 
                  criterion, optimizer, epochs, print_every, steps):
    # Check Model Kwarg
    if type(epochs) == type(None):
        epochs = 5
        print("Number of epochs specificed as 5.")    
 
    print("Training process initializing .....\n with epochs: " + str(epochs))

    # Train Model
    for e in range(epochs):
        running_loss = 0
        model.train() # Technically not necessary, setting this for good measure
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = model_validation(model, testloader, criterion, device)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.3f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.3f} | ".format(valid_loss/len(testloader)),
                     "Validation Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
                running_loss = 0
                model.train()

    return model


def model_validation_checkpointing(Model, Testloader, Device, Save_Dir, Train_data):
   # Do validation on the test set
    correct_images = 0
    total_images = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            correct_images += (predicted == labels).sum().item()
    
    print('Accuracy achieved on test images is: %d%%' % (100 * correct_images / total_images))
    
    if isdir(Save_Dir):
        Model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
        checkpoint = {'architecture': Model.name,
                      'classifier': Model.classifier,
                      'class_to_idx': Model.class_to_idx,
                      'state_dict': Model.state_dict()}
            
        # Save checkpoint
        torch.save(checkpoint, 'my_checkpoint.pth')

    else: 
        print("Directory not found, model will not be saved.")

            
@click.command()
@click.option('--arch', type=str, help='Choose architecture from torchvision.models')
@click.option('--save_dir', required=True, type=str, help = 'Define save directory for checkpoints')
@click.option('--learning_rate', type=float, help='Define gradient descent learning rate')
@click.option('--hidden_units', type=int, help='Hidden units for DNN classifier')
@click.option('--epochs', type=int, help= 'Number of epochs for training')
@click.option('--gpu', is_flag=True, type=bool, help='Use GPU')
def main(arch, save_dir, learning_rate, hidden_units, epochs, gpu):
     
#     args = argument_parser()
    print(arch)
    print(save_dir)    
    print(learning_rate)    
    print(hidden_units)
    print(gpu)
    print(epochs)
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = data_transformer(train_dir)
    valid_data = data_transformer(valid_dir)
    test_data = data_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Load Model
    if arch:
        model = model_selector(architecture=arch)
    else:
        model = model_selector()
    
    # Build Classifier
    model.classifier = classifier_initilization(model, 
                                         hidden_units=hidden_units)
    
    # Check for GPU
    device = gpu_availability(gpu_arg=gpu)
    
    model.to(device)
    
    if type(learning_rate) == type(None):
        learning_rate = 0.001
    else: 
        learning_rate = learning_rate
    
    print("Learning rate specificed as " + str(learning_rate))    
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 30
    steps = 0
   
    # Train the classifier layers using backpropogation
    trained_model = training_network(model, trainloader, validloader, 
                                  device, criterion, optimizer, epochs, 
                                  print_every, steps)
    
    model_validation_checkpointing(trained_model, testloader, device, save_dir, train_data)
    print('Process Completed.')


if __name__ == '__main__': 
    main()