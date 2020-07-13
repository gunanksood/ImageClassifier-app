import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    # --save_dir is mandatory argument and if you are using GPU then please add --gpu while running this script.
    
    parser = argparse.ArgumentParser(description="Neural Network config")
    
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models')
    
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints')
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training')

    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU')
    
    # Parse args
    args = parser.parse_args()
    return args

# this method transforms training dataset
def train_transformer(train_dir):
    # Define transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    # Load the Data
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    print('training data transformed')
    return train_data

# this method transforms test / validation dataset
def test_transformer(test_dir):
    # Define transformation
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    print('validation/test data transformation done.')
    return test_data
    

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

def check_gpu(gpu_arg):
   # If gpu_arg is false then simply return the cpu device
    if not gpu_arg:
        print('Using default: ' + torch.device("cpu"))
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    print('Using: ' + str(device))
    return device

def primaryloader_model(architecture="vgg16"):
    # Load Defaults if none specified
    if type(architecture) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    print('Selected architecture: ' + str(model.name))
    for param in model.parameters():
        param.requires_grad = False 
    return model


def initial_classifier(model, hidden_units):
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


def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy


def network_trainer(model, trainloader, testloader, device, 
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
                    valid_loss, accuracy = validation(model, testloader, criterion, device)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(testloader)))
            
                running_loss = 0
                model.train()

    return model

#Function validate_model(Model, Testloader, Device) validate the above model on test data images
def validate_model(Model, Testloader, Device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

    
def initial_checkpoint(Model, Save_Dir, Train_data):
       
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            # Create `class_to_idx` attribute in model
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

def main():
     
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = test_transformer(train_dir)
    valid_data = train_transformer(valid_dir)
    test_data = train_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Load Model
    model = primaryloader_model(architecture=args.arch)
    
    # Build Classifier
    model.classifier = initial_classifier(model, 
                                         hidden_units=args.hidden_units)
     
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    model.to(device);
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
    else: learning_rate = args.learning_rate
    
    print("Learning rate specificed as " + str(learning_rate))    
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 30
    steps = 0
   
    # Train the classifier layers using backpropogation
    trained_model = network_trainer(model, trainloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining process is completed")
    
    validate_model(trained_model, testloader, device)
    
    initial_checkpoint(trained_model, args.save_dir, train_data)
    print('Process Completed.')


if __name__ == '__main__': 
    main()