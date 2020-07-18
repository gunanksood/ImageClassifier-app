import click
import json
import PIL
import torch
import numpy as np

from math import ceil
from train import gpu_availability
from torchvision import models


def checkpoint_loader(checkpoint_path):
    # Load the saved file
    checkpoint = torch.load("my_checkpoint.pth")
    
    model = eval("models.{}(pretrained=True)".format(checkpoint['architecture']))
    model.name = checkpoint['architecture']
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
  
    return model


def process_image(image_path):
    test_image = PIL.Image.open(image_path)

    orig_width, orig_height = test_image.size

    # Finding shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: 
        resize_size=[256, 256**600]
    else: 
        resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(test_image)

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std    
#     np_image = (np_image - normalise_means)
#     np_image = np_image / normalise_std
        
    np_image = np_image.transpose(2, 0, 1)
    return np_image


def predict(image_tensors, model, device, category_to_name, top_k):    
    # check top_k
    if type(top_k) == type(None):
        top_k = 5
        print("Top K not specified, assuming K=5.")
    
    # Set model to evaluate
    model.eval();

    torch_image = torch.from_numpy(np.expand_dims(image_tensors, 
                                                  axis=0)).type(torch.FloatTensor)
    model=model.cpu()
    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [category_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))

    
@click.command()
@click.option('--category_names', required=True, type=str, help = 'Mapping from categories to real names.')
@click.option('--image', type=str, required=True, help='Point to image file for prediction.')
@click.option('--checkpoint', type=str, required=True, help='Point to checkpoint file')
@click.option('--top_k', type=int, help= 'Choose top K matches')
@click.option('--gpu', is_flag=True, type=bool, help='Use GPU')    
def main(category_names, image, checkpoint, top_k, gpu):
    with open(category_names, 'r') as f:
        	category_to_name = json.load(f)

    model = checkpoint_loader(checkpoint)
    
    # image processing
    image_tensors = process_image(image)
    
    device = gpu_availability(gpu_arg=gpu);
    
    # Using `processed_image` to predict the top K most likely classes
    top_probs, top_labels, top_flowers = predict(image_tensors, model, 
                                                 device, category_to_name,
                                                 top_k)
    
    print_probability(top_flowers, top_probs)

    
if __name__ == '__main__': 
    main()