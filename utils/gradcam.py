from data.cifar10_datamodule import *
from models.model import *
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    deprocess_image,
    preprocess_image,
    show_cam_on_image,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models.resnet import *
import math
import matplotlib.pyplot as plt

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

device = get_device()
# -------------------- GradCam --------------------

# Find 10 misclassified images, and show them as a 5x2 image matrix in 3 separately annotated images. 

def get_misclassified_data(model,test_loader):
    
    # Prepare the model for evaluation
    model.eval()

    # List for storing misclassified images
    misclassified_data = []

    # Reset the Gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:
            # Move the data to device
            data,target = data.to(device),target.to(device)

            # Extract single batch of images, labels
            for img,label in zip(data,target):
                
                # Add a batch dimension
                img = img.unsqueeze(0)
                # Get prediction
                output = model(img)
                # Convert output probabilities to predicted class through one-hot encoding
                pred = output.argmax(dim=1, keepdim=True)

                # Compare prediction and true label
                if pred.item() != label.item():
                    misclassified_data.append((img, label, pred))

    return misclassified_data

def display_gradcam_output(data: list,
classes: list[str],
inv_normalize: transforms.Normalize,
model: 'DL Model',
target_layers: list['model_layer'],
targets=None,
number_of_samples: int = 10,
transparency: float = 0.60):
    """
    Function to visualize GradCam output on the data
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param model: Model architecture
    :param target_layers: Layers on which GradCam should be executed
    :param targets: Classes to be focused on for GradCam
    :param number_of_samples: Number of images to print
    :param transparency: Weight of Normal image when mixed with activations
    """
    # Plot configuration
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples /
    x_count)
    # Create an object for GradCam
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # Iterate over number of specified images
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        input_tensor = data[i][0]
        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        # Get back the original image
        img = input_tensor.squeeze(0).to('cpu')
        img = inv_normalize(img)
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()
        # Mix the activations on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True,
        image_weight=transparency)
        # Display the images on the plot
        plt.imshow(visualization)
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' +
        classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig('gradcam.png')
