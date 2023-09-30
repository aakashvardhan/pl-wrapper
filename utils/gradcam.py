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

def collect_misclassified_images(self, num_images):
        misclassified_images = []
        misclassified_true_labels = []
        misclassified_predicted_labels = []
        num_collected = 0

        for batch in self.test_dataloader():
            x, y = batch
            y_hat = self.forward(x)
            pred = y_hat.argmax(dim=1, keepdim=True)
            misclassified_mask = pred.eq(y.view_as(pred)).squeeze()
            misclassified_images.extend(x[~misclassified_mask].detach())
            misclassified_true_labels.extend(y[~misclassified_mask].detach())
            misclassified_predicted_labels.extend(pred[~misclassified_mask].detach())

            num_collected += sum(~misclassified_mask)

            if num_collected >= num_images:
                break

        return misclassified_images[:num_images], misclassified_true_labels[:num_images], misclassified_predicted_labels[:num_images], len(misclassified_images)


def plot_grad_cam(model, target_layer, imgs_list, preprocess_args, **kwargs):
    misclassified_images, true_labels, predicted_labels, num_misclassified = collect_misclassified_images(num_images)
    count = 0
    k = 0
    misclassified_images_converted = list()
    gradcam_images = list()

    if target_layer == -2:
        target_layer = self.convblock2_l1.cpu()
    else:
        target_layer = self.convblock3_l1.cpu()
    rows, cols = int(len(imgs_list) / 5), 5
    figure = plt.figure(figsize=(cols * 2, rows * 2))

    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=torch.cuda.is_available())
    cam.batch_size = 32

    for i, img in enumerate(imgs_list):
        rgb_img = np.float32(img) / 255
        input_tensor = preprocess_image(rgb_img, **preprocess_args)

        grayscale_cam = cam(input_tensor=input_tensor, targets=None, **kwargs)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        figure.add_subplot(rows, cols, i + 1)  # adding sub plot
        plt.axis("off")  # hiding the axis
        plt.imshow(cam_image, cmap="rainbow")  # showing the plot

    plt.tight_layout()
    plt.show()