# Import necessary packages.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.resnet import ResNet18
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
from main import train_acc, train_losses, test_acc, test_losses
from gradcam.utils import visualize_cam
from gradcam import GradCAM
import matplotlib.pyplot as plt


classes_dict = {
    0: "Airplane", 1: "Automobile", 2: "Bird", 3: "Cat", 4:"Deer", 5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9:"Truck"
}


def get_train_parameters():
    """Function to get the training cretarians.

    Returns:
        Objects: All training parameters.
    """
    use_cuda = torch.cuda.is_available() # Boolean to get whether cuda is there in device or not.
    device = torch.device("cuda" if use_cuda else "cpu") # If cuda is present set the device to cuda, otherwise set to cpu.
    model = ResNet18().to(device) # Set the model to same device.
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4) # Initialize the optimizer to Adam with learning rate 0.001.
    
    criterion = nn.CrossEntropyLoss() # Define the entropy loss function
    
    return model, device, optimizer, criterion

def get_schedular(optimizer, train_loader, epochs, max_lr):
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=5/epochs,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )
    return scheduler

def GetCorrectPredCount(pPrediction, pLabels):
    """Function to get the correct prediction count.

    Args:
        pPrediction (Object): Predicted tensors
        pLabels (Object): Actual labels of the images.

    Returns:
        Object(Tensor): If the predicted lables are actual labels then those items will be counted and returned.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()



def suggest_lr(train_loader,model, optimizer, criterian, device):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")

    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state


def plot_loss_accuracy():
    """Plot train and test losses and accuracy."""
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot([i.item() for i in train_losses])
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc[:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def get_misclassified(model, device, test_loader, n = 10):
    """Get mis clasified outputs and images"""
    model.eval()
    wr = 0
    images = []
    tr = []
    pr = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for k in range(0, len(data)):
              if pred[k].item()!=target[k].item():
                  images.append(data[k])
                  tr.append(target[k])
                  pr.append(pred[k])
                  wr+=1
                  if wr==10:
                    break
            if wr==10:
                break
    return images, tr, pr


def apply_gradcam_on_image(image, model, target_layer):
    # Prepare the image tensor
    input_tensor = image.unsqueeze(0).to(device)

    # Initialize GradCAM with the ResNet-18 model and the target layer
    gradcam = GradCAM(model, target_layer=target_layer)

    # Compute the GRADCAM heatmap
    heatmap, _ = gradcam(input_tensor)

    # Convert the heatmap to a NumPy array and normalize
    heatmap = heatmap.squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU on the heatmap
    heatmap /= np.max(heatmap)  # Normalize
    heatmap = np.expand_dims(heatmap,axis=0)
    heatmap = np.expand_dims(heatmap,axis=0)
    heatmap = torch.tensor(heatmap, dtype=torch.float32).to(device)
    # Convert the input image back to a PyTorch tensor
    # image_tensor = torch.tensor(np.transpose(image.numpy(), (1, 2, 0)), dtype=torch.float32).to(device)
    # Overlay the heatmap on the input image
    _, result = visualize_cam(heatmap, input_tensor)

    return result


def show_misclassified(img, trs, prs):
    """Plot misclassified images."""
    f, axarr = plt.subplots(2, 5, figsize=(15, 15))
    for i in range(0, 5):
        axarr[0,i].imshow(img[i].cpu().T)
        axarr[0,i].set_title(classes_dict[int(trs[i].item())] + "/"+classes_dict[prs[i].item()])
        axarr[1,i].imshow(img[i+5].cpu().T)
        axarr[1,i].set_title(classes_dict[int(trs[i+5].item())] + "/"+classes_dict[prs[i+5].item()])


def show_gradcam_results(img, trs, prs):
    """Plot misclassified gradcam images."""
    f, axarr = plt.subplots(2, 5, figsize=(15, 15))
    for i in range(0, 5):
        result = apply_gradcam_on_image(img[i], model, target_layer=model.layer4[-1])

        axarr[0, i].set_title(classes_dict[int(trs[i].item())] + "/ "+classes_dict[prs[i].item()])
        axarr[0, i].imshow(result.cpu().T)
        result = apply_gradcam_on_image(img[i+5], model, target_layer=model.layer4[-1], classes=classes)

        axarr[1, i].set_title(classes_dict[int(trs[i+5].item())] + "/ "+classes_dict[prs[i+5].item()])
        axarr[1, i].imshow(result.cpu().T)