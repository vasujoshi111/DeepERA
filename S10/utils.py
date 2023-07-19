# Import necessary packages.
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Net


classes_dict = {
    0: "Airplane", 1: "Automobile", 2: "Bird", 3: "Cat", 4:"Deer", 5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9:"Truck"
}
def get_train_parameters(train_loader, epochs):
    """Function to get the training cretarians.

    Returns:
        Objects: All training parameters.
    """
    use_cuda = torch.cuda.is_available() # Boolean to get whether cuda is there in device or not.
    device = torch.device("cuda" if use_cuda else "cpu") # If cuda is present set the device to cuda, otherwise set to cpu.
    model = Net().to(device) # Set the model to same device.
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9) # Initialize the optimizer to SGD with learning rate 0.01 and momentum 0.9.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=4.93E-02,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=5/epochs,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )
    criterion = F.nll_loss # Define the entropy loss function
    
    return model, device, optimizer, scheduler, criterion


def GetCorrectPredCount(pPrediction, pLabels):
    """Function to get the correct prediction count.

    Args:
        pPrediction (Object): Predicted tensors
        pLabels (Object): Actual labels of the images.

    Returns:
        Object(Tensor): If the predicted lables are actual labels then those items will be counted and returned.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def get_misclassified(model, device, test_loader, n = 10):
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
                if wr==n:
                    break
            if wr==n:
                break
    return images, tr, pr