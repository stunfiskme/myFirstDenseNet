import torch
from denseNet import DenseNet
from train import train




if __name__ == '__main__':
    EPOCHS = 50
    #use gpu if its there
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = DenseNet().to(device)
    train(EPOCHS, model, device)
