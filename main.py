import torch
import os
from denseNet import DenseNet
from train import train




if __name__ == '__main__':
    EPOCHS = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    # Check if file was created
    PATH = '/content/drive/MyDrive/densenet.pth'
    if os.path.exists(PATH):
        model = DenseNet().to(device)
        model.load_state_dict(torch.load(PATH, weights_only=True))
    else:
        model = DenseNet().to(device)

    train(EPOCHS, model, device)
