import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from dataset import train_dataloader, validation_dataloader
from EarlyStopping import EarlyStopping


def train(EPOCHS, model, device):
    
    #model setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001,weight_decay=0.001)
    early_stopping = EarlyStopping(patience=10, delta=0.01, verbose=True)
     # Training loop
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            unit="batch",
        )
        for i, data in enumerate(pbar, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)  # forward 
            loss = criterion(outputs, labels) #  backward
            loss.backward() # optimize
            optimizer.step() # update weights
            #
            running_loss += loss.item()

            # print acc
            predicted = torch.argmax(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({
                "loss": running_loss / (i + 1),
                "acc": 100. * correct / total
            })

        #Validation 
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch")
            for i, data in enumerate(pbar, start=0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels) 
                val_loss += loss.item()

                #acc
                predicted = torch.argmax(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix({
                "Val_loss": val_loss / (i + 1),
                "acc": 100. * correct / total
                })

        #early stop if not learning
        avg_val_loss = val_loss / len(validation_dataloader)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    print('Finished Training')
    # Check if file was created
    PATH = './densenet.pth'
    if os.path.exists(PATH):
        print("✅ Model saved successfully.")
    else:
        print("❌ Save failed.")