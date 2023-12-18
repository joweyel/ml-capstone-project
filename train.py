from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from DINOClassifier import DINOClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Training & evaluation scripts ###

def accuracy_fn(outputs, labels):
    pred = torch.round(outputs)
    correc_pred = (pred == labels).sum().float()
    acc = correc_pred / len(outputs)
    return acc

def train(model, optimizer,loss_fn, n_epochs, train_loader, val_loader):

    history = {
        'loss': [], 
        'val_loss': [], 
        'accuracy': [], 
        'val_accuracy': []
    }

    for epoch in range(n_epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_accuracy = 0.0
        val_accuracy = 0.0

        # Training-loop
        model.train()
        for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Training]"):
            # Getting the Image(s) and label(s)
            inputs, labels = data

            # Put data to gpu (if available)
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            # Set all accumulated grad's to 0 to start new backprop
            outputs, _ = model(inputs)      
            outputs = outputs.squeeze()

            # Compute Loss
            loss = loss_fn(outputs, labels)

            # Perform optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(loss)
            train_loss += loss.item()
            train_accuracy += accuracy_fn(outputs, labels).item()

        # Validation-loop
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Validation]"):
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.float().to(device)
                
                outputs, _ = model(inputs)
                outputs = outputs.squeeze()

                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                val_accuracy += accuracy_fn(outputs, labels).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_accuracy /= len(train_loader)
        val_accuracy /= len(val_loader)
    
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        print(f"loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")

    return history

def eval(model, loss_fn, test_loader):
    history = {
        'loss': [], 
        'accuracy': []
    }

    test_loss = 0.0
    test_accuracy = 0.0

    # Validation-loop
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"[Test]"):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            outputs, _ = model(inputs)
            outputs = outputs.squeeze()

            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            test_accuracy += accuracy_fn(outputs, labels).item()

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    history['loss'].append(test_loss)
    history['accuracy'].append(test_accuracy)

    print(f"loss: {test_loss:.4f} - accuracy: {test_accuracy:.4f}")
    return history    

data_transform = T.Compose([
    T.Lambda(lambda image: image.convert('RGB')),  # Convert to RGB if necessary
    T.Resize((224, 224)),  # Resize to the same size
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),  # ImageNet statistics
])

inv_normalize = T.Normalize(mean=[-0.485/0.229,-0.456/0.224,-0.406/0.225],
                            std=[1/0.229,1/0.224,1/0.225])

def train_val_test_split(dataset, train_size, val_size, test_size):
    # Create indices for each split of the original dataset
    train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=test_size)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size / (train_size + val_size))

    # Create subset based on indices of different 
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, val_set, test_set

#################################################################################

# Parameters
dataset_path = "data/Car-Bike-Dataset"
image_size = 224 # required input-size for classification model
learning_rate = 0.007262717757314849
momentum = 0.8
batch_size = 64
n_epochs = 5

model_cfg = {
    "model_type": "s",
    "hidden_layer_dims": [64],
    "use_dropout": False, 
    "dropout_prob": 0.0, 
    "device": device
}

# creating the model
model = DINOClassifier(**model_cfg)

# Binary Cross-Entropy loss
loss_fn = nn.BCELoss()
# SGD-optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


### Load Datasets ###
dataset = datasets.ImageFolder(dataset_path, transform=data_transform)
splits = { "train_size": 0.6, "val_size": 0.2, "test_size": 0.2 }
train_data, val_data, test_data = train_val_test_split(dataset, **splits)

### Load DataLoaders ###
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


# Training the model
history = train(
    model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    n_epochs=n_epochs,
    train_loader=train_loader,
    val_loader=val_loader
)

# Evaluating on the test-set
test_history = eval(model,
                    loss_fn=loss_fn,
                    test_loader=test_loader)


# Torch: saving the trained model
model = model.to("cpu")
model_name = "model_v1"
torch.save(model, f"{model_name}.pth")