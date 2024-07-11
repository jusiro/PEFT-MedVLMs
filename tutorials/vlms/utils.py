import torch

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Feature extraction using vision encoder trough a dataset
def extract_features(loader, model):
    epoch_iterator = tqdm(loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)

    X, Y = [], []
    for step, batch in enumerate(epoch_iterator):
        images = batch["image"].to(device).to(torch.float32)

        with torch.no_grad():
            # Forward vision encoder
            x = model({"pixel_values": images}) # we keep consistency in the input of model wrapper using the dict.

        X.extend(x.cpu().detach().numpy())
        Y.extend(batch["label"].numpy())

    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

# Evaluate predictions and return metrics
def evaluate(refs, preds):
    
    # Confusion matrix
    cm = confusion_matrix(refs, np.argmax(preds, -1))
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))

    # Accuracy per class - and average
    acc_class = list(np.round(np.diag(cm_norm), 3))
    aca = np.round(np.mean(np.diag(cm_norm)) * 100, 2)

    return aca, cm

# Function to train a black-box Adapter using pre-computed features, X.
def train_adapter(X, Y, adapter, optimizer, batch_size, epochs):
        
        # Loop over epochs
        for epoch in range(epochs):
            
            # Set features
            x = torch.tensor(X).clone().to(device)
            y = torch.tensor(Y).clone().to(device)

            # Randomly shuffle for this epoch
            idx = np.random.rand(x.shape[0]).argsort(axis=0)
            x, y = x[idx, :], y[idx]

            loss_epoch = 0. # Init train loss for tracking porpuses
            iterations = max(x.shape[0]//batch_size, 1) # Number of forward-backward give batch size and samples
            for batch_idx in range(iterations):
                # Select sample indexes
                batch_init = batch_idx * batch_size
                batch_end = (batch_idx + 1) * batch_size
                # Forward
                logits = adapter(x[batch_init:batch_end, :])
                # Compute loss (we assume the adapter has defined a specific loss function)
                loss = adapter.loss(logits, y[batch_init:batch_end])

                # Update adapter
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track losses
                loss_epoch += loss.item()/iterations
            print('Epoch=%d: loss_train=%2.4f' % (epoch+1, loss_epoch), end="\n")
    

# Function fine-tune a model
def train_ft(loader, model, optimizer, batch_size, epochs):
   
    epoch_iterator = tqdm(loader, desc="Fine-tuning (X / X Steps)", dynamic_ncols=True)

    # Loop over epochs
    for epoch in range(epochs):
        
        loss_epoch = 0. # Init train loss for tracking porpuses
        for step, batch in enumerate(epoch_iterator):
            
            # Retrieve batch of images and labels
            images = batch["image"].to(device).to(torch.float32)
            Y = batch["label"].to(device)
            
            # Forward
            logits = model({"pixel_values": images}) 
                
            # Compute loss (we assume the head has defined a specific loss function)
            loss = model[1].loss(logits, Y)
            
            # Update adapter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses
            loss_epoch += loss.item()/len(epoch_iterator)
        
        # Display epoch-wise losses
        print('Epoch=%d: loss_train=%2.4f' % (epoch+1, loss_epoch), end="\n")

        
# Function fine-tune a model
def predict(loader, model):
    epoch_iterator = tqdm(loader, desc="Predicting (X / X Steps)", dynamic_ncols=True)

    Y, Yhat = [], []
    for step, batch in enumerate(epoch_iterator):
        images = batch["image"].to(device).to(torch.float32)

        with torch.no_grad():
            # Forward vision encoder + clasification head
            out = model({"pixel_values": images}) # we keep consistency in the input of model wrapper using the dict.

        Y.extend(batch["label"].numpy())
        Yhat.extend(out.detach().cpu().numpy())
        
    # Concatenate predictions
    Y = np.array(Y)
    Yhat = np.array(Yhat)
    
    return Yhat, Y
    