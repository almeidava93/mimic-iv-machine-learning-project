import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import time, copy
import matplotlib.pyplot as plt
import numpy as np


# Load data

X_train = pd.read_csv(Path('cleaned_data','X_train.csv'), index_col=0)
X_val = pd.read_csv(Path('cleaned_data','X_val.csv'), index_col=0)
X_test = pd.read_csv(Path('cleaned_data','X_test.csv'), index_col=0)

y_train = pd.read_csv(Path('cleaned_data','y_train.csv'), index_col=0)
y_val = pd.read_csv(Path('cleaned_data','y_val.csv'), index_col=0)
y_test = pd.read_csv(Path('cleaned_data','y_test.csv'), index_col=0)

# Define device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device



# Define custom dataset

class MimicIvDataset(Dataset):
    """MIMIC IV dataset."""

    def __init__(self, csv_file_X, csv_file_y):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
        """
        self.mimic_df_X = pd.read_csv(Path(csv_file_X), index_col=0)
        self.mimic_df_y = pd.read_csv(Path(csv_file_y), index_col=0)

    def __len__(self):
        return len(self.mimic_df_X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        inputs = torch.tensor(self.mimic_df_X.iloc[idx], dtype=torch.float64)
        labels = torch.tensor(self.mimic_df_y.iloc[idx], dtype=torch.float64)

        return inputs, labels

# Load datasets

train_dataset = MimicIvDataset(csv_file_X="cleaned_data/X_train.csv", csv_file_y="cleaned_data/y_train.csv")
val_dataset = MimicIvDataset(csv_file_X="cleaned_data/X_val.csv", csv_file_y="cleaned_data/y_val.csv")
test_dataset = MimicIvDataset(csv_file_X="cleaned_data/X_test.csv", csv_file_y="cleaned_data/y_test.csv")


# Create dataloaders

batch_size = 100

dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
               'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True),
               'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=True)}

dataset_sizes = {'train': len(train_dataset),
                 'val': len(val_dataset),
                 'test': len(test_dataset)}
print(f'dataset_sizes = {dataset_sizes}')


import torch
import torch.nn as nn

# Define model parameters
input_size = 67
hidden_size1 = 128
hidden_size2 = 128
hidden_size3 = 128
num_classes = 1
dropout_rate = 0.5  # Adjust the dropout rate for regularization

# External training parameters
learning_rate = 0.0001
num_epochs = 40

# Revised model with flexible activation and data types
class MimicAdmissionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes, dropout_rate, activation_fn=nn.ReLU, dtype=torch.float32):
        """
        A customizable multi-layer perceptron (MLP) model with dropout for classification tasks.

        Args:
            input_size (int): Size of the input features.
            hidden_size1 (int): Number of units in the first hidden layer.
            hidden_size2 (int): Number of units in the second hidden layer.
            hidden_size3 (int): Number of units in the third hidden layer.
            num_classes (int): Number of output classes (1 for binary classification).
            dropout_rate (float): Dropout rate to use between layers (0 for no dropout).
            activation_fn (torch.nn.Module): Activation function (default is ReLU).
            dtype (torch.dtype): Data type for the model (default is torch.float32).
        """
        super(MimicAdmissionClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer to prevent overfitting

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size1, dtype=dtype),
            activation_fn(),
            self.dropout,
            nn.Linear(hidden_size1, hidden_size2, dtype=dtype),
            activation_fn(),
            self.dropout,
            nn.Linear(hidden_size2, hidden_size3, dtype=dtype),
            activation_fn(),
            self.dropout,
            nn.Linear(hidden_size3, num_classes, dtype=dtype),
            nn.Sigmoid()
        )
        
        # Define the sequence of layers with dynamic activation function
        self.layers_1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1, dtype=dtype),
            activation_fn(),
            self.dropout,
        )
        
        self.layers_2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2, dtype=dtype),
            activation_fn(),
            self.dropout,
            nn.Linear(hidden_size2, hidden_size3, dtype=dtype),
            activation_fn(),
            self.dropout,
        )
        
        self.layers_3 = nn.Sequential(
            nn.Linear(hidden_size3, num_classes, dtype=dtype),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input data tensor of shape (batch_size, input_size).
        
        Returns:
            Tensor: Output prediction tensor of shape (batch_size, num_classes).
        """
        # out = self.layers_1(x)
        # out = self.layers_2(out) + out # skip connection
        # out = self.layers_3(out)

        return self.mlp(x)

# Example instantiation of the model
mimic_admission_classifier = MimicAdmissionClassifier(
    input_size=input_size,
    hidden_size1=hidden_size1,
    hidden_size2=hidden_size2,
    hidden_size3=hidden_size3,
    num_classes=num_classes,
    dropout_rate=dropout_rate,
    activation_fn=nn.ReLU,  # Can be changed to other activations like nn.LeakyReLU
    dtype=torch.float64  # Data type can be changed if needed
).to(device)

# Print model summary (optional)
print(mimic_admission_classifier)


# From https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryAUROC

def train_classification_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=5, threshold=0.5, eval=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # keep the best weights stored separately
    best_acc = 0.0
    best_epoch = 0

    # initialize metric
    f1 = BinaryF1Score(threshold=threshold).to(device)
    acc = BinaryAccuracy(threshold=threshold).to(device)
    auc = BinaryAUROC().to(device)

    # Each epoch has a training, validation, and test phase
    phases = ['train', 'val', 'test']
    
    # Keep track of how loss and accuracy evolves during training
    training_curves = {}
    for phase in phases:
        training_curves[phase+'_loss'] = []
        training_curves[phase+'_acc'] = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            since_phase = time.time()
                        
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # No need to flatten the inputs!
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + update weights only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                acc(outputs.detach(), labels.detach())
                            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = acc.compute()
            training_curves[phase+'_loss'].append(epoch_loss)
            training_curves[phase+'_acc'].append(epoch_acc)       

            phase_time_elapsed = time.time() - since_phase

            print(f'{phase:5} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time elapsed: {phase_time_elapsed // 60:.0f}m {phase_time_elapsed % 60:.0f}s')

            # deep copy the model if it's the best F1
            if phase == 'val' and epoch_acc > best_acc:
              best_epoch = epoch
              best_acc = epoch_acc
              best_model_wts = copy.deepcopy(model.state_dict())

            # reset metrics
            acc.reset()

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} at epoch {best_epoch}')

    # Compute relevant performance metrics
    print('\nEvaluating trained model')
    with torch.set_grad_enabled(False):
        model.eval()
        if eval:
            for phase in phases:
                if phase=='train': continue

                running_loss = 0.0
                since_phase = time.time()
                            
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    # No need to flatten the inputs!
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    acc(outputs.detach(), labels.detach())
                    f1(outputs.detach(), labels.detach())
                    auc(outputs.detach(), labels.detach())

                # Compute final metrics
                final_loss = running_loss / dataset_sizes[phase]
                final_acc = acc.compute()
                final_f1 = f1.compute()
                final_auc = auc.compute()

                phase_time_elapsed = time.time() - since_phase

                print(f'{phase:5} Loss: {final_loss:.4f} Acc: {final_acc:.4f} F1: {final_f1:.4f} AUC: {final_auc:.4f} Time elapsed: {phase_time_elapsed // 60:.0f}m {phase_time_elapsed % 60:.0f}s')

                # Reset metrics
                f1.reset()
                acc.reset()
                auc.reset()


    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, training_curves



    # Training
# loss and optimizer
criterion = nn.BCELoss() # BCELoss for binary classification
optimizer = torch.optim.Adam(mimic_admission_classifier.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Train the model. We also will store the results of training to visualize
mimic_admission_classifier, training_curves = train_classification_model(mimic_admission_classifier, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=num_epochs)

torch.save(mimic_admission_classifier.state_dict(), "mimic_admission_classifier_test.pt")

import sklearn.metrics as metrics

# Utility functions for plotting your results!
def plot_training_curves(training_curves, 
                         phases=['train', 'val', 'test'],
                         metrics=['loss','acc']):
    epochs = list(range(1, len(training_curves['train_loss'])+1))
    fig, axs = plt.subplots(len(metrics), 1, figsize=(5, 10), tight_layout=True)
    for idx, metric in enumerate(metrics):
        ax = axs[idx]     
        ax.set_title(f'Training curves - {metric}')
        ax.set_ylabel(metric)
        ax.xaxis.set_ticks(np.arange(1, len(epochs)+1, 1))
        for phase in phases:
            key = phase+'_'+metric
            if key in training_curves:
                if metric == 'acc' or metric == 'f1' or metric == 'auc':
                    ax.plot(epochs, [item.detach().cpu() for item in training_curves[key]])
                else:
                    ax.plot(epochs, training_curves[key])
        ax.set_xlabel('epoch')
        ax.legend(labels=phases)

def classify_predictions(model, device, dataloader):
    model.eval()   # Set model to evaluate mode
    all_labels = torch.tensor([]).to(device)
    all_scores = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = torch.where(outputs < 0.5, 0, 1).view(-1)
        scores = outputs.view(-1)
        all_labels = torch.cat((all_labels, labels), 0)
        all_scores = torch.cat((all_scores, scores), 0)
        all_preds = torch.cat((all_preds, preds), 0).view(-1)
    return all_preds.detach().cpu(), all_labels.detach().cpu(), all_scores.detach().cpu()

def plot_cm(model, device, dataloaders, phase='test'):
    class_labels = [0, 1]
    preds, labels, scores = classify_predictions(model, device, dataloaders[phase])
    
    cm = metrics.confusion_matrix(labels, preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    ax = disp.plot().ax_
    ax.set_title('Confusion Matrix -- counts')


plot_training_curves(training_curves, phases=['train', 'val', 'test'])
plt.show()
