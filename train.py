# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:40:25 2021

@author: HP
"""

import argparse
from data_extraction import data_extraction
from torch.optim import Adam
from model import GCN_LPA
import numpy as np
from torch.nn.functional import nll_loss
from utils import accuracy
import time
import copy
import torch

#parser
parser = argparse.ArgumentParser(description='Node classification on Cora dataser')
parser.add_argument('-e', '--Epochs', type = int, default = 200, help = 'NUmber of Epochs')
parser.add_argument('-lr', '--Learning_Rate', type = float, default = 0.05, help = 'Learning rate')
parser.add_argument('-hi', '--Hidden', default=32, type = int, help = 'Number of hidden layers')
parser.add_argument('-dp', '--Dropout', default = 0.2, type = float, help= 'Dropout')
parser.add_argument('-wd', '--Weight_Decay', default = 1e-4, type = float, help = 'Weight Decay for optimizer')
parser.add_argument('--Lambda', type=float, default=10,help='L2 Regularizer')
args = parser.parse_args()

np.random.seed(42)
torch.manual_seed(42)
#Data paths
path = 'data/citeseer/'
dataset = 'citeseer'

#Start timestamp
t0 = time.time() 

#Loading data
features, adjacency, labels, train_idx, val_idx, test_idx = data_extraction()
# adjacency = adjacency.to(torch.int)
eye = torch.eye(7)
labels_lpa = eye[labels]
#Hyperparameters
input_dim = features.shape[1]
hidden_dim = args.Hidden
output_dim = len(np.unique(labels))
learning_rate = args.Learning_Rate
dropout_ratio = args.Dropout
decay = args.Weight_Decay
epochs = args.Epochs

#Model
model = GCN_LPA(input_dim, hidden_dim, output_dim, adjacency, dropout_ratio)

#Loss and Optimizer
optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay = decay)
 

def train():
    
    best_model_wts = copy.deepcopy(model.state_dict()) 
    best_acc = 0.
    epoch_no = 0
    
    for epoch in (range(epochs)):
        
        #Model to training mode
        model.train()
        
        # Forward pass
        # labels[0]
        output, y_hat = model(features, adjacency, labels_lpa)
        train_loss_gcn = nll_loss(output[train_idx], labels[train_idx])
        train_loss_lpa = nll_loss(y_hat, labels)
        train_loss = train_loss_gcn + args.Lambda*train_loss_lpa
        
        #Backward and optimize
        train_loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        
        #Training accuracy
        train_acc = accuracy(labels[train_idx], output[train_idx])
        
        #Model to evaluation mode
        model.eval()
        # output,_ = model(features, adjacency, labels_lpa)
        
        #Validation loss and accuracy
        val_loss = nll_loss(output[val_idx], labels[val_idx])
        val_acc = accuracy(labels[val_idx], output[val_idx])
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epoch_no = epoch
            
        print(f'Epochs :{epoch+1}/{epochs}, '
              f'Train Loss :{train_loss:.4f}, '
              f'Train_acc :{train_acc:.4f}, '
              f'Val Loss :{val_loss:.4f}, '
              f'Val acc :{val_acc:.4f}')
            
    return best_acc, epoch_no, best_model_wts
  
#Training
best_acc, epoch_no, best_model_wts = train()
model.load_state_dict(best_model_wts)

print(f'Best val acc :{best_acc}, Epoch :{epoch_no}')

# #Saving Model
# FILE = "saved_model/model1.pth"
# torch.save(model.state_dict(), FILE)
# print('Model Saved!!')
model.eval()
output,_ = model(features, adjacency, labels_lpa)
val_loss = nll_loss(output[test_idx], labels[test_idx])
val_acc = accuracy(labels[test_idx], output[test_idx])
print(val_loss, val_acc)

#End timestamp
print(f'Time Taken :{time.time() - t0}')