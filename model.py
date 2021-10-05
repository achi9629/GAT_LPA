# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:01:45 2021

@author: HP
"""
from torch.nn import Module
from layers import GCNConv
from torch.nn.functional import dropout, log_softmax, relu

class GCN_LPA(Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, adj, bias=True, dropout_ratio = 0.5):
        super(GCN_LPA, self).__init__()
        
        self.layer1 = GCNConv(input_dim, hidden_dim, adj)
        self.layer5 = GCNConv(hidden_dim, output_dim, adj)
        self.dropout_ratio = .5
        
    def forward(self, X, adj, Y):
        
        output, Y_hat = self.layer1(X, adj, Y)
        output = relu(output)
        output = dropout(output, self.dropout_ratio, training = self.training)
        
        output, Y_hat = self.layer5(output, adj, Y_hat)
        output = log_softmax(output, dim=1)
        Y_hat = log_softmax(Y_hat,dim=1)
        return output, Y_hat