# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:01:37 2021

@author: HP
"""
# from torch.nn import Module, functional
# from torch.nn.parameter import Parameter
# import torch
from utils import glorot, zeros

# class GCNConv(Module):
    
#     def __init__(self, input_dim, output_dim, adj, bias=True):
#         super(GCNConv, self).__init__()
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(output_dim))
#         else:
#             self.register_parameter('bias', None)
            
#         self.reset_parameter()
        
#         self.adjacency_mask = Parameter(adj.clone()).to_dense()
    
#     def reset_parameter(self):
#         glorot(self.weights)
#         glorot(self.bias)
        
    # def forward(self, X, adj, Y):
    #     adj = adj.to_dense()
        
    #     mult1 = torch.mm(X,self.weights)
        
    #     adj = adj*self.adjacency_mask
    #     adj = functional.normalize(adj, p=1, dim=1)
        
    #     mult2 = torch.mm(adj,mult1)
        
    #     y_hat = torch.mm(adj,Y)
        
    #     if self.bias is not None:
    #         output = mult2 + self.bias
    #         return output, y_hat
    #     else:
    #         return mult2, y_hat      


import torch

from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn import functional


class GCNConv(nn.Module):
    """
    A GCN-LPA layer. Please refer to: https://arxiv.org/abs/2002.06755
    """

    def __init__(self, input_dim, output_dim, adj, bias=True):
        super(GCNConv, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameter()
        
        self.adjacency_mask = Parameter(adj.clone()).to_dense()
        
    def reset_parameter(self):
        glorot(self.weights)
        zeros(self.bias)

    def forward(self, X, adj, Y):
        adj = adj.to_dense()
        # W * x
        support = torch.mm(X, self.weights)

        adj = adj * self.adjacency_mask

        adj = functional.normalize(adj, p=1, dim=1)


        output = torch.mm(adj, support)

        y_hat = torch.mm(adj, Y)

        if self.bias is not None:
            return output + self.bias, y_hat
        else:
            return output, y_hat
        

      