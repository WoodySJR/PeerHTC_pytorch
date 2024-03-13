# Hierarchy-Aware Global Model for Hierarchical Text Classification

import torch, pandas as pd, numpy as np, math
from torch import nn
from pytorch_pretrained_bert import BertModel
import warnings
warnings.filterwarnings("ignore")
device = "cuda:0"

# define the network
class PeerHTC(nn.Module): 
    
    def __init__(self, hidden_dim, num_labels, num_1, num_2, num_3, 
                 mask, fre12, fre23):
        super(PeerHTC, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_1 = num_1
        self.num_2 = num_2
        self.num_3 = num_3
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.w12 = mask[0:num_1, num_1:(num_1+num_2)].to(device)
        self.w21 = self.w12.T
        self.w23 = mask[num_1:(num_1+num_2), (num_1+num_2):num_labels].to(device)
        self.w32 = self.w23.T
        self.fre12 = fre12.to(device)
        self.fre23 = fre23.to(device)
        
        # Bert for text and label_desc encoding
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_transform = nn.Linear(768, self.hidden_dim)
        
        # trainable label encoding
        self.label_encoding = nn.Parameter(torch.randn(num_labels, hidden_dim), requires_grad=True)
        self.mix = nn.Linear(3*self.hidden_dim, self.hidden_dim)
        
        # Tree-LSTM params
        ## top-down
        self.wi1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ui1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wf1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.uf1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wo1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.uo1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wu1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.uu1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        ## bottom-up
        self.wi2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ui2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wf2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.uf2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wo2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.uo2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wu2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.uu2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # GCN
        self.A = nn.Parameter(torch.randn(num_labels, num_labels), requires_grad=True)
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim), requires_grad=True)
        
        # output layer
        self.output = nn.Linear(self.hidden_dim*self.num_labels, self.num_labels)
        
    def forward(self, inputs): 
        # structure encoder
        label_encoding = self.label_encoding
        v1 = label_encoding[0:self.num_1,:] 
        v2 = label_encoding[self.num_1:(self.num_1+self.num_2),:]
        v3 = label_encoding[(self.num_1+self.num_2):self.num_labels,:]
        ## top-down manner
        ### 1
        i11 = self.sigmoid(self.wi1(v1))
        o11 = self.sigmoid(self.wo1(v1))
        u11 = self.tanh(self.wu1(v1))
        c11 = i11*u11
        h11 = o11*self.tanh(c11)
        ### 1->2
        h21_tilde = self.w21@h11
        i21 = self.sigmoid(self.wi1(v2) + self.ui1(h21_tilde))
        f21 = self.sigmoid(self.wf1(v2) + self.uf1(self.w21@h11))
        o21 = self.sigmoid(self.wo1(v2) + self.uo1(h21_tilde))
        u21 = self.tanh(self.wu1(v2) + self.uu1(h21_tilde))
        c21 = i21*u21 + f21*(self.w21@c11)
        h21 = o21*self.tanh(c21)
        ### 2->3
        h31_tilde = self.w32@h21
        i31 = self.sigmoid(self.wi1(v3) + self.ui1(h31_tilde))
        f31 = self.sigmoid(self.wf1(v3) + self.uf1(self.w32@h21))
        o31 = self.sigmoid(self.wo1(v3) + self.uo1(h31_tilde))
        u31 = self.tanh(self.wu1(v3) + self.uu1(h31_tilde))
        c31 = i31*u31 + f31*(self.w32@c21)
        h31 = o31*self.tanh(c31)
        ## bottom-up manner
        ### 3
        i32 = self.sigmoid(self.wi2(v3))
        o32 = self.sigmoid(self.wo2(v3))
        u32 = self.tanh(self.wu2(v3))
        c32 = i32*u32
        h32 = o32*self.tanh(c32)
        ### 3->2
        h22_tilde = self.fre23@h32
        i22 = self.sigmoid(self.wi2(v2) + self.ui2(h22_tilde))
        f22 = self.sigmoid(self.wf2(self.w32@v2) + self.uf2(h32))
        o22 = self.sigmoid(self.wo2(v2) + self.uo2(h22_tilde))
        u22 = self.tanh(self.wu2(v2) + self.uu2(h22_tilde))
        c22 = i22*u22 + self.w23@(f22*c32)
        h22 = o22*self.tanh(c22)
        ### 2->1
        h12_tilde = self.fre12@h22
        i12 = self.sigmoid(self.wi2(v1) + self.ui2(h12_tilde))
        f12 = self.sigmoid(self.wf2(self.w21@v1) + self.uf2(h22))
        o12 = self.sigmoid(self.wo2(v1) + self.uo2(h12_tilde))
        u12 = self.tanh(self.wu2(v1) + self.uu2(h12_tilde))
        c12 = i12*u12 + self.w12@(f12*c22)
        h12 = o12*self.tanh(c12)
        
        h1 = torch.cat((h11,h12), dim=1)
        h2 = torch.cat((h21,h22), dim=1)
        h3 = torch.cat((h31,h32), dim=1)
        label_encoding = torch.cat((h1,h2,h3), dim=0)
        temp = self.relu((self.A@torch.cat((v1,v2,v3), dim=0))@self.W)
        label_encoding = self.mix(torch.cat((label_encoding, temp), dim=1))
        
        # text encoding
        text_encoding, _ = self.text_encoder(input_ids = inputs.long(), 
                                             attention_mask = (inputs>0).long().to(device))
        text_encoding = self.tanh(self.text_transform(text_encoding[-1]))
        
        # multi-label attention
        att_weights = torch.bmm(label_encoding.unsqueeze(0).repeat(inputs.shape[0],1,1), 
                                text_encoding.permute(0,2,1)) 
        pad_mask = torch.zeros_like(inputs).to(device).float()
        pad_mask[inputs==0] = -1*math.inf
        pad_mask[inputs==101] = -1*math.inf
        pad_mask[inputs==102] = -1*math.inf
        att_weights = torch.softmax(att_weights+pad_mask.unsqueeze(1).repeat(1,self.num_labels,1), dim=2)
        features = torch.bmm(att_weights, text_encoding).reshape(inputs.shape[0], -1)
        output = self.sigmoid(self.output(features))
        
        return output