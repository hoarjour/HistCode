import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)  # W

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class CLAM_SB_Reg(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=10,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, freeze=False):
        super(CLAM_SB_Reg, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)
        if freeze:
            self.attention_net.requires_grad_(False)

    def relocate(self, device):
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    def forward(self, h, attention_only=False):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)
        logits = self.classifiers(M)
        return logits


class CLAM_SB_Reg_NN_Pool(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=1,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, freeze=False, N=100):
        super(CLAM_SB_Reg_NN_Pool, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1] * 2, n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.N = N

        initialize_weights(self)
        if freeze:
            self.attention_net.requires_grad_(False)

    def relocate(self, device):
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    def forward(self, h, attention_only=False):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N

        if h.shape[0] > self.N * 2:
            idxs = torch.argsort(A[0])
            low_n_idxs = idxs[:self.N]
            high_n_idxs = idxs[-self.N:]

            low_n = h[low_n_idxs].mean(axis=0)
            high_n = h[high_n_idxs].mean(axis=0)

            M = torch.cat([low_n, high_n])
            M = torch.unsqueeze(M, 0)
        else:
            M = torch.mm(A, h)
            M = torch.concat([M[0], M[0]])
            M = torch.unsqueeze(M, 0)
        logits = self.classifiers(M)
        return logits


class CLAM_SB_Pool(nn.Module):

    def __init__(self, size_arg="small", n_classes=2):
        super(CLAM_SB_Pool, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)

        self.n_classes = n_classes

        initialize_weights(self)

    def forward(self, h):
        h = self.fc(h)
        M = h.mean(axis=0)
        M = torch.unsqueeze(M, 0)
        logits = self.classifiers(M)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat


class EarlyStop:
    """Stops training if the validation loss does not decrease within a given
    range of epochs.
    Compared the lowest recorded loss to the latest loss. If the new loss is
    larger than the lowest recorded for a given set of epochs (persistence)
    the model is not improving and we can terminate it.
    """

    def __init__(self, persistence, path):
        """
        Arguments:
            persistence (int): Maximum checks
            path (str): Path to save checkpoints
        """
        self.lowest_loss = np.Inf
        self.counter = 0
        self.persistence = persistence
        self.early_stop = False
        self.path = path

    def __call__(self, model, loss):
        """
        Arguments:
            model (pytorch object): Model to save state from
            loss (float): Running loss (validation or test) for comparison
        """
        if self.lowest_loss > loss:
            # If the new loss < lowest loss the model is still improving
            self.lowest_loss = loss
            self.counter = 0  # Start counting over
        elif self.lowest_loss <= loss:
            if self.counter == 1:
                torch.save(model.state_dict(), self.path)
            # If the existing lowest loss is smaller than the new loss the model has not improved
            self.counter += 1

        self.early_stop = self.counter == self.persistence
