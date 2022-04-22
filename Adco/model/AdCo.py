# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torchvision.models as models
from networks.resnet_custom import resnet50_baseline


class AdCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, args, dim=128, m=0.999, T=0.07, mlp=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(AdCo, self).__init__()
        self.args=args
        self.m = m
        self.T = T
        self.T_m = args.mem_t
        self.sym = args.sym
        self.multi_crop = args.multi_crop
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50_baseline(pretrained=True)
        self.encoder_k = resnet50_baseline(pretrained=True)
        self.encoder_q.requires_grad_(True)
        self.encoder_k.requires_grad_(True)
        self.encoder_q.fc = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, args.moco_dim))
        self.encoder_k.fc = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, args.moco_dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.K = args.cluster


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _batch_shuffle(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size).to(self.args.device)
        idx_unshuffle = torch.argsort(idx_shuffle).to(self.args.device)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if self.multi_crop:
            q_list = []
            for k, im_q in enumerate(im_q):  # weak forward
                q = self.encoder_q(im_q)  # queries: NxC
                q = nn.functional.normalize(q, dim=1)
                # q = self._batch_unshuffle_ddp(q, idx_unshuffle)
                q_list.append(q)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                # Do not use ddp
                im_k, idx_unshuffle = self._batch_shuffle(im_k)

                k = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                # Do not use ddp
                k = self._batch_unshuffle(k, idx_unshuffle)

                k = k.detach()
            return q_list, k
        elif not self.sym:
            q = self.encoder_q(im_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            # compute key features
            with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                # Do not use ddp
                im_k, idx_unshuffle = self._batch_shuffle(im_k)

                k = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                
                # undo shuffle
                # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                # Do not use ddp
                k = self._batch_unshuffle(k, idx_unshuffle)
                k = k.detach()
            return q, k
        else:
            q = self.encoder_q(im_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            q_pred=q
            k_pred = self.encoder_q(im_k)  # queries: NxC
            k_pred = nn.functional.normalize(k_pred, dim=1)
            with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
                self._momentum_update_key_encoder()  # update the key encoder

                im_q, idx_unshuffle = self._batch_shuffle_ddp(im_q)
                q = self.encoder_k(im_q)  # keys: NxC
                q = nn.functional.normalize(q, dim=1)
                q = self._batch_unshuffle_ddp(q, idx_unshuffle)
                q = q.detach()


                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                k = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                k = k.detach()

            return q_pred,k_pred,q, k

class Adversary_Negatives(nn.Module):
    def __init__(self,bank_size,dim,multi_crop=0):
        super(Adversary_Negatives, self).__init__()
        self.multi_crop = multi_crop
        self.register_buffer("W", torch.randn(dim, bank_size))
        self.register_buffer("v", torch.zeros(dim, bank_size))
    def forward(self,q, init_mem=False):
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)
        if self.multi_crop and not init_mem:
            logit_list = []
            for q_item in q:
                logit = torch.einsum('nc,ck->nk', [q_item, memory_bank])
                logit_list.append(logit)
            return memory_bank, self.W, logit_list
        else:
            logit=torch.einsum('nc,ck->nk', [q, memory_bank])
            return memory_bank, self.W, logit
    def update(self, m, lr, weight_decay, g):
        g = g + weight_decay * self.W
        self.v = m * self.v + g
        self.W = self.W - lr * self.v
    def print_weight(self):
        print(torch.sum(self.W).item())

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
