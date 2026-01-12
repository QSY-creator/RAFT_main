import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class CrossAttention(nn.Module):
    def __init__(self, input_dim, d_model=512):
        super().__init__()
        self.d_model = d_model
        
        self.q_proj = nn.Linear(input_dim, d_model)
        self.k_proj = nn.Linear(input_dim, d_model)
        
    def forward(self, q, k, v):
        # q: (B, S, C)
        # k: (B, topm, S, C)
        # v: (B, topm, P, C)
        
        B, S, C = q.shape
        _, topm, _, _ = k.shape
        
        # Embed Q
        q_emb = self.q_proj(q) # (B, S, d)
        
        # Embed K
        k_emb = self.k_proj(k) # (B, topm, S, d)
        k_emb_flat = k_emb.view(B, topm * S, self.d_model) # (B, topm*S, d)
        
        # Calculate Attention
        # (B, S, d) @ (B, d, topm*S) -> (B, S, topm*S)
        attn_scores = torch.bmm(q_emb, k_emb_flat.transpose(1, 2))
        attn_scores = attn_scores / math.sqrt(self.d_model)
        
        # Softmax over all keys
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, S, topm*S)
        
        # Reshape to isolate neighbors
        attn_weights = attn_weights.view(B, S, topm, S)
        
        # Calculate neighbor weights
        # Sum over query time (S) and key time (S)
        neighbor_weights = attn_weights.sum(dim=-1).sum(dim=1) # (B, topm)
        
        # Normalize
        neighbor_weights = F.softmax(neighbor_weights, dim=1)
        
        # Aggregate V
        # (B, topm, 1, 1) * (B, topm, P, C) -> (B, topm, P, C) -> Sum dim 1 -> (B, P, C)
        output = torch.sum(neighbor_weights.unsqueeze(-1).unsqueeze(-1) * v, dim=1)
        
        return output

class RetrievalTool(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
        d_model=512,
    ):
        super().__init__()
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm
        
        self.with_dec = with_dec
        self.return_key = return_key

        # Linear layer to replace Pearson correlation
        self.similarity_linear = nn.Linear(seq_len * channels, seq_len * channels)
        
        # Cross Attention Module
        self.cross_attention = CrossAttention(channels, d_model=d_model)
        
    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            
            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])
            
        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        
        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            
            mg.append(cur)
#             data_all = data_all - cur
            
        mg = torch.stack(mg, dim=0) # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None
            
        offset = torch.stack(offset, dim=0)
            
        return mg, offset
    
    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape
        
        # bx = key - torch.mean(key, dim=2, keepdim=True)
        bx = self.similarity_linear(key)
        
        iters = math.ceil(train_len / in_bsz)
        
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            # ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            ax = self.similarity_linear(cur_data)
            
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2)
        
        return sim
        
    def retrieve(self, x, index, train=True):
        index = index.to(x.device)
        
        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)
        
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        # 1. Calculate Similarity and find Top-M
        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
            x_mg.flatten(start_dim=2), # G, B, S * C
        ) # G, B, T
            
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
            
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
            
            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

        sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T
        topm_index = torch.topk(sim, self.topm, dim=1).indices # (G*B, topm)
        
        # 2. Cross Attention Fusion
        pred_list = []
        
        # Reshape topm_index to (G, B, topm)
        topm_index = topm_index.reshape(self.n_period, bsz, self.topm)
        
        for i in range(self.n_period):
            # For each granularity
            
            # Prepare Data
            # x_mg[i]: (B, S, C) -> Query
            q = x_mg[i]
            
            # Indices for this granularity: (B, topm)
            indices = topm_index[i] # (B, topm)
            
            # Gather K (History) and V (Future)
            # train_data_all_mg[i]: (T, S, C)
            # y_data_all_mg[i]: (T, P, C)
            
            # We need to gather (B, topm, S, C) from (T, S, C)
            # train_data_all_mg is on CPU usually (from prepare_dataset)
            # But we should move relevant parts to GPU
            
            k_source = self.train_data_all_mg[i].to(x.device) # (T, S, C)
            v_source = self.y_data_all_mg[i].to(x.device) # (T, P, C)
            
            # Expand indices for gather? 
            # indices is (B, topm)
            # We want k: (B, topm, S, C)
            # k = k_source[indices] works if indices matches first dim.
            
            k = k_source[indices] # (B, topm, S, C)
            v = v_source[indices] # (B, topm, P, C)
            
            # Apply Cross Attention
            # q: (B, S, C)
            # k: (B, topm, S, C)
            # v: (B, topm, P, C)
            # output: (B, P, C)
            out = self.cross_attention(q, k, v)
            
            pred_list.append(out)
            
        # Stack results: (G, B, P, C)
        pred_from_retrieval = torch.stack(pred_list, dim=0)
        
        return pred_from_retrieval
    
    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        assert(self.train_data_all_mg != None)
        
        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        
        retrievals = []
        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                pred_from_retrieval = self.retrieve(batch_x.float().to(device), index, train=train)
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                
        retrievals = torch.cat(retrievals, dim=1)
        
        return retrievals
