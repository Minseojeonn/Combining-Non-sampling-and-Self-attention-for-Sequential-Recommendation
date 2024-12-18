import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.prediction = torch.nn.Linear(args.hidden_units, 1)
        
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        #seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        return seqs
    def reg_loss(self):
        l2_reg = self.item_emb.weight.norm(2)
        return l2_reg
    def left(self, log_seqs):
        F = self.log2feats(log_seqs)  # 사용자 시퀀스 특징
        batch_size, seq_len, d = F.shape  # F: (batch_size, seq_len, d)
        FF = F.unsqueeze(2) * F.unsqueeze(3)
        FF = FF.sum(dim=1)
        E = self.item_emb.weight  # 아이템 임베딩 (num_items, d)
        EE = E.unsqueeze(2) * E.unsqueeze(1)  # E_i,k @ E_i,l -> (d, d)
        EE = EE.sum(dim=0)  # sum over items
        c = 0.001  # 가중치 스칼라 값, 예시로 사용
        cEE = c * EE 
        
        hh = torch.matmul(self.prediction.weight.t(), self.prediction.weight) 
        left = FF @ (cEE) @ hh  # 계산 순서: FF @ EE @ hh
        left = left.sum(dim=-1)
        left = left.sum(dim=-1)
        #left == (batch_size,1)
        return left, F

    def right(self, F, pos_seqs): #batch, seq, dim
        pos_seqs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))  # 아이템 임베딩 (pos_seqs)
        
        pu = F.transpose(1, 2)
        qi = pos_seqs.transpose(1, 2)
        ht = self.prediction.weight
        R_hat = ht @ (pu * qi)
        c = 0.001  # 가중치 스칼라 값
        right = (1 - c) * (R_hat**2) - (2 * R_hat)  # 수식 그대로 적용
        right = right.sum(dim=-1)
        right = right.sum(dim=-1)
        return right

    def forward(self, user_ids, log_seqs, pos_seqs):
        # 첫 번째 항 (left) 계산
        left, F = self.left(log_seqs)

        # 두 번째 항 (right) 계산
        right = self.right(F, pos_seqs)
        loss = left + right
        #loss = loss.mean()
        # 세 번째 항: 정규화
        reg_loss = self.reg_loss()
        # 최종 손실
        loss = loss + 0.0001 * reg_loss
        return loss.mean()
        

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs*final_feat
        logits = self.prediction(logits).squeeze(1).unsqueeze(0) # (U, I) #

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
