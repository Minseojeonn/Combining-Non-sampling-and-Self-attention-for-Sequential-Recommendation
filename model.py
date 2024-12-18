import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.FC1 = torch.nn.Linear(hidden_units, hidden_units)
        self.LayerNorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        self.relu = torch.nn.ReLU()
        self.FC2 = torch.nn.Linear(hidden_units, hidden_units)
        self.LayerNorm2 = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        
        
    def forward(self, inputs):
        outputs = self.FC2(self.relu(self.FC1(self.LayerNorm(inputs)))) + self.LayerNorm2(inputs)
        
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
        torch.nn.init.xavier_normal_(self.item_emb.weight)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
       
        self.forward_layers = torch.nn.ModuleList()
        self.prediction = torch.nn.Linear(args.hidden_units, 1, bias=False)
        
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))

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

            #seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        return seqs
    def reg_loss(self):
        # 정규화 손실 계산
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param)
        return l2_reg
    
    def left(self, F):
          # 사용자 시퀀스 특징
        batch_size, seq_len, d = F.shape  # F: (batch_size, seq_len, d)
        FF = F.unsqueeze(2) * F.unsqueeze(3)
        FF = FF.sum(dim=1)
        
        item_sum = torch.bmm(
            self.item_emb.weight.unsqueeze(2),
            self.item_emb.weight.unsqueeze(1),
            ).sum(dim=0)  # E_i,k @ E_i,l -> (d, d)
        
        c = 0.001  # 가중치 스칼라 값, 예시로 사용
        cEE = c * item_sum 
        
        hh = torch.matmul(self.prediction.weight.t(), self.prediction.weight) 
        left = torch.sum(FF * (cEE) * hh)  # 계산 순서: FF @ EE @ hh
        return left

    def right(self, pos_socre): #batch, seq, dim 
        c = 0.001  # 가중치 스칼라 값
        right = (1 - c) * torch.square(pos_socre) - (2 * pos_socre)  # 수식 그대로 적용
        
        return torch.sum(right)

    def forward(self, user_ids, log_seqs, pos_seqs):
        # 첫 번째 항 (left) 계산
        log_feats = self.log2feats(log_seqs)
        left = self.left(log_feats)

        pos_score = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        pos_score = self.prediction(log_feats * pos_score)
        # 두 번째 항 (right) 계산
        right = self.right(pos_score)
        loss = left + right
        #loss = loss.mean()
        # 세 번째 항: 정규화
        reg_loss = self.reg_loss()
        # 최종 손실
        loss = loss + 0.1 * reg_loss
        
        return loss
        

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        pu = final_feat
        qi = item_embs
        
        R_hat = self.prediction(pu * qi)
        #logits = item_embs*final_feat
        #logits = self.prediction(logits).squeeze(1).unsqueeze(0) # (U, I) #

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return R_hat.t() # preds # (U, I)
