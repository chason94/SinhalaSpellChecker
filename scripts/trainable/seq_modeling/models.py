

# pip install allennlp
# pip install torch


# from tqdm import tqdm, tqdm_notebook
# import os, sys
# import numpy as np
# import re
# import time
# from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


#################################################
# CharCNNWordLSTMModel(CharCNNModel)
#################################################

class CharCNNModel(nn.Module):
    def __init__(self, nembs, embdim, padding_idx, filterlens, nfilters):
        super(CharCNNModel, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(nembs, embdim, padding_idx=padding_idx)
        #torch.nn.init.normal_(self.embeddings.weight.data, std=1.0)
        self.embeddings.weight.requires_grad = True

        # Unsqueeze [BS, MAXSEQ, EMDDIM] as [BS, 1, MAXSEQ, EMDDIM] and send as input
        self.convmodule = nn.ModuleList()
        for length,n in zip(filterlens, nfilters):
            self.convmodule.append(
                nn.Sequential(
                    nn.Conv2d(1, n, (length,embdim), padding=(length-1,0), dilation=1, bias=True, padding_mode='zeros'),
                    nn.ReLU()
                )
            )
        # each conv outputs [BS, nfilters, MAXSEQ, 1]
    def forward(self, batch_tensor):

        batch_size = len(batch_tensor)

        # [BS, max_seq_len]->[BS, max_seq_len, emb_dim]
        embs = self.embeddings(batch_tensor)
        
        # [BS, max_seq_len, emb_dim]->[BS, 1, max_seq_len, emb_dim]
        embs_unsqueezed = torch.unsqueeze(embs,dim=1)

        # [BS, 1, max_seq_len, emb_dim]->[BS, out_channels, max_seq_len, 1]->[BS, out_channels, max_seq_len]
        conv_outputs = [conv(embs_unsqueezed).squeeze(3) for conv in self.convmodule]

        # [BS, out_channels, max_seq_len]->[BS, out_channels]
        maxpool_conv_outputs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outputs]

        # cat( [BS, out_channels] )->[BS, sum(nfilters)]
        source_encodings = torch.cat(maxpool_conv_outputs,dim=1)
        return source_encodings

class CharCNNWordLSTMModel(nn.Module):
    def __init__(self, nchars, char_emb_dim, char_padding_idx, padding_idx, output_dim):
        super(CharCNNWordLSTMModel,self).__init__()

        # cnn module
        # takes in a list[pad_sequence] with each pad_sequence of dim: [BS][nwords,max_nchars]
        # runs a for loop to obtain list[tensor] with each tensor of dim: [BS][nwords,sum(nfilters)]
        # then use rnn.pad_sequence(.) to obtain the dim: [BS, max_nwords, sum(nfilters)]
        nfilters, filtersizes = [50,100,100,100], [2,3,4,5]
        self.cnnmodule = CharCNNModel(nchars, char_emb_dim, char_padding_idx, filtersizes, nfilters)
        self.cnnmodule_outdim = sum(nfilters)

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 650, 2
        self.lstmmodule = nn.LSTM(self.cnnmodule_outdim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.3, bidirectional=bidirectional)
        self.lstmmodule_outdim = hidden_size*2 if bidirectional else hidden_size

        # output module
        assert output_dim>0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.lstmmodule_outdim,output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=padding_idx)
    def forward(self, 
                batch_idxs: "list[pad_sequence]", 
                batch_lengths: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk = 1):

        batch_size = len(batch_idxs)

        # cnn
        cnn_encodings = [self.cnnmodule(pad_sequence_) for pad_sequence_ in batch_idxs]
        cnn_encodings = pad_sequence(cnn_encodings,batch_first=True,padding_value=0)

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = cnn_encodings
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings,aux_word_embs),dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
        intermediate_encodings = pack_padded_sequence(intermediate_encodings,batch_lengths,
                                                      batch_first=True,enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

        # dense
        # [BS,max_nwords,self.lstmmodule_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(lstm_encodings))

        # loss
        if targets is not None:
            assert len(targets)==batch_size                # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)               # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted,targets)
        
        # eval preds
        if not self.training:
            probs = F.softmax(logits,dim=-1)            # [BS,max_nwords,output_dim]
            if topk>1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True, sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk==1:
                topk_inds = torch.argmax(probs,dim=-1)   # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions

            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss




#################################################
# CharLSTMWordLSTMModel(CharLSTMModel)
#################################################

class CharLSTMModel(nn.Module):
    def __init__(self, nembs, embdim, padding_idx, hidden_size, num_layers, bidirectional, output_combination):
        super(CharLSTMModel, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(nembs, embdim, padding_idx=padding_idx)
        #torch.nn.init.normal_(self.embeddings.weight.data, std=1.0)
        self.embeddings.weight.requires_grad = True

        # lstm module
        # expected input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        self.lstmmodule = nn.LSTM(embdim, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=bidirectional)
        self.lstmmodule_outdim = hidden_size*2 if bidirectional else hidden_size

        # output
        assert output_combination in ["end","max","mean"], print('invalid output_combination; required one of {"end","max","mean"}')
        self.output_combination = output_combination

    def forward(self, batch_tensor, batch_lengths):

        batch_size = len(batch_tensor)
        # print("************ stage 2")

        # [BS, max_seq_len]->[BS, max_seq_len, emb_dim]
        embs = self.embeddings(batch_tensor)
        
        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
        embs_packed = pack_padded_sequence(embs,batch_lengths, batch_first=True,enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(embs_packed)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

        #[BS, max_seq_len, self.lstmmodule_outdim]->[BS, self.lstmmodule_outdim]
        if self.output_combination=="end":
            last_seq_idxs = torch.LongTensor([x-1 for x in batch_lengths])
            source_encodings = lstm_encodings[range(lstm_encodings.shape[0]), last_seq_idxs, :]
        elif self.output_combination=="max":
            source_encodings, _ = torch.max(lstm_encodings, dim=1)
        elif self.output_combination=="mean":
            sum_ = torch.sum(lstm_encodings, dim=1)
            lens_ = batch_lengths.unsqueeze(dim=1).expand(batch_size,self.lstmmodule_outdim)
            assert sum_.size()==lens_.size()
            source_encodings = torch.div(sum_,lens_)
        else:
            raise NotImplementedError

        return source_encodings

class CharLSTMWordLSTMModel(nn.Module):
    def __init__(self, nchars, char_emb_dim, char_padding_idx, padding_idx, output_dim):
        super(CharLSTMWordLSTMModel,self).__init__()

        # charlstm module
        # takes in a list[pad_sequence] with each pad_sequence of dim: [BS][nwords,max_nchars]
        # runs a for loop to obtain list[tensor] with each tensor of dim: [BS][nwords,charlstm_outputdim]
        # then use rnn.pad_sequence(.) to obtain the dim: [BS, max_nwords, charlstm_outputdim]
        hidden_size, num_layers, bidirectional, output_combination = 256, 1, True, "end"
        self.charlstmmodule = CharLSTMModel(nchars, char_emb_dim, char_padding_idx, hidden_size, num_layers, bidirectional, output_combination)
        self.charlstmmodule_outdim = self.charlstmmodule.lstmmodule_outdim

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 650, 2
        self.lstmmodule = nn.LSTM(self.charlstmmodule_outdim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.3, bidirectional=bidirectional)
        self.lstmmodule_outdim = hidden_size*2 if bidirectional else hidden_size

        # output module
        assert output_dim>0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.lstmmodule_outdim,output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=padding_idx)
    def forward(self, 
                batch_idxs: "list[pad_sequence]",
                batch_char_lengths: "list[tensor]",
                batch_lengths: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk = 1):

        batch_size = len(batch_idxs)
        # print("************ stage 1")

        # charlstm
        charlstm_encodings = [self.charlstmmodule(pad_sequence_,lens) for pad_sequence_,lens in zip(batch_idxs,batch_char_lengths)]
        charlstm_encodings = pad_sequence(charlstm_encodings,batch_first=True,padding_value=0)

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = charlstm_encodings
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings,aux_word_embs),dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
        intermediate_encodings = pack_padded_sequence(intermediate_encodings,batch_lengths,
                                                      batch_first=True,enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

        # dense
        # [BS,max_nwords,self.lstmmodule_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(lstm_encodings))

        # loss
        if targets is not None:
            assert len(targets)==batch_size                # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)               # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted,targets)
        
        # eval preds
        if not self.training:
            probs = F.softmax(logits,dim=-1)            # [BS,max_nwords,output_dim]
            if topk>1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True, sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk==1:
                topk_inds = torch.argmax(probs,dim=-1)   # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions

            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss





#################################################
# SCLSTM
#################################################

class SCLSTM(nn.Module):
    def __init__(self, screp_dim, padding_idx, output_dim):
        super(SCLSTM,self).__init__()
        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 650, 2
        self.lstmmodule = nn.LSTM(screp_dim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.4, bidirectional=bidirectional) # 0.3 or 0.4
        self.lstmmodule_outdim = hidden_size*2 if bidirectional else hidden_size

        # output module
        assert output_dim>0
        self.dropout = nn.Dropout(p=0.5) # 0.4 or 0.5
        self.dense = nn.Linear(self.lstmmodule_outdim,output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=padding_idx)
    def forward(self, 
                batch_screps: "list[pad_sequence]", 
                batch_lengths: "tensor",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk = 1):

        # cnn
        batch_size = len(batch_screps)
        batch_screps = pad_sequence(batch_screps,batch_first=True,padding_value=0)

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = batch_screps
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings,aux_word_embs),dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
        intermediate_encodings = pack_padded_sequence(intermediate_encodings,batch_lengths,
                                                      batch_first=True,enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

        # dense
        # [BS,max_nwords,self.lstmmodule_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(lstm_encodings))

        # loss
        if targets is not None:
            assert len(targets)==batch_size                # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)               # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted,targets)
        
        # eval preds
        if not self.training:
            probs = F.softmax(logits,dim=-1)            # [BS,max_nwords,output_dim]
            if topk>1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True, sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk==1:
                topk_inds = torch.argmax(probs,dim=-1)   # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions
            
            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss