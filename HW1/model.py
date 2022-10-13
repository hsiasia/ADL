from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn import functional as F


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        # 300, 512, 2, 0.1, 2
        self.lstm = nn.LSTM(input_size = self.embed_dim, hidden_size = self.hidden_size, num_layers = self.num_layers,
                            dropout = self.dropout, bidirectional = self.bidirectional, batch_first = True)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout),
                                        nn.Linear(in_features = self.hidden_size * 2, out_features = self.num_class))

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        output_dict = {}

        # batch:  {'id': ['train-11240', ..(*128).., 'train-621'], 
        #          'text': tensor([[3892, ..(*128).., 0], ..(*128).., [449, ..(*128).., 0]]), 
        #          'len': tensor([12, ..(*128).., 6]), 
        #          'intent': tensor([27, ..(*128).., 117])}
        text, intent = batch['text'], batch['intent'] # [batch_size = 128, max_len = 128] # [batch_size = 128]

        # attention part
        # # >0 -> trueflase -> 01
        # mask = (text.gt(0)).float() # [batch_size, max_len]

        # text_embedded: [[[-0.1130, ..(*300)..,  0.1077], ..(*128).., [-0.4820, ..(*300).., -0.1723]],
        #                 ..(*128)..,
        #                 [[-0.7481, ..(*300)..,  0.8008], ..(*128).., [-0.4820, ..(*300).., -0.1723]]],
        text_embedded = self.embed(text) # [batch_size = 128, max_len = 128, embed_dim = 300]

        # PackedSequence(data=tensor([[-0.1038, ..(*300).., 0.1792], ..(*size_after_packed).., [-0.3201, ..(*300).., 0.0997]]), 
        #                batch_sizes=tensor([128, ..(*mini_batch_max_len).., 1]), 
        #                sorted_indices=tensor([12, ..(*128).., 78]), 
        #                unsorted_indices=tensor([2, ..(*128).., 76]))
        packed_text = nn.utils.rnn.pack_padded_sequence(text_embedded, batch['len'], batch_first=True) # [size_after_packed, embed_dim = 300]
        self.lstm.flatten_parameters()

        # PackedSequence(data=tensor([[-2.6123e-03, ..(*1024).., -7.1546e-03], ..(*size_after_packed).., [-5.4017e-03, ..(*1024).., -5.2137e-03]]), 
        #                batch_sizes=as packed, 
        #                sorted_indices=as packed, 
        #                unsorted_indices=as packed)
        # hidden: tensor([[[5.1340e-02, ..(*512).., -2.1012e-02], ..(*128).., [4.0590e-02, ..(*512).., -9.3412e-02]],
        #                 ..(*4)..
        #                 [[-6.5971e-02, ..(*512).., -6.0133e-04], ..(*128).., [-5.4275e-02, ..(*512).., -7.6376e-03]]]
        # cell: as hidden
        output_packed, (hidden, cell) = self.lstm(packed_text) # [size_after_packed, output_cells = hidden_size*bidirectinal] # [all_layers = num_layers*bidirectinal, batch_size = 128, hidden_size = 512]

        # output: [[[-0.0214, ..(*1024).., 0.0040], ..(*mini_batch_max_len).., [ 0.0000, ..(*1024).., 0.0000]],
        #          ..(*128)..,
        #          [[-0.0164, ..(*1024).., 0.0133], ..(*mini_batch_max_len).., [ 0.0000, ..(*1024).., 0.0000]]],
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True) # [batch_size = 128, mini_batch_max_len, output_cells = hidden_size*bidirectinal]
        
        # -1 the last, -2 the second last
        if self.bidirectional:
            hidden = torch.cat((hidden[-1], hidden[-2]), axis=-1) # [batch_size = 128, hidden_size*2 = 1024]
            # hidden = hidden[-1] + hidden[-2]

        # the result of model save as list, original classifier
        pred_logits = self.classifier(hidden)
        # output_dict['pred_logits'] = [pred_logits]

        # # attention start
        # hidden = hidden.squeeze(0)
        # att_weights = torch.bmm(output, hidden.unsqueeze(2)).squeeze(2)

        # # 128 -> seq_len
        # mask = mask[:, :output.size(1)]
        # mask[mask == 0] = -1e12

        # soft_attn_weights = F.softmax(att_weights + mask[:, :output.size(1)], 1)
        # new_hidden_state = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        # pred_logits = self.classifier(new_hidden_state)
        # # output_dict['pred_logits'] = [pred_logits]
        # # attention end

        # 1 select the max and indices from each row in matrix [num_classes], True makes output shape same as pred_logits[-1]
        # select second element (label's indices)
        # reshape row to column
        # pred_logits.max(1, keepdim=True): torch.return_types.max(values=tensor([[0.0596], ..(*128).., [0.0532]],
        #                                                          indices=tensor([[122], ..(*128).., [35]]))
        # pred_logits.max(1, keepdim=True)[1]: [batch_size = 128, 1]
        # output_dict['pred_labels'] = pred_logits.max(1, keepdim=True)[1].reshape(-1) # [batch_size = 128]
        output_dict['pred_labels'] = torch.argmax(pred_logits, dim=-1) # [batch_size = 128]

        # caculate the loss
        # intent.long():　[batch_size = 128]
        # pred_logits: [batch_size = 128, num_classes = 150]
        output_dict['loss'] = F.cross_entropy(pred_logits, intent.long())

        return output_dict


class SeqTagger(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_classes: int,
    ) -> None:
        super(SeqTagger, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # model architecture
        self.embed_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # 300, 512, 2, 0.1, 2
        self.lstm = nn.LSTM(input_size = self.embed_dim, hidden_size = self.hidden_size, num_layers = self.num_layers,
                            dropout = self.dropout, bidirectional = self.bidirectional, batch_first = True)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout),
                                        nn.Linear(in_features = self.hidden_size * 2, out_features = self.hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(in_features = self.hidden_size, out_features = self.num_classes))

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def getIdx(self, tokens_len):
        # put indexi for t_len times, and concat
        batch_idx = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(tokens_len)])
        # put 0 to t_len, and concat
        tok_idx = torch.cat([torch.arange(0, t_len) for t_len in tokens_len])
        return batch_idx, tok_idx

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        output_dict = {}

        tokens = batch['tokens'] # [batch_size = 128, max_len = 128]

        tokens_embedded = self.embed(tokens) # [batch_size = 128, max_len = 128, embed_dim = 300]

        # all batch data delete the pad part
        # batch_first for lstm time step input (order by batch first status)
        packed_tokens = nn.utils.rnn.pack_padded_sequence(tokens_embedded, batch['len'], batch_first=True) # [size_after_packed, embed_dim = 300]
        
        self.lstm.flatten_parameters()
        # (stil order by word status)
        output_packed, _ = self.lstm(packed_tokens) # [size_after_packed, output_cells = hidden_size*bidirectinal]
        
        # all batch data with pad part (but it means mini_batch_max_len)
        # batch_first for fc input (order by batch first status)
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True) # [batch_size = 128, mini_batch_max_len, output_cells = hidden_size*bidirectinal]
        
        pred_logits = self.classifier(output) # [batch_size = 128, mini_batch_max_len, num_classes = 9]

        # -1 select the max and indices from each row in matrix [mini_batch_max_len*num_classes], True makes output shape same as pred_logits
        # select second element (label's indices)
        # output_dict['pred_logits'] = pred_logits.max(-1, keepdim=True)[1] # [batch_size = 128, mini_batch_max_len, 1]
        output_dict['pred_labels'] = pred_logits.max(-1, keepdim=True)[1].squeeze(2) # [batch_size = 128, mini_batch_max_len]
        
        # caculate the loss
        batch['tags'] = batch['tags'][:, :output.size(1)] # [batch_size = 128, max_len = 128] -> [batch_size = 128, mini_batch_max_len]
        # get all data, token index in batch
        idx = self.getIdx(batch['len'])
        # ? how to 2 tensor turn to 1 index
        # pred_logits[idx]: [size_after_packed, num_classes]
        # batch['tags'][idx]: [size_after_packed]
        output_dict['loss'] = F.cross_entropy(pred_logits[idx], batch['tags'][idx])

        return output_dict