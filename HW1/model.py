from typing import Dict

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
        self.embed_dim = embeddings.size(1)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        self.lstm = nn.LSTM(input_size = self.embed_dim, hidden_size = self.hidden_size, num_layers = self.num_layers,
                     dropout = self.dropout, bidirectional = self.bidirectional, batch_first = True)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(self.hidden_size * 2, self.num_class)
        # )
        self.classifier = nn.Sequential(nn.Dropout(self.dropout),
                                        nn.Linear(in_features = self.hidden_size * 2, out_features = self.hidden_size * 2),
                                        # nn.Relu(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(in_features = self.hidden_size * 2, out_features = self.num_class))

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        output_dict = {}

        # batch:  {'id': ['train-11240' ..(id*128).. 'train-621'], 
        #         'text': tensor([[3892, 5790, 4329,  ..(word*128)..,    0,    0,    0] ..(sentence*128).. [ 449, 4931, 3684,  ..(word*128)..,    0,    0,    0]]), 
        #         'len': tensor([12 ..(word_in_a_sentence*128)..  6]), 
        #         'intent': tensor([27 ..(intent2idx*128).. 117])}
        # batch['text']: torch.Size([128, 128]) / [batch_size max_word_in_a_sentence]
        # batch['intent']: torch.Size([128]) / [batch_size]
        text, intent = batch['text'], batch['intent']
        
        # text_embedded: torch.Size([128, 128, 300]) / [batch_size, max_word_in_a_sentence, embed_dim]
        # text_embedded[128][128][300]
        # text_embedded: [[[-0.1130,  0.3207, -0.5438 ..word3embd*300.. 0.1905,  0.3545,  0.1077],
        #                   ..word*128..,
        #                  [-0.4820,  0.3705,  0.3682 ..*300.. -0.0015,  0.1438, -0.1723]],
        #                 ..sentence*128..,
        #                 [[-0.7481,  0.8208,  0.0107 ..*300..  0.2947, -0.0981,  0.8008],
        #                   ..word*128..,
        #                  [-0.4820,  0.3705,  0.3682 ..*300.. -0.0015,  0.1438, -0.1723]]],
        text_embedded = self.embed(text)

        # ?
        # pack_padded_sequence to dymanicly address each batch's len
        # PackedSequence(data=tensor([[-0.1038,  0.2065, -0.1937 ..*300.. -0.2407,  0.3086,  0.1792],
        #                               ..*?..,
        #                             [-0.3201,  0.8325,  0.6346 ..*300.. -0.1604, -0.0658,  0.0997]]), 
        #               batch_sizes=tensor([128, 126, ..max batch['len'].. 2,   1]), 
        #               sorted_indices=tensor([ 12, 120 ..*128.. 24,  78]), 
        #               unsorted_indices=tensor([ 2,  90, ..*128.. 35,  76]))
        # data: torch.Size([?, 300]) gard
        packed_text = nn.utils.rnn.pack_padded_sequence(text_embedded, batch['len'], batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()

        # ?
        # PackedSequence(data=tensor([[-2.6123e-03,  3.0056e-02,  2.3325e-03,  ..1024..,  4.2309e-02, -5.2573e-02, -7.1546e-03],
        #                               ..?..,
        #                             [-5.4017e-03,  5.9926e-03,  9.0346e-03,  ...,  1.2922e-02, -2.9177e-02, -5.2137e-03]]), 
        #                batch_sizes=as packed, 
        #                sorted_indices=as packed, 
        #                unsorted_indices=as packed)
        # data: torch.Size([?, 1024]) gard
        # hidden: torch.Size([4, 128, 512]) grad
        # hidden: tensor([
        # [[ 5.1340e-02, -8.9667e-02, -4.2291e-02,  ..*512.., -3.3602e-02, -2.5190e-02, -2.1012e-02],
        #  ..*128..,
        #  [ 4.0590e-02, -6.4623e-02, -3.6845e-04,  ..., -4.8889e-02, 5.1251e-02, -9.3412e-02]],
        # [[ 1.6498e-02,  3.1234e-02, -2.2041e-02,  ..., -2.2359e-02, -5.0190e-02, -6.6134e-02],
        #  ..*128..,
        #  [-3.4370e-02,  1.9222e-02, -3.2772e-02,  ...,  1.7065e-02, -4.5621e-02, -3.4005e-02]],
        # [[-8.7334e-03,  2.8085e-02,  2.5695e-03,  ..., -2.6970e-02, 8.2791e-03, -1.3702e-02],
        #  ..*128..,
        #  [ 9.8418e-03,  2.7368e-02,  1.9901e-02,  ..., -5.4476e-02, -8.0476e-03, -2.1952e-02]],
        # [[-6.5971e-02,  1.2094e-02, -2.9507e-02,  ...,  9.6665e-03, 3.3974e-02, -6.0133e-04],
        #  ..*128..,
        #  [-5.4275e-02,  3.0708e-02, -2.2902e-02,  ...,  2.9815e-05, 3.8676e-02, -7.6376e-03]]]
        # cell: as hidden
        output_packed, (hidden, cell) = self.lstm(packed_text)

        # output: torch.Size([128, max_word_in_sentence_this_batch, 1024]) grad / [batch_size, max_word_in_sentence_this_batch, hid_dim]
        # output: [[[-0.0214, -0.0208, -0.0008 ..*1024.. -0.0178,  0.0170,  0.0040],
        #           ..*max_word_in_sentence_this_batch..,
        #           [ 0.0000,  0.0000,  0.0000 ..*1024..  0.0000,  0.0000,  0.0000]],
        #          ..*128..,
        #          [[-0.0164, -0.0189,  0.0013 ..*1024.. -0.0185,  0.0105,  0.0133],
        #           ..*max_word_in_sentence_this_batch..,
        #           [ 0.0000,  0.0000,  0.0000 ..*1024..  0.0000,  0.0000,  0.0000]]],
        # output_length: torch.Size([128]) / [max_word_in_sentence_this_batch]
        # [ 6 ..length*128.. 13]
        output, output_length = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        
        # hidden[-1]:  torch.Size([128, 512])
        # hidden[cat]:  torch.Size([128, 1024])
        # -1 the last, -2 the second last
        if self.bidirectional:
            hidden = torch.cat((hidden[-1], hidden[-2]), axis=-1) 
        else:
            hidden = hidden[-1]

        # the result of model save as list
        pred_logits = [self.classifier(hidden)]

        output_dict['pred_logits'] = pred_logits
        
        # pred_logits[-1].max(1, keepdim=True)[1]: tensor([128, 1])
        # torch.return_types.max(
        #   values=tensor([[0.0596], [0.0496] ..*128.. [0.0481], [0.0532]], grad_fn=<MaxBackward0>),
        #   indices=tensor([[122], [ 44], ..*128.. , [ 13], [ 35]]))

        # list[-1] = list[0] (the first element in list)
        # 1. max(1, keepdim=True) 1 means select the max and index, True makes output shape same as pred_logits[-1]
        # 2. [1] and select second element (label's index)
        # 3. reshape(-1) row to column
        # output_dict['pred_labels']: torch.Size([128])
        output_dict['pred_labels'] = pred_logits[-1].max(1, keepdim=True)[1].reshape(-1)

        # caculate the loss
        # intent.long():　torch.Size([128])
        # pred_logits[-1]: torch.Size([128, 150]) grad
        output_dict['loss'] = F.cross_entropy(pred_logits[-1], intent.long())

        return output_dict
        raise NotImplementedError


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
