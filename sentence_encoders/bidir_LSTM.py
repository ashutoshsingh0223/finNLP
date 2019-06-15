import torch
from torch.optim import *

from sentence_encoders.constants import RNN_HIDDEN_SIZE, DEPTH, NUMBER_OF_LAYERS, BIDIRECTIONAL
from constants import DEVICE


class Encoder(torch.nn.Module):
    def __init__(self,
                 reproject_words: bool = True,
                 depth: int = DEPTH,
                 reproject_words_dimension: int = DEPTH,
                 bidirectional: bool = BIDIRECTIONAL):

        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.reproject_words = reproject_words

        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.rnn = torch.nn.LSTM(input_size=depth,
                                 hidden_size=RNN_HIDDEN_SIZE,
                                 num_layers=NUMBER_OF_LAYERS,
                                 batch_first=True,
                                 bidirectional=self.bidirectional)

        self.word_reprojection_map = torch.nn.Linear(
            depth, self.embeddings_dimension
        )

        self._init_weights()
        self.to(DEVICE)

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

    def forward(self, sentence_tensor):

        batch_size = sentence_tensor.shape[0]
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)
        _, (_, final_state) = self.rnn(sentence_tensor)

        if self.bidirectional:
            final_state = final_state.view(NUMBER_OF_LAYERS, 2, batch_size, RNN_HIDDEN_SIZE)
            final_state = final_state[-1]
            h_1, h_2 = final_state[0], final_state[1]
            final_state = torch.cat([h_1, h_2], 1)
        else:
            final_state = final_state.view(NUMBER_OF_LAYERS, 1, batch_size, RNN_HIDDEN_SIZE)
            final_state = final_state[-1][0]
        return final_state
