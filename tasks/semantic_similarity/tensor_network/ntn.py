import torch
from torch.optim import *
import warnings

from sentence_encoders.bidir_LSTM import Encoder
from constants import DEVICE


class NTN(torch.nn.Module):
    """
    Neural Tensor Network Model
    """

    def __init__(self, weights_matrix: torch.tensor, k: int, trainable: bool = True, encoder: torch.nn.module = None):

        super(NTN, self).__init__()
        self.k = k
        print(f"--k = {self.k}---------")
        self.trainable = trainable
        self.num_embeddings, self.embedding_dim = weights_matrix.size()
        self.emb_layer = torch.nn.Embedding(self.num_embeddings, self.embedding_dim, padding_idx=0)
        self.emb_layer.load_state_dict({'weight': weights_matrix})
        if not self.trainable:
            self.emb_layer.weight.requires_grad = False

        if encoder is None:
            self.encoder = Encoder(depth=self.embedding_dim, reproject_words_dimension=self.embedding_dim)
        else:
            self.encoder = encoder

        self.bilinear_size = RNN_HIDDEN_SIZE
        if self.encoder.bidirectional:
            self.bilinear_size = 2 * self.bilinear_size

        self.W = torch.nn.Bilinear(self.bilinear_size, self.bilinear_size, k)
        self.V = torch.nn.Linear(2 * self.bilinear_size, k)
        self.output = torch.nn.Linear(k, 2, bias=False)
        self.loss_function = torch.nn.CrossEntropyLoss()

        self.to(DEVICE)

    def forward(self, batch_1, batch_2):
        batch_1 = self.emb_layer(batch_1)
        batch_2 = self.emb_layer(batch_2)

        #         batch_1 = torch.nn.utils.rnn.pack_padded_sequence(batch_1, lengths_1, batch_first=True)
        #         batch_2 = torch.nn.utils.rnn.pack_padded_sequence(batch_2, lengths_2, batch_first=True)

        encoded_1 = self.encoder(batch_1)
        encoded_2 = self.encoder(batch_2)

        bilinear_out = self.W(encoded_1, encoded_2)
        v_out = self.V(torch.cat((encoded_1, encoded_1), -1))
        tensor_layer_out = torch.add(bilinear_out, v_out)
        result = self.output(tensor_layer_out)
        return result

    def _calculate_loss(self, scores, labels):
        return self.loss_function(scores, labels)

    def forward_loss(self, batch_1, batch_2, labels):

        scores = self.forward(batch_1, batch_2)
        return self._calculate_single_label_loss(scores, labels)

    def _obtain_labels(self, scores):
        return [self._get_single_label(s) for s in scores]

    def _get_single_label(self, label_scores):
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        conf, idx = torch.max(softmax, 0)
        label = idx.item()
        return tuple(label, conf)

    def _calculate_single_label_loss(self, label_scores, labels):
        return self.loss_function(label_scores, labels)

    def predict(self, sentences_1, sentences_2, mini_batch_size):

        sentences_1 = torch.from_numpy(sentences_1).float().to(DEVICE)
        sentences_2 = torch.from_numpy(sentences_2).float().to(DEVICE)
        batches_1 = [
            sentences_1[x: x + mini_batch_size]
            for x in range(0, len(sentences_1), mini_batch_size)
        ]

        batches_2 = [
            sentences_2[x: x + mini_batch_size]
            for x in range(0, len(sentences_2), mini_batch_size)
        ]

        results = []
        for batch_1, batch_2 in zip(batches_1, batches_2):
            scores = self.forward(batch_1, batch_2)
            probs = list(map(lambda x: self._get_single_label(x),scores))
            results.extend(probs)
        return results

    def save(self, model_file):

        model_state = {
            "state_dict": self.state_dict(),
            "k": self.k,
            "embeddings": self.emb_layer,
            "encoder": self.encoder,
        }
        torch.save(model_state, str(model_file), pickle_protocol=4)

    def save_checkpoint(self, model_file, optimizer_state, epoch, loss):
        model_state = {
            "state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "epoch": epoch,
            "loss": loss,
            "k": self.k,
            "embeddings": self.emb_layer,
            "encoder": self.encoder,
        }
        torch.save(model_state, str(model_file), pickle_protocol=4)

    @classmethod
    def load_from_file(cls, model_file):
        """
        Loads the model from the given file.
        :param model_file: the model file
        :return: the loaded  classifier model
        """
        state = NTN._load_state(model_file)
        embeddings = state["embeddings"]
        encoder = state["encoder"]
        model = NTN(embeddings.weight, state["k"], trainable=True, encoder=encoder)
        model.load_state_dict(state["state_dict"])
        model.eval()
        model.to(DEVICE)

        return model

    @classmethod
    def _load_state(cls, model_file):
        # ATTENTION: suppressing torch serialization warnings. This needs to be taken out once we sort out recursive
        # serialization of torch objects
        # https://docs.python.org/3/library/warnings.html#temporarily-suppressing-warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround by https://github.com/highway11git to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = load_big_file(str(model_file))
            state = torch.load(f, map_location=torch.device("cuda:0"))
            return state

    @classmethod
    def load_checkpoint(cls, model_file):
        state = NTN._load_state(model_file)
        model = NTN.load_from_file(model_file)

        epoch = state["epoch"] if "epoch" in state else None
        loss = state["loss"] if "loss" in state else None
        optimizer_state_dict = (
            state["optimizer_state_dict"] if "optimizer_state_dict" in state else None
        )
        k = state["k"] if "k" in state else None
        return {
            "model": model,
            "optimizer_state_dict": optimizer_state_dict,
            "epoch": epoch,
            "loss": loss,
            "k": k,
            "embeddings": state["embeddings"],
            "encoder": state["encoder"],
        }