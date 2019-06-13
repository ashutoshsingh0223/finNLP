import argparse
import os
import pickle
import numpy as np
# import pandas as pd
# from collections import Counter, defaultdict
# import gensim.downloader as api
# from gensim.utils import save_as_line_sentence
# from gensim.models.word2vec import Word2Vec
import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import operator

from typing import List, Dict

# make directory to store model files
os.mkdir("model_dir_wiki_glove")


# class to save w2v model after each epoch.
class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        list_of_existing_files = os.listdir(".")
        output_path = 'model_dir/{}_epoch{}.model'.format(self.path_prefix, self.epoch)
        try:
            model.save(output_path)
        except:
            model.wv.save_word2vec_format('model_dir/model_{}.bin'.format(self.epoch), binary=True)
        print("number of epochs completed = {}".format(self.epoch))
        self.epoch += 1
        list_of_total_files = os.listdir(".")
        # for file_name in list_of_total_files:
        #   if file_name not in list_of_existing_files:
        #       os.system("gsutil cp {} gs://freebase_haptik/".format(file_name))


saver = EpochSaver("epoch_for_finetuning")


# Load glove vectors
def load_vectors(token2id: Dict[str, int], path: str) -> List[np.ndarray]:
    """
    Loads vectors from text file of pretrained model
    Args:
        token2id: token to id
        path: path of pretrained embedding text file

    Returns:

    """
    limit = None
    embed_shape = (len(token2id), 300)
    freqs = np.zeros((len(token2id)), dtype='f')

    vectors = np.zeros(embed_shape, dtype='f')
    i = 0
    with open(path, encoding="utf8", errors='ignore') as f:
        for o in f:
            token, *vector = o.split(' ')
            token = str.lower(token)
            if len(o) <= 100:
                continue
            if limit is not None and i > limit:
                break
            vectors[token2id[token]] = np.array(vector, 'f')
            i += 1

    return vectors


def load_corpus_file(data: str, token2id: Dict[str, int], vocab_freq_dict: Dict[str, int]):
    """

    Args:
        data: path of corpus file
        token2id: token to id
        vocab_freq_dict: token to frequency map

    Returns:
        training_examples, token2id, vocab_freq_dict
    """

    # reading `data` to get training exmaples and `token2id` and `vocab_freq_dict`
    # starting with 3 because 0, 1, 2 are already taken by <PAD>, <s>, <e>
    id_ = 3
    training_examples = []
    file = open("{}".format(data),'r', encoding="utf-8")
    for line in file.readlines():
        words = line.strip().split(" ")
        training_examples.append(words)
        for word in words:
            if word not in vocab_freq_dict:
                vocab_freq_dict.update({word:0})
            vocab_freq_dict[word] += 1
            if word not in token2id:
                token2id.update({word: id_})
                id_ += 1
    return training_examples, token2id, vocab_freq_dict


def add_glove_vocab(embedding_path, token2id, vocab_freq_dict):
    # including glove vocab(words absent in training data but present in glove) into token2id and vocab_freq_dict
    max_id = max(token2id.items(), key=operator.itemgetter(1))[0]
    max_token_id = token2id[max_id]
    with open(embedding_path, encoding="utf8", errors='ignore') as f:
        for o in f:
            token, *vector = o.split(' ')
            token = str.lower(token)
            if len(o) <= 100:
                continue
            if token not in token2id:
                max_token_id += 1
                token2id.update({token: max_token_id})
                vocab_freq_dict.update({token: 1})
    return token2id, vocab_freq_dict


def fine_tune_glove(token2id, vocab_freq_dict, embedding_path):
    # Loading vectors
    vectors = load_vectors(token2id, embedding_path)
    vec = KeyedVectors(300)
    vec.add(list(token2id.keys()), vectors, replace=True)

    # setting vectors(numpy_array) to None to release memory
    vectors = None

    # defining parameters for w2v model
    params = dict(min_count=1,workers=14,iter=6,size=300)

    # initializing model
    model = Word2Vec(**params)
    print("model created")
    model.build_vocab_from_freq(vocab_freq_dict)
    print("\nvocab built")
    idxmap = np.array([token2id[w] for w in model.wv.index2entity])
    print("\nindex map built")
    model.wv.vectors[:] = vec.vectors[idxmap]
    print("\nsyn0 done")
    model.trainables.syn1neg[:] = vec.vectors[idxmap]
    print("\nsyn1 done")

    print("\ntraining started")
    model.train(training_examples, total_examples=len(training_examples), epochs=model.epochs)
    print("\ntraining completed")
    output_path = 'model_dir_wiki_glove/final_model.model'
    model.save(output_path)
    print("\nmodel saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='finetune_embeddings')
    parser.add_argument('--corpus_path', type=str, default=None, help='path of corpus file')
    parser.add_argument('--embed_dimension', type=int, default=300, help='dimension of word embedding')
    parser.add_argument('--pretraied_glove_path', type=str, default=None, help='path of glove file')

    parser.add_argument('--mode', type=int, default=300, help='mode of training.')
    parser.add_argument('--corpus_dir', type=str, default=None, help='path of corpus file')
    parser.add_argument('--embed_dimension', type=int, default=300, help='dimension of word embedding')

    args = parser.parse_args()

    # path of glove embeddings
    embedding_path = args.pretraied_glove_path
    # path to data file
    data_path = args.corpus_path
    # map of token to unique id
    token2id = {"<PAD>": 0, "<s>": 1, "<e>": 2}
    # map of token to its frequency in data
    vocab_freq_dict = {}

    training_examples, token2id, vocab_freq_dict = load_corpus_file(data_path, token2id, vocab_freq_dict)
    if add_glove_vocab:
        token2id, vocab_freq_dict = add_glove_vocab(embedding_path, token2id, vocab_freq_dict)

    # saving `vocab_freq_dict`
    with open("vocab_freq_dict_wiki_glove", "wb") as vocab_file:
        pickle.dump(vocab_freq_dict, vocab_file)
    # saving `token2id`
    with open("token2id_wiki_glove", "wb") as token2id_file:
        pickle.dump(token2id, token2id_file)

    fine_tune_glove(token2id, vocab_freq_dict, embedding_path)
