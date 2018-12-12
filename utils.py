import numpy as np
import pandas as pd
import tensorflow as tf
from keras.losses import mse
from keras.preprocessing.text import Tokenizer
import sys
import importlib

config_path = ".".join(sys.argv[1].split("/")[-3:]) + "." \
    if len(sys.argv) >= 2 and sys.argv[1].split("/")[-1].startswith("run") else ""
config = importlib.import_module(config_path+"config")

class __NtsTokenizer(Tokenizer):

    def __init__(self, nt):

        Tokenizer.__init__(self)
        if nt == 3:
            self.dic = [a + b + c for a in 'ATCG' for b in 'ATCG' for c in 'ATCG']
        elif nt == 2:
            self.dic = [a + b for a in 'ATCG' for b in 'ATCG']
        elif nt == 1:
            self.dic = [a for a in 'ATCG']
        else:
            self.dic = []
        self.fit_on_texts(self.dic)

def split_seqs(seq, nt = config.word_len):

    t = __NtsTokenizer(nt = nt)
    result = ''

    lens = len(seq)
    for i in xrange(lens):
        result += ' ' + seq[i:i+nt]

    seq_result = t.texts_to_sequences([result])
    return pd.Series(seq_result[0])

def get_weight_matrix():
    # get the seq2vec pre-trained vector representation of 3-mer
    embedding_index = {}
    t = __NtsTokenizer(nt = config.word_len)
    with open(config.seq2vec_mapping, 'r') as seq2vec_map:
        for line in seq2vec_map:
            data = line.split()
            trimer = data[0].lower()
            vector = np.asarray(data[1:], dtype='float32')
            embedding_index[trimer] = vector

    weight_matrix = np.zeros((config.embedding_voca_size, config.embedding_vec_dim))
    for word, index in t.word_index.items():
        embedding_vector = embedding_index[word]
        if embedding_vector is not None:
            weight_matrix[index] = embedding_vector

    return weight_matrix


def revised_mse_loss(y_true, y_pred):

    alpha = 0.9
    mse_result = mse(y_true, y_pred)
    large_coefficient = tf.where(tf.abs(y_true)<5, tf.fill(tf.shape(y_true), 0.0), tf.fill(tf.shape(y_true), 1.0))

    coefficient = tf.multiply(alpha, large_coefficient) + tf.multiply(1.0-alpha, 1.0-large_coefficient)
    result = tf.multiply(mse_result, coefficient)
    return result


def ytest_and_prediction_output(y_test, y_prediction):

    if isinstance(y_test, np.ndarray):
        y_test = pd.DataFrame(y_test)
    if isinstance(y_prediction, np.ndarray):
        y_prediction = pd.DataFrame(y_prediction.reshape(-1,))
    y_prediction.index = y_test.index
    test_prediction = pd.concat([y_test, y_prediction], axis=1)
    test_prediction.columns = ["ground_truth", "prediction"]
    test_prediction.to_csv(config.test_prediction)


def print_to_logfile(fun):

    def inner(*args, **kwargs):
        old_stdout = sys.stdout
        logfile = open(config.run_specific_log, 'a+')
        sys.stdout = logfile
        result = fun(*args, **kwargs)
        sys.stdout = old_stdout
        logfile.close()
        return result

    return inner


def print_to_training_log(fun):

    def inner(*args, **kwargs):

        old_stdout = sys.stdout
        logfile = open(config.training_log, 'a+')
        sys.stdout = logfile
        result = fun(*args, **kwargs)
        sys.stdout = old_stdout
        logfile.close()
        return result

    return inner

def cosine_decay_lr(epoch):

    global_step = min(epoch, config.decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / config.decay_steps))
    decayed = (1 - config.min_lr) * cosine_decay + config.min_lr
    decayed_learning_rate = config.start_lr * decayed
    return decayed_learning_rate


@print_to_training_log
def output_config_info():
    print "\n".join([(attr.ljust(40) + str(getattr(config, attr))) for attr in config.output_attrs])


@print_to_training_log
def output_model_info(model):
    model.summary()


