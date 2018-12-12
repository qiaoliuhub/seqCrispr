#!/usr/bin/env python

import logging
import os
import importlib
import sys
import pickle
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from scipy.stats import spearmanr
from keras.layers import Input
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import h2o
import feature_imp
import utils
import process_features
import models

# setting up nvidia GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Setting up the correct config file
config_path = ".".join(sys.argv[1].split("/")[-3:]) + "." if len(sys.argv) >= 2 and sys.argv[1].split("/")[-1].startswith("run") else ""
config = importlib.import_module(config_path+"config")

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(config.run_specific_log, mode='a')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Recurrent neural network")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

def get_prediction_regular(model, unique_list, output_data, test_data):

    prediction = model.predict(x=[data[unique_list, :] for data in test_data])
    performance = spearmanr(prediction, output_data[unique_list, :])[0]

    return performance, pd.Series(list(prediction))


def __ml_train(X, extra_crispr_df, y, train_index, test_index):

    logger.debug("Creating h2o working environment")
    # ### Start H2O
    # Start up a 1-node H2O cloud on your local machine, and allow it to use all CPU cores and up to 2GB of memory:
    h2o.init(max_mem_size="2G")
    h2o.remove_all()
    logger.debug("Created h2o working environment successfully")

    from h2o.estimators import H2ORandomForestEstimator

    rf_crispr = H2ORandomForestEstimator(
        model_id="rf_crispr",
        categorical_encoding="enum",
        nfolds=5,
        ntrees=30,
        stopping_rounds=30,
        score_each_iteration=True,
        seed=10)

    seq_data = X.iloc[:, :config.seq_len]
    seq_data.columns = ['pos_' + str(i) for i in range(len(seq_data.columns))]
    pre_h2o_df = pd.concat([seq_data, extra_crispr_df, y], axis=1)
    h2o_crispr_df_train = h2o.H2OFrame(pre_h2o_df.loc[train_index, :])
    h2o_crispr_df_test = h2o.H2OFrame(pre_h2o_df.loc[test_index, :])

    logger.debug("Training machine learning model")
    rf_crispr.train(x=h2o_crispr_df_train.col_names[:-1], y=h2o_crispr_df_train.col_names[-1],
                    training_frame=h2o_crispr_df_train)
    logger.debug("Trained successfully. Output feature importance")
    feature_importance = rf_crispr._model_json['output']['variable_importances'].as_data_frame()[
        ['variable', 'percentage']]
    feature_importance.to_csv(config.feature_importance_path, index=False)

    logger.debug("Predicting training data")
    test_prediction_train = rf_crispr.predict(h2o_crispr_df_train[:-1])
    performance = spearmanr(test_prediction_train.as_data_frame()['predict'], h2o_crispr_df_train.as_data_frame()['log2fc'])[0]
    logger.debug("spearman correlation coefficient for training dataset is: %f" % performance)

    logger.debug("Predicting test data")
    test_prediction = rf_crispr.predict(h2o_crispr_df_test[:-1])
    performance = spearmanr(test_prediction.as_data_frame()['predict'], h2o_crispr_df_test.as_data_frame()['log2fc'])[0]
    logger.debug("spearman correlation coefficient for test dataset is: %f" % performance)

    logger.debug("Saving model")
    h2o.save_model(rf_crispr, config.ml_model_path)
    logger.debug("Saved model to disk")

def machine_learning_process(X, extra_crispr_df, y, train_index, test_index):

    try:
        __ml_train(X, extra_crispr_df, y, train_index, test_index)
    except:
        logger.debug("Fail to use random forest")
    finally:
        h2o.cluster().shutdown()

def read_data(input):

    logger.debug("Getting and processing Crispr dataset %s" % input)
    crispr = pd.read_csv(input)
    # scale_features
    process_features.scale_features(crispr)
    process_features.scale_output(crispr)
    logger.debug("Read and process data successfully")
    return crispr

def transform_data(crispr):

    logger.debug("Transforming data")
    # The last three nucletides are PAM sequence
    crispr['PAM'] = crispr['sequence'].str[-3:]

    # Seperate sequence to 3mers
    X = crispr.loc[:, 'sequence'].apply(lambda seq: utils.split_seqs(seq[:20]))
    logger.debug("Get sequence sucessfully")

    # upscale y values
    y = pd.DataFrame(crispr[config.y] * 10)

    # generate groups
    logger.debug("Generating groups based on gene names")
    if config.group:
        crispr.loc[:, "group"] = pd.Categorical(crispr.loc[:, config.group])
    logger.debug("Generated groups information successfully")
    logger.debug("Transformed data successfully")
    return X, y

def split_data(crispr):

    logger.debug("Splitting dataset to training dataset and testing dataset based on genes")
    if os.path.exists(config.train_index) and os.path.exists(config.test_index):
        train_index = pickle.load(open(config.train_index, "rb"))
        test_index = pickle.load(open(config.test_index, "rb"))
    else:
        train_test_split = getattr(process_features, config.split_method+"_split", process_features.regular_split)
        train_index, test_index = train_test_split(crispr, group_col = config.group_col, n_split = max(len(crispr)/1200, 2), rd_state=0)

        with open(config.train_index, 'wb') as train_file:
                pickle.dump(train_index, train_file)
        with open(config.test_index, 'wb') as test_file:
                pickle.dump(test_index, test_file)

    # Only select test data in one cell line if needed
    if config.test_cellline:
        test_cellline_index = crispr[crispr['cellline'] == config.test_cellline].index
        test_index = test_cellline_index.intersection(test_index)
    logger.debug("Splitted data successfully")

    return train_index, test_index

def process_biological_features(crispr):

    logger.debug("Generating one hot vector for categorical data")
    extra_crispr_df = crispr[config.extra_categorical_features + config.extra_numerical_features]
    n_values = [config.pam_dim] + ([2] * (len(config.extra_categorical_features)-1)) if config.with_pam else [2] * len(config.extra_categorical_features)
    process_features.process_categorical_features(extra_crispr_df, n_values)
    logger.debug("Generating on hot vector for categorical data successfully")
    return extra_crispr_df

def _transfer_learning_model():

    if config.retraining:
        loaded_model = load_model(config.transfer_learning_model,
                                  custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf': tf})

        if config.model_type == 'cnn':

            for_layer = loaded_model.get_layer(name='embedding_1')
            for_layer.trainable = config.fine_tune_trainable

            full_connected = loaded_model.get_layer(name='sequential_2')

        elif config.model_type == 'mixed':

            for_layer = loaded_model.get_layer(name='sequential_1')
            for_layer = for_layer.get_layer(name='embedding_2')
            for_layer.trainable = config.fine_tune_trainable

            cnn_layer = loaded_model.get_layer(name='sequential_2')
            cnn_layer = cnn_layer.get_layer(name='embedding_1')
            cnn_layer.trainable = config.fine_tune_trainable

            full_connected = loaded_model.get_layer(name='sequential_3')

        else:
            for_layer = loaded_model.get_layer(name='sequential_1')
            if config.frozen_embedding_only:
                for_layer = for_layer.get_layer(name='embedding_1')
            for_layer.trainable = config.fine_tune_trainable
            if config.rev_seq:
                rev_layer = loaded_model.get_layer(name='sequential_2')
                if config.frozen_embedding_only:
                    rev_layer = rev_layer.get_layer(name='embedding_2')
                rev_layer.trainable = config.fine_tune_trainable
                full_connected = loaded_model.get_layer(name='sequential_3')
            else:
                full_connected = loaded_model.get_layer(name='sequential_2')

        for i in xrange(int((len(full_connected.layers) / 4) * (1 - config.fullly_connected_train_fraction))):
            dense_layer = full_connected.get_layer(name='dense_' + str(i + 1))
            dense_layer.trainable = config.fine_tune_trainable

        crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)
        return crispr_model

def built_model(x_train, extra_crispr_df):

    logger.debug("Building the RNN graph")
    weight_matrix = [utils.get_weight_matrix()] if config.word2vec_weight_matrix else None
    x_train_len = x_train.shape[1]
    extra_x_len = extra_crispr_df.shape[1]
    for_seq_input = Input(shape=(x_train_len,))
    bio_features = Input(shape=(extra_x_len,))
    crispr_model = models.CrisprCasModel(bio_features=bio_features, for_seq_input=for_seq_input,
                                         weight_matrix=weight_matrix).get_model()
    if config.transfer_learning:
        crispr_model = _transfer_learning_model()

    utils.output_model_info(crispr_model)
    logger.debug("Built the RNN model successfully")
    return crispr_model

def deep_learning_process(crispr_model, x_input_train, extra_x_train, y_train):

    try:
        if config.training:
            logger.debug("Training the model")
            checkpoint = ModelCheckpoint(config.temp_hdf5_path, verbose=1, save_best_only=True, period=1)
            reduce_lr = LearningRateScheduler(utils.cosine_decay_lr)

            index_range = range(len(y_train))
            np.random.shuffle(index_range)
            selected_index = index_range[:int(config.retraining_datasize*len(y_train))]
            logger.debug("selecting %d data for training" %(config.retraining_datasize*len(y_train)))

            features_list = [x_input_train[selected_index], extra_x_train[selected_index]]
            training_history = utils.print_to_training_log(crispr_model.fit)(x=features_list,
                                                validation_split=0.1, y=y_train[selected_index],
                                                epochs=config.n_epochs,
                                                batch_size=config.batch_size, verbose=2,
                                                callbacks=[checkpoint, reduce_lr])

            logger.debug("Saving history")
            with open(config.training_history, 'wb') as history_file:
                pickle.dump(training_history.history, history_file)
            logger.debug("Saved training history successfully")

            logger.debug("Trained crispr model successfully")

    except KeyboardInterrupt:

        logger.debug("Loading model")
        loaded_model = load_model(config.temp_hdf5_path, custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf':tf})
        crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)
        logger.debug("Load in model successfully")

    finally:
        return crispr_model

def predict(train_list, y_train, unique_train_index, test_list, y_test, unique_test_index, model_path):

    logger.debug("Loading model for testing")
    loaded_model = load_model(model_path,
                              custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf': tf})
    crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)
    logger.debug("Load in model successfully")
    logger.debug("Predicting data with model")

    train_prediction = crispr_model.predict(x = train_list)
    train_performance = spearmanr(train_prediction, y_train[unique_train_index])
    logger.debug("GRU model spearman correlation coefficient for training dataset is: %s" % str(train_performance))

    get_prediction = getattr(sys.modules[__name__], "get_prediction_" + config.test_method, get_prediction_regular)
    performance, prediction = get_prediction(crispr_model, unique_test_index, y_test, test_list)
    logger.debug("GRU model spearman correlation coefficient: %s" % str(performance))
    return performance, prediction

def feature_importance(crispr_model, x_len, xs, y, unique_test_index):

    logger.debug("Getting features ranks")
    names = []
    names += ["for_" + str(i) for i in range(x_len)]
    names += config.extra_categorical_features + config.extra_numerical_features
    ranker = feature_imp.InputPerturbationRank(names)
    feature_ranks = ranker.rank(20, y[unique_test_index], crispr_model,
                                [data[unique_test_index] for data in xs])
    feature_ranks_df = pd.DataFrame(feature_ranks)
    feature_ranks_df.to_csv(config.feature_importance_path, index=False)
    logger.debug("Get features ranks successfully")

def run():

    # Prepare data
    crispr = read_data(config.input_dataset)
    X, y = transform_data(crispr)

    # Split data
    train_index, test_index = split_data(crispr)
    logger.debug("training data amounts: %s, testing data amounts: %s" % (len(train_index), len(test_index)))
    x_train, x_test, y_train_df, y_test_df = X.loc[train_index, :], X.loc[test_index, :], \
                                       y.loc[train_index, :], y.loc[test_index, :]
    x_input_train, x_input_test, y_train, y_test = x_train.values, x_test.values, \
                                                   y_train_df.values, y_test_df.values

    # process biological data
    extra_crispr_df = process_biological_features(crispr)
    extra_x_train, extra_x_test = extra_crispr_df.loc[train_index, :].values, extra_crispr_df.loc[test_index, :].values

    # deduplication
    logger.debug("Deduplication")
    _, unique_train_index = np.unique(pd.concat([x_train, y_train_df], axis=1), return_index=True, axis=0)
    _, unique_test_index = np.unique(pd.concat([x_test, y_test_df], axis=1), return_index=True, axis=0)
    logger.debug("after deduplication, training data amounts: %s, testing data amounts: %s" % (len(unique_train_index), len(unique_test_index)))

    # training random forest model and predict results
    if config.ml_train:
        machine_learning_process(X, extra_crispr_df, y, train_index, test_index)
        return

    # training deep learning model
    crispr_model = built_model(x_train, extra_crispr_df)
    crispr_model = deep_learning_process(crispr_model, x_input_train, extra_x_train, y_train)
    logger.debug("Persisting model")
    crispr_model.save(config.hdf5_path)

    # predict with new model
    train_list = [x_input_train[unique_train_index], extra_x_train[unique_train_index]]
    test_list = [x_input_test, extra_x_test]
    performance, prediction = predict(train_list, y_train, unique_train_index, test_list, y_test,
            unique_test_index, config.temp_hdf5_path)
    last_performance, last_prediction = predict(train_list, y_train, unique_train_index, test_list, y_test,
            unique_test_index, config.hdf5_path)
    utils.output_config_info()

    # save prediction results
    logger.debug("Saving test and prediction data plot")
    if last_performance > performance:
        prediction = last_prediction
    utils.ytest_and_prediction_output(y_test[unique_test_index], prediction)
    logger.debug("Saved test and prediction data plot successfully")

    # Check features importance
    if config.check_feature_importance:
        if performance > last_performance:
            loaded_model = load_model(config.temp_hdf5_path,custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf': tf})
            crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)
        feature_importance(crispr_model, x_train.shape[1], test_list, y_test, unique_test_index)

if __name__ == "__main__":

    try:
        run()
        logger.debug("new directory %s" % config.run_dir)

    except:

        import shutil
        shutil.rmtree(config.run_dir)
        logger.debug("clean directory %s" %config.run_dir)
        raise