import logging
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold, ShuffleSplit, StratifiedKFold
import numpy as np
import pandas as pd
import sys
import importlib

config_path = ".".join(sys.argv[1].split("/")[-3:]) + "." if len(sys.argv) >= 2 and sys.argv[1].split("/")[-1].startswith("run") else ""
config = importlib.import_module(config_path+"config")

logging.basicConfig()
logger = logging.getLogger("features_processing")
logger.setLevel(logging.DEBUG)

def scale_features(df):

    logger.debug("Scaling features")
    scaler = MinMaxScaler()
    if len(config.scale_feature):
        df.loc[:, config.scale_feature] = scaler.fit_transform(df.loc[:, config.scale_feature])
    logger.debug("Scale data successfully")


def scale_output(df):
    logger.debug("Scaling output")
    scaler = StandardScaler()
    scaler.fit_transform(df[config.y])
    logger.debug("Scale data successfully")


def process_categorical_features(df, cat_values):

    # Transfer categorical features to numerical label data using LabelEncoder
    if not len(config.extra_categorical_features):
        return

    for col in config.extra_categorical_features:

        le = LabelEncoder()
        if col == 'PAM':
            train3mer = [a + b + c for a in 'ATCG' for b in 'AG' for c in 'G']
            le.fit(train3mer)
            df.loc[:, col] = le.transform(df.loc[:,col])

        df.loc[:, col] = le.fit_transform(df.loc[:,col])

    # Generate a boolean mask, only for the one needed for one hot encoder
    categorical_features = df.columns.isin(config.extra_categorical_features)
    # Generate the one hot encoder to transform categorical data
    ohe = OneHotEncoder(n_values= cat_values, categorical_features=categorical_features, handle_unknown="ignore", sparse=False)
    ohe.fit_transform(df)

def regular_split(df, group_col=None, n_split = 10, rd_state = 3):

    shuffle_split = ShuffleSplit(test_size=1.0/n_split, random_state = rd_state)
    return shuffle_split.split(df).next()

def stratified_split(df, group_col=None, n_split = 10, rd_state = 3):

    skf = StratifiedKFold(n_splits=n_split, shuffle = True, random_state= rd_state)
    if group_col:
        label = df[group_col].values

    else:
        label = pd.cut(df[config.y[0]], bins=10).astype('category')

    le = LabelEncoder()
    label_y = le.fit_transform(label)
    skf_split = skf.split(np.zeros(len(df)), label_y)
    for i in range(rd_state):
        skf_split.next()
    return skf_split.next()