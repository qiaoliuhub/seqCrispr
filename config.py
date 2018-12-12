import os
from datetime import datetime
import shutil
import sys
from pytz import timezone

# cellline folder
if len(sys.argv) >= 2 and not sys.argv[1].split("/")[-1].startswith("run"):
    cellline = sys.argv[1].split("/")[-1]
elif len(sys.argv) >= 2:
    cellline = sys.argv[1].split("/")[-2]
else:
    cellline = "test"

# data speficic name, for example, if file name is K562_CN, then data_specific is _CN
data_specific = "_CN_centrality" if cellline != "test" else ""

# split method, it could be stratified or regular
split_method = 'regular'

# model types: rnn, cnn or mixed
model_type = "rnn"
seq_cnn = False

# deco_part of output directory, run_ + deco_part + time
deco_part = "_".join([cellline, data_specific, split_method[:3], model_type[:3]])

# sequence length
seq_len = 20

# epochs
n_epochs = 20
patience = 100
decay_steps = 100
min_lr = 0.001

# model hyperparameters
start_lr = 0.01
dropout = 0.4
LSTM_hidden_unit = 32
batch_size = 32
embedding_voca_size = 65
embedding_vec_dim = 8
activation_method =["relu", "sigmoid"]
fully_connected_layer_layout = [64] * 2 + [32] * 2
cnn_levels = [8, 16, 32, 64, 128]
bio_fully_connected_layer_layout = [32, 64]
lr_decay = 0.0005
maxnorm = 20
with_pam = False
pam_dim = 4
LSTM_stacks_num = 4
rnn_time_distributed_dim = 32
word_len = 3
test_cellline = None

# could be None, "group" and "cellline"
group_col = None

# weight matrix cold start
word2vec_weight_matrix = False

# MIN_MAX scale feature
scale_feature = [] #+ ["r_d_tm" ] + ['aa_cut'] + ['essentiality'] + ['log_CN']

# RNN input features, every unit input is 200 elements vector
seq2vec_features = ["sequence"]

# extra features
extra_numerical_features = [] + ["r_d_tm" ] + ['aa_cut'] #+ ['essentiality'] + ['log_CN']
extra_categorical_features = [] #+ ["DNase"] + ["CTCF", "RRBS", "H3K4me3"]
if with_pam:
    extra_categorical_features = ["PAM"] + extra_categorical_features

# output
y = ["log2fc"]

# cols having gene names info or False to avoid this
group = False

# current directory
cur_dir = os.getcwd()

# Whether check feature importance using deep learning
check_feature_importance = True

# Whether to do RF training
ml_train = False

# Whether or not to train the model again
training = True
test_method = 'regular'

# parameters when retraining
transfer_learning = False
fine_tune_trainable = False
retraining_datasize = 1.0/1 if transfer_learning else 1
frozen_embedding_only = True
fullly_connected_train_fraction = 1.0/1
retraining_model_path = os.path.join(cur_dir, "dataset/test/run_test__lea_cnn1810111631")
transfer_learning_model = os.path.join(retraining_model_path, "lstm_model.h5")
retraining_dataset = ""

# each run specific
rerun_name = sys.argv[1].split("/")[-1] if len(sys.argv) >= 2 and sys.argv[1].split("/")[-1].startswith("run") else None
unique_name = rerun_name if rerun_name else "run_" + deco_part + datetime.now(timezone('US/Eastern')).strftime("%y%m%d%H%M")

data_dir = os.path.join(cur_dir, "dataset", cellline)
run_dir = os.path.join(data_dir, unique_name)

# pretrained seq2vec 3mer vector representation
seq2vec_mapping = os.path.join(cur_dir, "dataset", "cds_vector")
mismatch_matrix = os.path.join(cur_dir, "dataset", "off_target_matrix.csv")
pam_matrix = os.path.join(cur_dir, "dataset", "Pam_score.csv")

input_dataset = os.path.join(data_dir, "{!s}{!s}.csv".format(cellline, data_specific))
if transfer_learning and retraining_dataset != "":
    input_dataset = retraining_dataset

# training and test index
train_index = os.path.join(data_dir, "train_index" + data_specific)
test_index = os.path.join(data_dir, "test_index" + data_specific)
if transfer_learning:
    train_index += "_retrain"
    test_index += "_retrain"

# if execute with a new dataset, use the config.py in cur working directory, create a new
# directory for this execution, create __init__ to generate a package, then copy
# cur working directory to the new created directory
# else use the existed config file
run_specific_config = os.path.join(run_dir, "config.py")
cur_dir_config = os.path.join(cur_dir, "config.py")

if not os.path.exists(run_dir):
    os.makedirs(run_dir)
    open(os.path.join(data_dir, "__init__.py"), 'w+').close()
    open(os.path.join(run_dir, "__init__.py"), 'w+').close()
    shutil.copyfile(cur_dir_config, run_specific_config)

# Directory to save test and prediction y plot
test_prediction = os.path.join(run_dir, "test_prediction.csv")

# Directory to save the pickled training history
training_history = os.path.join(run_dir, "training_history.pi")

# model saving in hdf5 directory
hdf5_path = os.path.join(run_dir, "lstm_model.h5")

# machine learning model saving path
ml_model_path = os.path.join(run_dir, "ml_model")
feature_importance_path = os.path.join(run_dir, 'features.csv')

# temprature save models
temp_hdf5_path = os.path.join(run_dir, "temp_model.h5")

# directory to save logfile
run_specific_log = os.path.join(run_dir, "logfile")

# logfile to save the training process
training_log = os.path.join(run_dir, "training_log")

# logfile to save the file used for tensorboard
tb_log = os.path.join(run_dir, "tb_log")

# output_attrs list
output_attrs = ["split_method", "model_type", "dropout", "scale_feature", "seq2vec_features", "extra_numerical_features",
                "extra_categorical_features", "cellline", "word2vec_weight_matrix", "input_dataset"]



