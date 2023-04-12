#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
from evaluate_model import *
import numpy as np, os, sys
import random
import pandas as pd
from io import StringIO
from sklearn.model_selection import KFold
import tensorflow as tf
from scipy import signal
from typing import Generator
from tensorflow.keras.preprocessing.sequence import pad_sequences
import coral_ordinal as coral
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    
    SIGNAL_LEN = 30000 # samples
    BATCH_SIZE = 20
    EPOCHS = 5
    LEARNING_RATE = 0.00001
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    patient_ids = np.asarray(patient_ids)
    num_patients = len(patient_ids)
    

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Get data filenames.
    if verbose >= 1:
        print('Get valid filenames...')

    filenames = get_valid_filenames_from_patient_ids(data_folder, patient_ids)
    random.shuffle(filenames)

    # Define the models.
    if verbose >= 1:
        print('Building models...')

    cpc_model = build_iception_model((SIGNAL_LEN,18), 5)
    cpc_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss = coral.OrdinalCrossEntropy(), metrics = [coral.MeanAbsoluteErrorLabels()])

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    cpc_model.fit(x = batch_generator(batch_size=BATCH_SIZE, signal_len=SIGNAL_LEN, gen = generate_data(data_folder, filenames,samples=SIGNAL_LEN)), 
                  epochs=EPOCHS, steps_per_epoch=len(filenames)/BATCH_SIZE, verbose=verbose)

    # Save the models.
    save_challenge_model(model_folder, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    SIGNAL_LEN = 30000 # samples
    model = build_iception_model((SIGNAL_LEN,18), 5)
    filename = os.path.join(model_folder, 'model_weights.h5')
    model.load_weights(filename)
    if verbose >= 1:
        print('Model loaded...')
    return model


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    SIGNAL_LEN = 30000 # samples

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    patient_pred = [] 
    for recording in recording_data:
        if recording[1] == None:
            patient_pred.append(np.nan)
        else:
            current_recording = recording[0]
            current_recording = np.moveaxis(current_recording,0,-1)
            current_prediction = models.predict(np.expand_dims(current_recording[:SIGNAL_LEN],0))
            current_prediction = np.asarray(coral.ordinal_softmax(current_prediction))
            current_outcome_hat_probability = current_prediction[:,3:].sum(axis=1)
            patient_pred.append(current_outcome_hat_probability)
    patient_pred = np.asarray(patient_pred)

    outcome_probability = np.nanmean(patient_pred)
    outcome = int((outcome_probability > 0.5) * 1)


    cpc = (outcome_probability * 4) + 1
    cpc = np.nan_to_num(cpc, nan=3.0)

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    if verbose >= 1:
        print('Prediction on patient N succeeded...')

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


def cross_validate_model(data_folder, num_folds, verbose):
    # Find data files.

    SIGNAL_LEN = 30000 # samples
    BATCH_SIZE = 20
    EPOCHS = 5
    LEARNING_RATE = 0.00001

    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)
    patient_ids = np.asarray(patient_ids)
    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    #os.makedirs(model_folder, exist_ok=True)

    # Make CV folds
    if verbose >= 1:
        print('Split the data into {} cross-validation folds'.format(num_folds))
    
    kf = KFold(n_splits=num_folds)

    challenge_score = np.zeros(num_folds)
    auroc_outcomes = np.zeros(num_folds)
    auprc_outcomes = np.zeros(num_folds)
    accuracy_outcomes = np.zeros(num_folds)
    f_measure_outcomes = np.zeros(num_folds)
    mse_cpcs = np.zeros(num_folds)
    mae_cpcs = np.zeros(num_folds)
    all_preds_outcome  = []
    all_labels_outcome  = []

    for i, (train_index, val_index) in enumerate(kf.split(patient_ids)): #TODO: Stratify based on bothe outcomes and cpcs
        print(f"Fold {i}:")

        train_ids, val_ids = patient_ids[train_index], patient_ids[val_index]

        train_filenames = get_valid_filenames_from_patient_ids(data_folder, train_ids)
        val_filenames = get_valid_filenames_from_patient_ids(data_folder, val_ids)

        # Random shuffle filenames to avoid all signals from on person ending up in the same batch
        random.shuffle(train_filenames)
        random.shuffle(val_filenames)

        # Define the models.

        cpc_model = build_iception_model((SIGNAL_LEN,18), 5)
        cpc_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss = coral.OrdinalCrossEntropy(), metrics = [coral.MeanAbsoluteErrorLabels()])


        # Train the models.
        if verbose >= 1:
            print('Training the Challenge models on the Challenge data...')
    
        cpc_model.fit(x = batch_generator(batch_size=BATCH_SIZE, signal_len=SIGNAL_LEN, gen = generate_data(data_folder, train_filenames,samples=SIGNAL_LEN)), epochs=EPOCHS, 
                                              validation_data=batch_generator(batch_size=BATCH_SIZE, signal_len=SIGNAL_LEN, gen = generate_data(data_folder, val_filenames)),
                                              steps_per_epoch=len(train_filenames)/BATCH_SIZE, validation_steps=len(val_filenames)/BATCH_SIZE,validation_freq=1)

        print('Test model on validation data...')
        val_prediction = []
        for j in range(len(val_ids)):
            if verbose >= 2:
                print('    {}/{}...'.format(j+1, len(val_ids)))

            # Load data.
            patient_id = val_ids[j]
            patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

            patient_pred = []
            patient_pred.append(patient_id)
            patient_pred.append(get_outcome(patient_metadata))
            patient_pred.append(get_cpc(patient_metadata))    
            for val_recording in recording_data:
                if val_recording[1] == None:
                    patient_pred.append(np.nan)
                else:
                    current_recording = val_recording[0]
                    current_recording = np.moveaxis(current_recording,0,-1)
                    current_prediction = cpc_model.predict(np.expand_dims(current_recording[:SIGNAL_LEN],0))
                    current_prediction = np.asarray(coral.ordinal_softmax(current_prediction))
                    current_outcome_hat_probability = current_prediction[:,3:].sum(axis=1)
                    #current_outcome_hat = (current_outcome_hat_probability > 0.5)*1
                    patient_pred.append(current_outcome_hat_probability)
                    #outcome_hat.append(current_outcome_hat)
            val_prediction.append(patient_pred)
        val_prediction = np.asarray(val_prediction) 
        val_prediction = pd.DataFrame(val_prediction)
        val_prediction.iloc[:,1:3] = val_prediction.iloc[:,1:3].astype("int")
        val_prediction.iloc[:,3:] = val_prediction.iloc[:,3:].astype("float")
        val_prediction.to_csv("prediction_fold_{}.csv".format(i))

        val_outcome = np.nanmean(val_prediction.iloc[:,4:],axis=1)

        val_cpc = val_outcome * 5
        val_cpc = np.nan_to_num(val_cpc, nan=2.5)

        # Ensure that the CPC score is between (or equal to) 1 and 5.

        val_cpc = np.clip(val_cpc, 1, 5)

        challenge_score[i] = compute_challenge_score(val_prediction.iloc[:,1],val_outcome)
        auroc_outcomes[i], auprc_outcomes[i] = compute_auc(val_prediction.iloc[:,1],val_outcome)
        accuracy_outcomes[i], _, _  = compute_accuracy(val_prediction.iloc[:,1],val_outcome)
        f_measure_outcomes[i], _, _  = compute_f_measure(val_prediction.iloc[:,1],val_outcome)
        mse_cpcs[i] = compute_mse(val_prediction.iloc[:,2],val_cpc)
        mae_cpcs[i] = compute_mae(val_prediction.iloc[:,2],val_cpc)

        print(f"challenge_score={challenge_score[i]},  auroc_outcomes={auroc_outcomes[i]}, auprc_outcomes={auprc_outcomes[i]},accuracy_outcomes={accuracy_outcomes[i]}, f_measure_outcomes={f_measure_outcomes[i]}, mse_cpcs={mse_cpcs[i]}, mae_cpcs={mae_cpcs[i]}")
    all_preds_outcome = np.asarray(all_preds_outcome)
    all_labels_outcome = np.asarray(all_labels_outcome)
    return all_preds_outcome , all_labels_outcome , challenge_score, auroc_outcomes, auprc_outcomes, accuracy_outcomes, f_measure_outcomes, mse_cpcs, mae_cpcs

# Save your trained model.
#def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
#    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
#    filename = os.path.join(model_folder, 'models.sav')
#    joblib.dump(d, filename, protocol=0)

def save_challenge_model(model_folder, model):
    filename = os.path.join(model_folder, 'model_weights.h5')
    model.save_weights(filename)


def _inception_module(input_tensor, stride=1, activation='linear', use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = tf.keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                              strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                  padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = tf.keras.layers.Concatenate(axis=2)(conv_list)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                      padding='same', use_bias=False)(input_tensor)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    x = tf.keras.layers.Add()([shortcut_y, out_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_iception_model(input_shape, nb_classes, depth=6, use_residual=True, lr_init = 0.001, kernel_size=40, bottleneck_size=32, nb_filters=32, outputfunc="sigmoid", loss=tf.keras.losses.BinaryCrossentropy()):
    input_layer = tf.keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x,kernel_size = kernel_size, bottleneck_size=bottleneck_size, nb_filters=nb_filters)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)     
    output_layer = coral.CoralOrdinal(num_classes = 5)(gap_layer)
    #output_layer = tf.keras.layers.Dense(units=nb_classes,activation=outputfunc)(gap_layer)  
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    print("Inception model built.")
    return model

def get_signal_indx(recording_metadata, task):
  """
  task: 0,1,2,3 - 0: 0,12 hours, 1: 0-24 hours, 2: 0-48 hours, 3: 0-72 hours
  signal_len: desired signal lenght to extract from the total signal
  """
  TIMESLOTS = np.asarray([(0,12),(0,24),(0,48),(0,72)])
  time = TIMESLOTS[task]
  valid_signals = pd.read_csv(StringIO(recording_metadata), sep='\t')[time[0]:time[1]].dropna()
  if valid_signals.shape[0] > 0:
    indx = valid_signals.index
  else:
    indx = np.nan
  return indx

def get_best_signal(recordings,indx, time):
    signal = recordings[indx][0]
    sample_freq = recordings[indx][1]
    truncated_signal = signal[:,:time*int(sample_freq)]
    return truncated_signal

def extract_signal(recording_metadata,recordings, task=4,time=30):
    indx = get_signal_indx(recording_metadata, task)
    signal = get_best_signal(recordings,indx, time)
    return signal


def get_valid_filenames_from_patient_ids(data_folder, patient_ids):
    filenames = []
    for ids in patient_ids:
        for filepath in os.listdir(os.path.join(data_folder,ids)):
            if filepath.endswith(".mat"):
                
                filename = os.path.join(data_folder,ids,filepath.strip(".mat"))
                filenames.append(filename)
    return np.asarray(filenames)


def batch_generator(batch_size: int, gen: Generator, signal_len:int):
    batch_features = np.zeros((batch_size, signal_len, 18))
    batch_labels = np.zeros((batch_size, 1))

    while True:
        for i in range(batch_size):
            batch_features[i], batch_labels[i] = next(gen)
        yield batch_features, batch_labels


def generate_data(folder: str, filenames, samples):
    while True:
        for filename in filenames:
            patient_id = get_patient_id_from_path(filename)
            patient_metadata, _, _ = load_challenge_data(folder, patient_id)
            #current_outcome = get_outcome(patient_metadata)
            current_cpc = get_cpc(patient_metadata)
            recording_data, sampling_frequency, _ = load_recording(filename)
            recording_data = np.moveaxis(recording_data,0,-1)[:int(samples)]
            yield recording_data, current_cpc

def map_regression_to_proba(pred):
    new_pred = []
    if pred <= 3:
        new_pred = pred / 6
    elif pred > 3:
        new_pred = 0.5 + (pred-3) / 4
    return new_pred

def map_proba_to_regression(pred):
    new_pred = []
    if pred <= 3:
        new_pred = pred / 6
    elif pred > 3:
        new_pred = 0.5 + (pred-3) / 4
    return new_pred

def get_patient_id_from_path(path,location = "colab"):
    if location == "colab":
        return path.split("/")[-2]
    elif location == "local":
        return path.split("/")[-1].split("\\")[0]

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
