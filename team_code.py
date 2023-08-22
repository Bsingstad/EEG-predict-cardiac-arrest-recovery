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
    
    SIGNAL_LEN = 300 # sec
    FREQ = 100
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

    cpc_model = build_iception_model((SIGNAL_LEN*FREQ,18), 5)
    cpc_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss = coral.OrdinalCrossEntropy(), metrics = [coral.MeanAbsoluteErrorLabels()])

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    cpc_model.fit(x = batch_generator(batch_size=BATCH_SIZE, signal_len=SIGNAL_LEN, gen = generate_data(data_folder, filenames,seconds=SIGNAL_LEN)), 
                  epochs=EPOCHS, steps_per_epoch=len(filenames)/BATCH_SIZE, verbose=verbose)

    # Save the models.
    save_challenge_model(model_folder, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    SIGNAL_LEN = 300 
    SAMPLE_FREQ = 100 # seconds
    model = build_iception_model((SIGNAL_LEN * SAMPLE_FREQ,18), 5)
    filename = os.path.join(model_folder, 'model_weights.h5')
    model.load_weights(filename)
    if verbose >= 1:
        print('Model loaded...')
    return model


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    SIGNAL_LEN = 300 # sec
    FREQ = 100
    # Load data.
    #patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)

    #patient_ids = find_data_folders(data_folder)
    patient_pred = []
    for recording_id in recording_ids:
        try:
            temp_recording_data, _, sampling_frequency = load_recording_data(os.path.join(data_folder,patient_id,recording_id + "_EEG"))
            if temp_recording_data.shape[1] > int(SIGNAL_LEN*sampling_frequency):
                recording_data = temp_recording_data[:,:int(SIGNAL_LEN*sampling_frequency)]
            elif temp_recording_data.shape[1] <= int(SIGNAL_LEN*sampling_frequency):
                diff = int(SIGNAL_LEN*sampling_frequency) - temp_recording_data.shape[1]
                recording_data = np.pad(temp_recording_data,((0, 0), (0, diff)), mode='constant')
            recording_data = raw_to_bipolar(recording_data[:,:int(SIGNAL_LEN*sampling_frequency)])
            recording_data = np.moveaxis(recording_data,0,-1)
            recording_data = scipy.signal.resample(recording_data, int((FREQ/sampling_frequency)*recording_data.shape[0]), axis=1)
            recording_data = recording_data[:SIGNAL_LEN*FREQ,:18] # stygg hardkoding her
        except:
            recording_data = np.zeros((SIGNAL_LEN*FREQ,18))
        #print(recording_data.shape)
        current_prediction = models.predict(np.expand_dims(recording_data,0))
        current_prediction = np.asarray(coral.ordinal_softmax(current_prediction))
        current_outcome_hat_probability = current_prediction[:,3:].sum(axis=1)
        patient_pred.append(current_outcome_hat_probability)
    patient_pred = np.asarray(patient_pred)

    outcome_probability = np.nanmean(patient_pred)
    if np.isnan(outcome_probability) == True:
        # If all probas = nan --> set = 0.5 +- 0.005
        outcome_probability = 0.5 + random.random()/100 -0.005
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

    BATCH_SIZE_LSTM = 30
    EPOCHS_LSTM = 20
    LEARNING_RATE_LSTM = 0.0001

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
    
        cpc_model.fit(x = batch_generator(batch_size=BATCH_SIZE, signal_len=SIGNAL_LEN, gen = generate_data(data_folder, train_filenames,seconds=SIGNAL_LEN)), epochs=EPOCHS, 
                                              validation_data=batch_generator(batch_size=BATCH_SIZE, signal_len=SIGNAL_LEN, gen = generate_data(data_folder, val_filenames,seconds=SIGNAL_LEN)),
                                              steps_per_epoch=len(train_filenames)/BATCH_SIZE, validation_steps=len(val_filenames)/BATCH_SIZE,validation_freq=1)


        cnn_backbone = delete_last_layer(cpc_model)

        cnn_feature_vectors = np.zeros((len(train_ids),72, 128,1))
        cpc_train = np.zeros((len(train_ids),1))

        for k in range(len(train_ids)):
            if verbose >= 2:
                print('    {}/{}...'.format(k+1, len(train_ids)))
            
            patient_id = train_ids[j]
            patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
            cpc_train[k,:] = get_cpc(patient_metadata)
            for hour, val_recording in enumerate(recording_data):
                if val_recording[1] == None:
                    continue
                    #cnn_feature_vectors[k,hour,:,0] = np.zeros(128)
                else:
                    current_recording = val_recording[0]
                    current_recording = np.moveaxis(current_recording,0,-1)
                    current_prediction = cnn_backbone.predict(np.expand_dims(current_recording[:SIGNAL_LEN],0))
                    cnn_feature_vectors[k,hour,:,0] = current_prediction
        

  

        reccurent_model = td_lstm_model()
        reccurent_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_LSTM), loss = coral.OrdinalCrossEntropy(), metrics = [coral.MeanAbsoluteErrorLabels()])
        
        reccurent_model.fit(x=cnn_feature_vectors, y=cpc_train, epochs=EPOCHS_LSTM, batch_size=BATCH_SIZE_LSTM)
        



        print('Test model on validation data...')
        all_predictions = []
        for j in range(len(val_ids)):
            if verbose >= 2:
                print('    {}/{}...'.format(j+1, len(val_ids)))

            # Load data.
            patient_id = val_ids[j]
            patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
            feature_vectors = np.zeros((72,128))
            patient_pred = []
            patient_pred.append(patient_id)
            patient_pred.append(get_outcome(patient_metadata))
            patient_pred.append(get_cpc(patient_metadata))    
            for rec_num, val_recording in enumerate(recording_data):
                if val_recording[1] == None:
                    continue
                else:
                    current_recording = val_recording[0]
                    current_recording = np.moveaxis(current_recording,0,-1)
                    feature_vectors[rec_num,:] = cnn_backbone.predict(np.expand_dims(current_recording[:SIGNAL_LEN],0))
            current_prediction = reccurent_model.predict(np.expand_dims(feature_vectors,0))
            current_prediction = np.asarray(coral.ordinal_softmax(current_prediction))
            #current_outcome_hat_probability = current_prediction[:,3:].sum(axis=1)
            #current_outcome_hat = (current_outcome_hat_probability > 0.5)*1
            patient_pred.append(current_prediction)
            all_predictions.append(patient_pred)

        all_predictions = np.asarray(all_predictions) 
        all_predictions = pd.DataFrame(all_predictions)
        all_predictions.iloc[:,1:3] = all_predictions.iloc[:,1:3].astype("int")
        all_predictions.iloc[:,3:] = all_predictions.iloc[:,3:].astype("float")
        all_predictions.to_csv("prediction_fold_{}.csv".format(i))
        #FIXME: 
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

    gap_layer = tf.keras.layers.GlobalAveragePooling1D(name="feature_vector")(x)     
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

def padarray(A, size):
    t = size - A.shape[1]
    return np.pad(A, pad_width=(0, t), mode='constant')

def extract_signal(recording_metadata,recordings, task=4,time=30):
    indx = get_signal_indx(recording_metadata, task)
    signal = get_best_signal(recordings,indx, time)
    return signal


def get_valid_filenames_from_patient_ids(data_folder, patient_ids):
    filenames = []
    for patient_id in patient_ids:
        recording_ids = find_recording_files(data_folder, patient_id)
        for recording_id in recording_ids:
            filename = os.path.join(data_folder,patient_id,recording_id + "_EEG")
            filenames.append(filename)
    return np.asarray(filenames)


def batch_generator(batch_size: int, gen: Generator, signal_len:int):
    FREQ = 100
    batch_features = np.zeros((batch_size, signal_len*FREQ, 18))
    batch_labels = np.zeros((batch_size, 1))

    while True:
        for i in range(batch_size):
            batch_features[i], batch_labels[i] = next(gen)
        yield batch_features, batch_labels


def generate_data(folder: str, filenames, seconds):
    FREQ = 100
    SIGNAL_LEN = 300
    while True:
        for filename in filenames:
            patient_id = get_patient_id_from_path(filename)
            patient_metadata = load_challenge_data(folder, patient_id)
            #current_outcome = get_outcome(patient_metadata)
            current_cpc = get_cpc(patient_metadata)
            try:
                temp_recording_data, _, sampling_frequency = load_recording_data(filename)
                if temp_recording_data.shape[1] > int(seconds*sampling_frequency):
                    recording_data = temp_recording_data[:,:int(seconds*sampling_frequency)]
                elif temp_recording_data.shape[1] <= int(seconds*sampling_frequency):
                    diff = int(seconds*sampling_frequency) - temp_recording_data.shape[1]
                    recording_data = np.pad(temp_recording_data,((0, 0), (0, diff)), mode='constant')

                recording_data = raw_to_bipolar(recording_data[:,:int(seconds*sampling_frequency)])
                recording_data = np.moveaxis(recording_data,0,-1)
                recording_data = scipy.signal.resample(recording_data, int((FREQ/sampling_frequency)*recording_data.shape[0]), axis=0)
                recording_data = recording_data[:FREQ*SIGNAL_LEN,:18] # stygg hardkoding her
            except:
                recording_data = np.zeros((FREQ*SIGNAL_LEN,18))
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

def raw_to_bipolar(raw_data):
    ## Bipolar reference
    bipolar_data = np.zeros((18,raw_data.shape[1]))
    bipolar_data[8,:] = raw_data[0,:] - raw_data[1,:]; # Fp1-F3
    bipolar_data[9,:] = raw_data[1,:] - raw_data[2,:]; # F3-C3
    bipolar_data[10,:] = raw_data[2,:] - raw_data[3,:]; # C3-P3
    bipolar_data[11,:] = raw_data[3,:] - raw_data[7,:]; # P3-O1

    bipolar_data[12,:] = raw_data[11,:] - raw_data[12,:]; # Fp2-F4
    bipolar_data[13,:] = raw_data[12,:] - raw_data[13,:]; # F4-C4
    bipolar_data[14,:] = raw_data[13,:] - raw_data[14,:]; # C4-P4
    bipolar_data[15,:] = raw_data[14,:] - raw_data[18,:]; # P4-O2

    bipolar_data[0,:] = raw_data[0,:] - raw_data[4,:];  # Fp1-F7
    bipolar_data[1,:] = raw_data[4,:] - raw_data[5,:]; # F7-T3
    bipolar_data[2,:] = raw_data[5,:] - raw_data[6,:]; # T3-T5
    bipolar_data[3,:] = raw_data[6,:] - raw_data[7,:]; # T5-O1

    bipolar_data[4,:] = raw_data[11,:] - raw_data[15,:]; # Fp2-F8
    bipolar_data[5,:] = raw_data[15,:] - raw_data[16,:]; # F8-T4
    bipolar_data[6,:] = raw_data[16,:] - raw_data[17,:]; # T4-T6
    bipolar_data[7,:] = raw_data[17,:] - raw_data[18,:]; # T6-O2

    bipolar_data[16,:] = raw_data[8,:] - raw_data[9,:];   # Fz-Cz
    bipolar_data[17,:] = raw_data[9,:] - raw_data[10,:]; # Cz-Pz
    return bipolar_data


def delete_last_layer(model, layer_name):
    """
    Take the model and return the same model without the last layer
    """
    new_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer(layer_name).output]
)
    return new_model

def td_lstm_model():
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.LSTM(128, activation='relu', return_sequences=False)
    )
    model.add(tf.keras.layers.LSTM(72, activation='relu')
    )
    model.add(coral.CoralOrdinal(num_classes = 5))
    return model
