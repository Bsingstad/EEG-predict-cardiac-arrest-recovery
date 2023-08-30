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
    EPOCHS = 7
    LEARNING_RATE = 0.0001
    LSTM_EPOCHS = 50
    LSTM_BS = 20
    LEADS = ["Fp1","Fp2","F7","F8","F3","F4","T3", "T4", "C3","C4","T5","T6","P3","P4","O1","O2","Fz","Cz","Pz","Fpz","Oz","F9"]
    NUM_CLASS = 5
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

    cpc_model = build_iception_model((SIGNAL_LEN*FREQ,len(LEADS)), NUM_CLASS)
    cpc_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss = tf.keras.losses.BinaryCrossentropy(), metrics = [tf.keras.metrics.AUC()])

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    cpc_model.fit(x = batch_generator(batch_size=BATCH_SIZE, signal_len=SIGNAL_LEN, gen = generate_data(data_folder, filenames,seconds=SIGNAL_LEN)), 
                  epochs=EPOCHS, steps_per_epoch=len(filenames)/BATCH_SIZE, verbose=verbose)
    
    cnn_backbone = delete_last_layer(cpc_model, "feature_vector")
    reccurent_model = lstm_dnn_model()
    reccurent_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss = tf.keras.losses.BinaryCrossentropy(), metrics = [tf.keras.metrics.AUC()])
    
    cnn_features = np.zeros((num_patients,72,128))
    lstm_labels = np.zeros((num_patients,1))
    patient_features = np.zeros((num_patients,8))

    # Train the models.
    if verbose >= 1:
        print('Training the LSTM model on the Challenge data...')


    for patient_num, patient_id in enumerate(patient_ids):
        print("Patient {}".format(patient_num))
        recording_ids = np.asarray(find_recording_files(data_folder, patient_id))
        patient_metadata = load_challenge_data(data_folder, patient_id)
        patient_features[patient_num,:] = get_patient_features(patient_metadata)
        #lstm_labels[patient_num,:] = get_cpc(patient_metadata)
        lstm_labels[patient_num,:] = get_outcome(patient_metadata)
        cnt = 0
        for i in range(72):
            if cnt < len(recording_ids):
                rec_num = int(recording_ids[cnt].split("_")[-1])
                if i == rec_num:
                    try:
                        temp_recording_data, channels, sampling_frequency = load_recording_data(os.path.join(data_folder,patient_id,recording_ids[cnt] + "_EEG"))
                        if temp_recording_data.shape[1] > int(SIGNAL_LEN*sampling_frequency):
                            recording_data = temp_recording_data[:,:int(SIGNAL_LEN*sampling_frequency)]
                        elif temp_recording_data.shape[1] <= int(SIGNAL_LEN*sampling_frequency):
                            diff = int(SIGNAL_LEN*sampling_frequency) - temp_recording_data.shape[1]
                            recording_data = np.pad(temp_recording_data,((0, 0), (0, diff)), mode='constant')
                        recording_data = raw_to_bipolar(recording_data[:,:int(SIGNAL_LEN*sampling_frequency)])
                        recording_data = np.moveaxis(recording_data,0,-1)
                        recording_data = scipy.signal.resample(recording_data, int((FREQ/sampling_frequency)*recording_data.shape[0]), axis=0)
                        recording_data = add_and_restructure_eeg_leads(LEADS, channels, recording_data)
                        recording_data = recording_data[:SIGNAL_LEN*FREQ]
                    except:
                        recording_data = np.zeros((SIGNAL_LEN*FREQ,len(LEADS)))
                    cnt += len(np.where(np.char.endswith(recording_ids,str(rec_num).zfill(3)))[0])
                else:
                    recording_data = np.zeros((SIGNAL_LEN*FREQ,len(LEADS)))
                cnn_features[patient_num,i,:] = cnn_backbone(np.expand_dims(recording_data,0))
            else:
                break
    patient_features = clean_tabular_data(patient_features)
    reccurent_model.fit(x=[cnn_features,patient_features],y=lstm_labels,batch_size=LSTM_BS,epochs=LSTM_EPOCHS,verbose=verbose)


    #mem_incept = tf.keras.models.Sequential()
    #mem_incept.add(tf.keras.layers.TimeDistributed(cnn_backbone, input_shape=(72, SIGNAL_LEN * FREQ,18)))
    #mem_incept.add(reccurent_model)
    #mem_incept.add(coral.CoralOrdinal(num_classes = 5))
    #mem_incept.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss = coral.OrdinalCrossEntropy(), metrics = [coral.MeanAbsoluteErrorLabels()])
    #mem_incept = join_models(cnn_backbone,reccurent_model)
    mem_incept = join_models_2(cnn_backbone,reccurent_model)



    #memory_inception = tf.keras.models.Sequential()
    #memory_inception.add(
    #tf.keras.layers.TimeDistributed(cnn_backbone, input_shape=(10, SIGNAL_LEN,18)))
    #memory_inception.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
    #memory_inception.add(tf.keras.layers.LSTM(72, activation='relu', return_sequences=False))
    #memory_inception.add(coral.CoralOrdinal(num_classes = 5))
    #memory_inception.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = coral.OrdinalCrossEntropy(), metrics = [coral.MeanAbsoluteErrorLabels()])
    # Save the models.
    save_challenge_model(model_folder, mem_incept)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    LEADS = ["Fp1","Fp2","F7","F8","F3","F4","T3", "T4", "C3","C4","T5","T6","P3","P4","O1","O2","Fz","Cz","Pz","Fpz","Oz","F9"]
    SIGNAL_LEN = 300 
    SAMPLE_FREQ = 100 # seconds
    NUM_CLASS = 5
    cnn_model = build_iception_model((SIGNAL_LEN * SAMPLE_FREQ,len(LEADS)), NUM_CLASS)
    cnn_backbone = delete_last_layer(cnn_model, "feature_vector")
    #reccurent_model = td_lstm_model()
    reccurent_model = lstm_dnn_model()

    mem_incept = join_models_2(cnn_backbone,reccurent_model)
    filename = os.path.join(model_folder, 'model_weights.h5')
    mem_incept.load_weights(filename)
    if verbose >= 1:
        print('Model loaded...')
    return mem_incept


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    SIGNAL_LEN = 300 # sec
    FREQ = 100
    LEADS = ["Fp1","Fp2","F7","F8","F3","F4","T3", "T4", "C3","C4","T5","T6","P3","P4","O1","O2","Fz","Cz","Pz","Fpz","Oz","F9"]
    # Load data.
    #patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    patient_features = get_patient_features(patient_metadata)
    patient_features = np.nan_to_num(patient_features, nan=0)


    recordings = np.zeros((72,SIGNAL_LEN*FREQ,len(LEADS)))

    for rec_num, recording_id in enumerate(recording_ids):
        hour = int(recording_id.split("_")[-1])
        if hour > 71:
            break
        if np.all(recordings[hour,:,:]) == False:
            try:
                temp_recording_data, channels, sampling_frequency = load_recording_data(os.path.join(data_folder,patient_id,recording_id + "_EEG"))
                if temp_recording_data.shape[1] > int(SIGNAL_LEN*sampling_frequency):
                    recording_data = temp_recording_data[:,:int(SIGNAL_LEN*sampling_frequency)]
                elif temp_recording_data.shape[1] <= int(SIGNAL_LEN*sampling_frequency):
                    diff = int(SIGNAL_LEN*sampling_frequency) - temp_recording_data.shape[1]
                    recording_data = np.pad(temp_recording_data,((0, 0), (0, diff)), mode='constant')
                #recording_data = raw_to_bipolar(recording_data[:,:int(SIGNAL_LEN*sampling_frequency)])
                recording_data = np.moveaxis(recording_data,0,-1)
                recording_data = scipy.signal.resample(recording_data, int((FREQ/sampling_frequency)*recording_data.shape[0]), axis=0)
                recording_data = add_and_restructure_eeg_leads(LEADS, channels, recording_data)
                recording_data = recording_data[:SIGNAL_LEN*FREQ] 
                recordings[hour,:,:] = recording_data
            except:
                continue
        else:
            continue

    current_prediction = models.predict([np.expand_dims(recordings,0),np.expand_dims(patient_features,0)])
    outcome_probability = current_prediction.ravel()[0]
    outcome = int((outcome_probability > 0.5) * 1)
    cpc = (outcome_probability * 4) + 1
    #cpc = np.nan_to_num(cpc, nan=3.0)


    # Ensure that the CPC score is between (or equal to) 1 and 5.
    #cpc = np.clip(cpc, 1, 5)

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

    #SIGNAL_LEN = 30000 # samples
    #BATCH_SIZE = 20
    #EPOCHS = 5
    #LEARNING_RATE = 0.00001

    #BATCH_SIZE_LSTM = 30
    #EPOCHS_LSTM = 20
    #LEARNING_RATE_LSTM = 0.0001

    LEADS = ["Fp1","Fp2","F7","F8","F3","F4","T3", "T4", "C3","C4","T5","T6","P3","P4","O1","O2","Fz","Cz","Pz","Fpz","Oz","F9"]
    SIGNAL_LEN = 300 # sec
    FREQ = 100
    BATCH_SIZE = 20
    EPOCHS = 5
    LEARNING_RATE = 0.0001
    LSTM_EPOCHS = 50
    LSTM_BS = 30
    NUM_CLASS = 5

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

        cpc_model = build_iception_model((SIGNAL_LEN*FREQ,len(LEADS)), NUM_CLASS)
        cpc_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss = coral.OrdinalCrossEntropy(), metrics = [coral.MeanAbsoluteErrorLabels()])


        # Train the models.
        if verbose >= 1:
            print('Training the Challenge models on the Challenge data...')
    
        cpc_model.fit(x = batch_generator(batch_size=BATCH_SIZE, signal_len=SIGNAL_LEN, gen = generate_data(data_folder, train_filenames,seconds=SIGNAL_LEN)), epochs=EPOCHS, 
                                              validation_data=batch_generator(batch_size=BATCH_SIZE, signal_len=SIGNAL_LEN, gen = generate_data(data_folder, val_filenames,seconds=SIGNAL_LEN)),
                                              steps_per_epoch=len(train_filenames)/BATCH_SIZE, validation_steps=len(val_filenames)/BATCH_SIZE,validation_freq=1)


        cnn_backbone = delete_last_layer(cpc_model, "feature_vector")
        reccurent_model = td_lstm_model()
        reccurent_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss = coral.OrdinalCrossEntropy(), metrics = [coral.MeanAbsoluteErrorLabels()])

    
        cnn_features = np.zeros((num_patients,72,128))
        lstm_labels = np.zeros((num_patients,1))

        # Train the models.
        if verbose >= 1:
            print('Training the LSTM model on the Challenge data...')


        for patient_num, patient_id in enumerate(patient_ids):
            print("Patient {}".format(patient_num))
            recording_ids = np.asarray(find_recording_files(data_folder, patient_id))
            patient_metadata = load_challenge_data(data_folder, patient_id)
            lstm_labels[patient_num,:] = get_cpc(patient_metadata)
            cnt = 0
            for i in range(72):
                if cnt < len(recording_ids):
                    rec_num = int(recording_ids[cnt].split("_")[-1])
                    if i == rec_num:
                        try:
                            temp_recording_data, channels, sampling_frequency = load_recording_data(os.path.join(data_folder,patient_id,recording_ids[cnt] + "_EEG"))
                            if temp_recording_data.shape[1] > int(SIGNAL_LEN*sampling_frequency):
                                recording_data = temp_recording_data[:,:int(SIGNAL_LEN*sampling_frequency)]
                            elif temp_recording_data.shape[1] <= int(SIGNAL_LEN*sampling_frequency):
                                diff = int(SIGNAL_LEN*sampling_frequency) - temp_recording_data.shape[1]
                                recording_data = np.pad(temp_recording_data,((0, 0), (0, diff)), mode='constant')
                            recording_data = raw_to_bipolar(recording_data[:,:int(SIGNAL_LEN*sampling_frequency)])
                            recording_data = np.moveaxis(recording_data,0,-1)
                            recording_data = scipy.signal.resample(recording_data, int((FREQ/sampling_frequency)*recording_data.shape[0]), axis=0)
                            recording_data = add_and_restructure_eeg_leads(LEADS, channels, recording_data)
                            recording_data = recording_data[:SIGNAL_LEN*FREQ]
                        except:
                            recording_data = np.zeros((SIGNAL_LEN*FREQ,len(LEADS)))
                        cnt += len(np.where(np.char.endswith(recording_ids,str(rec_num).zfill(3)))[0])
                    else:
                        recording_data = np.zeros((SIGNAL_LEN*FREQ,len(LEADS)))
                    cnn_features[patient_num,i,:] = cnn_backbone(np.expand_dims(recording_data,0))
                else:
                    break

        reccurent_model.fit(x=cnn_features,y=lstm_labels,batch_size=LSTM_BS,epochs=LSTM_EPOCHS,verbose=verbose)


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
    #output_layer = coral.CoralOrdinal(num_classes = 5)(gap_layer)
    output_layer = tf.keras.layers.Dense(units=1,activation="sigmoid")(gap_layer)  
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
    LEADS = 22
    batch_features = np.zeros((batch_size, signal_len*FREQ, LEADS))
    batch_labels = np.zeros((batch_size, 1))

    while True:
        for i in range(batch_size):
            batch_features[i], batch_labels[i] = next(gen)
        yield batch_features, batch_labels


def generate_data(folder: str, filenames, seconds):
    FREQ = 100
    SIGNAL_LEN = 300
    LEADS = ["Fp1","Fp2","F7","F8","F3","F4","T3", "T4", "C3","C4","T5","T6","P3","P4","O1","O2","Fz","Cz","Pz","Fpz","Oz","F9"]
    while True:
        for filename in filenames:
            patient_id = get_patient_id_from_path(filename)
            patient_metadata = load_challenge_data(folder, patient_id)
            current_outcome = get_outcome(patient_metadata)
            #current_cpc = get_cpc(patient_metadata)
            try:
                temp_recording_data, channels, sampling_frequency = load_recording_data(filename)
                if temp_recording_data.shape[1] > int(seconds*sampling_frequency):
                    start_time = np.random.randint(0,temp_recording_data.shape[1]-(int(seconds*sampling_frequency)+1))
                    recording_data = temp_recording_data[:,start_time:start_time+int(seconds*sampling_frequency)]
                elif temp_recording_data.shape[1] <= int(seconds*sampling_frequency):
                    diff = int(seconds*sampling_frequency) - temp_recording_data.shape[1]
                    recording_data = np.pad(temp_recording_data,((0, 0), (0, diff)), mode='constant')

                #recording_data = raw_to_bipolar(recording_data[:,:int(seconds*sampling_frequency)])
                recording_data = np.moveaxis(recording_data,0,-1)
                recording_data = scipy.signal.resample(recording_data, int((FREQ/sampling_frequency)*recording_data.shape[0]), axis=0)
                recording_data = add_and_restructure_eeg_leads(LEADS, channels, recording_data)
                #recording_data = recording_data[:FREQ*SIGNAL_LEN] # make sure that the signal is no longer than it is supposed to be
            except:
                recording_data = np.zeros((FREQ*SIGNAL_LEN,len(LEADS)))
            yield recording_data, current_outcome

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

def join_models(cnn,lstm):
    inputs = tf.keras.Input(shape=(72,30000,22))
    x = tf.keras.layers.TimeDistributed(cnn, input_shape=(72, 30000,22))(inputs)
    out = lstm(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=out)
    return model

def td_lstm_model():
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )
    model.add(tf.keras.layers.LSTM(72, return_sequences=False)
    )
    model.add(coral.CoralOrdinal(num_classes = 5))
    return model

def demo_net():
    inputs = tf.keras.Input(shape=(8,))
    x = tf.keras.layers.Dense(32)(inputs)
    output_layer = coral.CoralOrdinal(num_classes = 5)(x) 
    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model

def lstm_dnn_model():
    inputs_1 = tf.keras.Input(shape=(72,128))
    inputs_2 = tf.keras.Input(shape=(8,))
    lstm1 =  tf.keras.layers.LSTM(128, return_sequences=True)(inputs_1)
    lstm2 = tf.keras.layers.LSTM(72, return_sequences=False)(lstm1)
    
    mod1 = tf.keras.models.Model(inputs=inputs_1, outputs=lstm2)
    
    dense = tf.keras.layers.Dense(8,name="feature_vector")(inputs_2)
    
    mod2 = tf.keras.models.Model(inputs=inputs_2, outputs=dense)
    combined = tf.keras.layers.concatenate([mod1.output, mod2.output])
    
    #output = coral.CoralOrdinal(num_classes = 5)(combined)
    output = tf.keras.layers.Dense(units=1,activation="sigmoid")(combined)  
    model = tf.keras.models.Model(inputs=[mod1.input, mod2.input], outputs=output)
    return model

def join_models_2(cnn,lstm):
    inputs1 = tf.keras.Input(shape=(72,30000,22))
    inputs2 = tf.keras.Input(shape=(8,))
    x = tf.keras.layers.TimeDistributed(cnn, input_shape=(72, 30000,22))(inputs1)
    out = lstm([x,inputs2])
    model = tf.keras.models.Model(inputs=[inputs1,inputs2], outputs=out)
    return model

def add_and_restructure_eeg_leads(reference_leads, current_leads, signal):
    missing_elements = [element for element in reference_leads if element not in current_leads]
    num_missing = len(missing_elements)
    
    new_rows = np.zeros((signal.shape[0], num_missing)) 
    new_signal = np.hstack((signal, new_rows)) 
    
    current_leads.extend(missing_elements)
    indices = [reference_leads.index(item) for item in current_leads]
    
    new_signal = new_signal[:,np.argsort(indices)]
    
    return new_signal

def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features

def nan_to_mean(arr):
    if arr.all():
        return np.zeros((len(arr)))
    else:
        mean = np.nanmean(arr)
        return np.nan_to_num(arr,nan=mean)

def nan_to_median(arr):
    if arr.all():
        return np.zeros((len(arr)))
    else:
        median = np.nanmedian(arr)
        return np.nan_to_num(arr,nan=median)

def clean_tabular_data(data):
    data[:,0] = nan_to_mean(data[:,0])
    data[:,4] = nan_to_mean(data[:,4])
    data[:,5] = nan_to_median(data[:,5])
    data[:,6] = nan_to_median(data[:,6])
    data[:,7] = nan_to_median(data[:,7])
    return data