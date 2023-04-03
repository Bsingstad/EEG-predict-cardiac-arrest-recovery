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
import pandas as pd
from io import StringIO
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
import joblib
import tensorflow as tf
from scipy import signal
from typing import Generator
from tensorflow.keras.preprocessing.sequence import pad_sequences
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # Extract features.
        current_features = get_features(patient_metadata, recording_metadata, recording_data)
        features.append(current_features)

        # Extract labels.
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)

    # Train the models.
    features = imputer.transform(features)
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # Extract features.
    features = get_features(patient_metadata, recording_metadata, recording_data)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


def cross_validate_model(data_folder, num_folds, verbose):
    # Find data files.

    SIGNAL_LEN = 30000 # samples
    DIVISOR = 10
    BATCH_SIZE = 30
    EPOCHS = 15
    LEARNING_RATE = 0.00001

    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    #os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    outcomes = list()
    cpcs = list()
    X_data = list()
    recording_id = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        for rec in recording_data:
            if rec[1] != None:
                dwn_smp_rec = []
                for lead in rec[0]:
                    dwn_smp_rec.append(signal.resample(lead,SIGNAL_LEN//DIVISOR))
                dwn_smp_rec = np.asarray(dwn_smp_rec)
                outcomes.append(get_outcome(patient_metadata))
                cpcs.append(get_cpc(patient_metadata))
                X_data.append(dwn_smp_rec)
                recording_id.append(get_patient_id(patient_metadata))

    X_data = np.asarray(X_data)
    X_data = np.moveaxis(X_data,1,-1)

    outcomes = np.expand_dims(outcomes,1)

    cpcs = np.expand_dims(cpcs,1)

    recording_id = np.asarray(recording_id)

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

    for i, (train_index, val_index) in enumerate(kf.split(np.unique(recording_id))): #TODO: Stratify based on bothe outcomes and cpcs
        print(f"Fold {i}:")

        train_id, val_id = np.unique(recording_id)[train_index], np.unique(recording_id)[val_index]

        train_indx = np.where(np.isin(recording_id,train_id))[0]
        val_indx = np.where(np.isin(recording_id,val_id))[0]

        X_train, X_val = X_data[train_indx], X_data[val_indx]
        outcomes_train, outcomes_val = outcomes[train_indx], outcomes[val_indx]
        cpcs_train, cpcs_val = cpcs[train_indx], cpcs[val_indx]

        # Define the models.

        outcome_model = build_iception_model(X_data.shape[1:], outcomes.shape[1], outputfunc="sigmoid")
        outcome_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=[
            tf.keras.metrics.AUC(num_thresholds=200,curve='ROC', summation_method='interpolation',name="ROC",multi_label=False),
            tf.keras.metrics.AUC(num_thresholds=200,curve='PR',summation_method='interpolation',name="PRC",multi_label=False)])

        cpc_model = build_iception_model(X_data.shape[1:], cpcs.shape[1], outputfunc="linear")
        cpc_model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

        tf_cpcs_train = tf.data.Dataset.from_tensor_slices((X_train, cpcs_train)) 
        tf_cpcs_train = tf_cpcs_train.cache()
        tf_cpcs_train = tf_cpcs_train.batch(BATCH_SIZE)
        tf_cpcs_train = tf_cpcs_train.prefetch(tf.data.AUTOTUNE)

        tf_cpcs_val = tf.data.Dataset.from_tensor_slices((X_val, cpcs_val))
        tf_cpcs_val = tf_cpcs_val.cache()
        tf_cpcs_val = tf_cpcs_val.batch(BATCH_SIZE)
        tf_cpcs_val = tf_cpcs_val.prefetch(tf.data.AUTOTUNE)

        tf_outcome_train = tf.data.Dataset.from_tensor_slices((X_train, outcomes_train))  
        tf_outcome_train = tf_outcome_train.cache()
        tf_outcome_train = tf_outcome_train.batch(BATCH_SIZE)
        tf_outcome_train = tf_outcome_train.prefetch(tf.data.AUTOTUNE)

        tf_outcome_val = tf.data.Dataset.from_tensor_slices((X_val, outcomes_val))
        tf_outcome_val = tf_outcome_val.cache()
        tf_outcome_val = tf_outcome_val.batch(BATCH_SIZE)
        tf_outcome_val = tf_outcome_val.prefetch(tf.data.AUTOTUNE)

        # Train the models.
        if verbose >= 1:
            print('Training the Challenge models on the Challenge data...')
    
        outcome_model.fit(tf_outcome_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=tf_outcome_val)


        cpc_model.fit(tf_cpcs_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=tf_cpcs_val)

        # Apply model on validation fold
        outcome_hat_probability = outcome_model.predict(X_val)

        outcome_hat = (outcome_hat_probability > 0.5)*1
        #print(f"outcome shape = {outcome_hat.shape}")
        #outcome_hat_probability = np.expand_dims(outcome_model.predict_proba(X_val)[:,0],1)
        #print(f"outcome_hat_probability shape = {outcome_hat_probability.shape}")
        cpc_hat = cpc_model.predict(X_val)
        #print(f"cpc_hat shape = {cpc_hat.shape}")

        # Uncomment to use regression as clf
        #outcome_hat = (cpc_hat > 3)*1
        #map_proba = map(map_regression_to_proba, cpc_hat)
        #outcome_hat_probability = np.asarray(list(map_proba))
        

        # Ensure that the CPC score is between (or equal to) 1 and 5.
        cpc_hat = np.clip(cpc_hat, 1, 5)
        all_preds_outcome.append(outcome_hat_probability)
        all_labels_outcome.append(outcomes_val)

        challenge_score[i] = compute_challenge_score(outcomes_val.ravel(),outcome_hat_probability.ravel())
        auroc_outcomes[i], auprc_outcomes[i] = compute_auc(outcomes_val.ravel(),outcome_hat_probability.ravel())
        accuracy_outcomes[i], _, _  = compute_accuracy(outcomes_val.ravel(), outcome_hat.ravel())
        f_measure_outcomes[i], _, _  = compute_f_measure(outcomes_val.ravel(), outcome_hat.ravel())
        mse_cpcs[i] = compute_mse(cpcs_val.ravel(), cpc_hat.ravel())
        mae_cpcs[i] = compute_mae(cpcs_val.ravel(), cpc_hat.ravel())
        print(f"challenge_score={challenge_score[i]},  auroc_outcomes={auroc_outcomes[i]}, auprc_outcomes={auprc_outcomes[i]},accuracy_outcomes={accuracy_outcomes[i]}, f_measure_outcomes={f_measure_outcomes[i]}, mse_cpcs={mse_cpcs[i]}, mae_cpcs={mae_cpcs[i]}")
    all_preds_outcome = np.asarray(all_preds_outcome)
    all_labels_outcome = np.asarray(all_labels_outcome)
    return all_preds_outcome , all_labels_outcome , challenge_score, auroc_outcomes, auprc_outcomes, accuracy_outcomes, f_measure_outcomes, mse_cpcs, mae_cpcs
 

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
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

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            available_signal_data.append(signal_data)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # recent recording.
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            index = i
            break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        quality_score = float('nan')

    recording_features = np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    return features

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

    output_layer = tf.keras.layers.Dense(units=nb_classes,activation=outputfunc)(gap_layer)  
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


def shuffle_batch_generator_age(
    batch_size: int,
    gen_x: Generator,
    gen_y: Generator,
    num_leads: int,
    samp_freq: int,
    time: int,
):
    batch_features = np.zeros((batch_size, samp_freq * time, num_leads))
    batch_labels_1 = np.zeros((batch_size, 1))

    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels_1[i] = next(gen_y)

        yield batch_features, batch_labels_1


def generate_y_age(y_train: np.ndarray):
    while True:
        for i in y_train:
            yield i


def generate_X_age(X_train: np.ndarray, samp_freq: int, num_leads: int):
    while True:
        for h in X_train:
            data, header_data = load_challenge_data(h)
            if int(header_data[0].split(" ")[2]) != samp_freq:
                data_new = np.ones(
                    [
                        num_leads,
                        int(
                            (
                                int(header_data[0].split(" ")[3])
                                / int(header_data[0].split(" ")[2])
                            )
                            * samp_freq
                        ),
                    ]
                )
                for i, j in enumerate(data):
                    data_new[i] = signal.resample(
                        j,
                        int(
                            (
                                int(header_data[0].split(" ")[3])
                                / int(header_data[0].split(" ")[2])
                            )
                            * samp_freq
                        ),
                    )
                data = data_new
                data = pad_sequences(
                    data, maxlen=samp_freq * 10, truncating="post", padding="post"
                )
            data = np.moveaxis(data, 0, -1)
            yield data








def batch_generator(batch_size: int, gen: Generator):
    batch_features = []
    batch_labels = []

    while True:
        for i in range(batch_size):
            X_temp, y_temp = next(gen)
            batch_features.append(X_temp)
            batch_labels.append(y_temp)

        yield np.asarray(batch_features), np.asarray(batch_labels)

def generate_data(folder: str, patient_ids: list, label: str, rec_df:pd.DataFrame):
    while True:
        for roll_indx in range(72): 
            for id in patient_ids:
                indx = np.where(np.asarray(rec_df["Record"].str.split("_").to_list())[:,1] == id.split("_")[1])[0]
                indx = np.roll(indx,-roll_indx)
                row = rec_df.T.pop(indx[0])
                rec_id = int(row["Record"].split("_")[-1])
                patient_metadata, recording_metadata, recordings = load_challenge_data(folder, id)
                X_data = []
                if recordings[rec_id][1] != None:
                    for recording in recordings[rec_id][0]:
                        S = librosa.feature.melspectrogram(y=recording, sr=recordings[rec_id][1],n_mels=128,fmin=0,fmax=recordings[rec_id][1]/2.5,n_fft=256,hop_length=128)
                        S_dB = librosa.power_to_db(S, ref=np.max)
                        X_data.append(S_dB)
                    X_data = np.asarray(X_data)
                    X_data = np.moveaxis(X_data,0,-1)
                    if label == "outcome":
                        y_data = get_outcome(patient_metadata)
                    elif label == "cpcs":
                        indx  = get_cpc(patient_metadata)
                        y_data = np.zeros(5)
                        y_data[int(indx)-1] = 1
                    else:
                        print("wrong label")

                    yield X_data, y_data

def map_regression_to_proba(pred):
    new_pred = []
    if pred <= 3:
        new_pred = pred / 6
    elif pred > 3:
        new_pred = 0.5 + (pred-3) / 4
    return new_pred