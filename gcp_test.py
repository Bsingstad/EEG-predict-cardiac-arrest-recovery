import sys
from helper_code import *
from team_code import *
import os

if __name__ == '__main__':
    data_folder = "../../data/physionet.org/files/i-care/2.0/training/"
    patient = "0284"
    #ecording_ids = find_recording_files(data_folder,patient)
    #patient_ids = find_data_folders(data_folder)
    #print(patient_ids)
    recording_ids = find_recording_files(data_folder,patient)
    for id in recording_ids:
        #recording_data, _, sampling_frequency = load_recording_data(os.path.join(data_folder,patient,id+"_EEG"))
        #patient_id = get_patient_id_from_path(filename)
        patient_metadata = load_challenge_data(data_folder, patient)
        #current_outcome = get_outcome(patient_metadata)
        current_cpc = get_cpc(patient_metadata)
        recording_data, _, sampling_frequency = load_recording_data(os.path.join(data_folder,patient,id+"_EEG"))
        recording_data = raw_to_bipolar(recording_data[:,:int(60*sampling_frequency)])
        recording_data = np.moveaxis(recording_data,0,-1)
        recording_data = scipy.signal.resample(recording_data, int((100/sampling_frequency)*recording_data.shape[0]), axis=0)
        #recording_data = recording_data[:6000,:18] # stygg hardkoding her
        print(recording_data.shape)