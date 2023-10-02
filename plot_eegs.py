import os
from helper_code import *
from team_code import *
from tempfile import TemporaryFile

data_folder = "../../data/physionet.org/files/i-care/2.0/training/"
LEADS = ["Fp1","Fp2","F7","F8","F3","F4","T3", "T4", "C3","C4","T5","T6","P3","P4","O1","O2","Fz","Cz","Pz","Fpz","Oz","F9"]
SIGNAL_LEN = 10
FREQ=500

patient_outcome = []
recordings = []

for patient_id in os.listdir(data_folder):
    print(patient_id)
    try:
        patient_metadata = load_challenge_data(data_folder, patient_id)
    except:
        continue
    for file_id in reversed(sorted(os.listdir(os.path.join(data_folder,patient_id)))):
        if file_id.endswith("EEG.mat"):
            temp_recording_data, channels, sampling_frequency = load_recording_data(os.path.join(data_folder,patient_id,file_id.split(".")[0]))
            #print(sampling_frequency)
            recording_data = temp_recording_data[:,:int(SIGNAL_LEN*sampling_frequency)]
            recording_data = np.moveaxis(recording_data,0,-1)
            recording_data = scipy.signal.resample(recording_data, int((FREQ/sampling_frequency)*recording_data.shape[0]), axis=0)
            recording_data = add_and_restructure_eeg_leads(LEADS, channels, recording_data)
            print(recording_data.shape)
            recordings.append(recording_data)
            break
    patient_outcome.append(get_outcome(patient_metadata))
patient_outcome = np.asarray(patient_outcome) 
recordings = np.asarray(recordings) 

np.save("patient_outcome", patient_outcome)
np.save("recordings", recordings)