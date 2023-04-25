from evaluate_model import *
from team_code import *
from run_model import *
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import KFold

font = {'size'   : 16}

matplotlib.rc('font', **font)

data_folder = "./training_subset/"



all_preds_outcome , all_labels_outcome , challenge_score, auroc_outcomes, auprc_outcomes, accuracy_outcomes, f_measure_outcomes, mse_cpcs, mae_cpcs = cross_validate_model("./training_subset/",5,1)

np.savetxt('./score.txt', challenge_score, delimiter=',')