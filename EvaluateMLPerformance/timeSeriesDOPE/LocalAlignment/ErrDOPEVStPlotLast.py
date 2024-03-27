import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import sys, os

#uncomment/edit as appropriate for your system
#sys.path.insert(0, f"C:{os.sep}Users{os.sep}xdzl45{os.sep}workspace{os.sep}molearn{os.sep}src") 

import molearn
from molearn.analysis import MolearnAnalysis
from molearn.data import PDBData
from molearn.models.foldingnet import AutoEncoder

import torch
from copy import deepcopy
import biobox as bb
import numpy as np

sns.set_theme(style='ticks')
sns.set_context("poster")

def main():
    
    input_files = sorted(glob("Input*.csv"))    # Read CSV files and add a new column with the row index and "Dataset" label
    df_input_sets = [pd.read_csv(file, header=None).rename(columns={0: 'DOPE'}).assign(TimeFrame=lambda x: x.index, Dataset='Input', Error=None) for file in input_files]
    
    dope_files = sorted(glob("DecodedDOPEOneHelix*.csv"))
    dope_dataframes = {file: pd.read_csv(file, header=None) for file in dope_files}
    dope_concatenated = pd.concat(dope_dataframes, keys=dope_files, axis=1, ignore_index=True)

    dope_std = dope_concatenated.std(axis=1)
    err_dope = dope_std/np.sqrt(len(dope_files))

    # Create a DataFrame for DOPE
    df_dope_set = pd.DataFrame({
    'TimeFrame': df_input_sets[0]['TimeFrame'],  # Assuming the 'TimeFrame' is the same for input and DOPE
    'Dataset': 'Decoded',
    'Error': err_dope
    })
    result_df = pd.concat([pd.concat(df_input_sets), df_dope_set], ignore_index=True)
    
    result_df.to_csv('CheckingResult.csv') #here you can check that there is nothing anomalous with your result
    
    # Plotting two separate lineplots for "input" and "training"
    plt.figure(figsize=(10,10))
    f=sns.lineplot(data=result_df, y='Error', x='TimeFrame', dashes=False)
    
    #insert line for average RMSD of extended conformation
    decoded_scores_extended = result_df[result_df['Dataset'] == 'Decoded']['Error'].iloc[:200]
    mean_score_extended = np.mean(decoded_scores_extended)
    plt.hlines(y=mean_score_extended, xmin=result_df['TimeFrame'].min(), xmax=200, ls='--', color='darkorange')
    plt.scatter([result_df['TimeFrame'].min(), 200], [mean_score_extended, mean_score_extended], color='darkorange', zorder=5)  # Adding ticks at the extremes

    #insert line for average RMSD of compact conformation
    decoded_scores_compact = result_df[result_df['Dataset'] == 'Decoded']['Error'].iloc[200:]
    mean_score_compact = np.mean(decoded_scores_compact)
    plt.hlines(y=mean_score_compact, xmin=200, xmax=result_df['TimeFrame'].max(), ls='--', color='navajowhite')
    plt.scatter([200, result_df['TimeFrame'].max()], [mean_score_compact, mean_score_compact], color='navajowhite', zorder=5)  # Adding ticks at the extremes

    
    
    f.set_ylim(ymin=0, ymax=6000)
    f.set_ylabel('Standard error on the DOPE score')
    f.set_xlabel('time [ns]')
    plt.savefig('StandardErrDOPEVStOneHelix4Report.jpg', dpi=300, bbox_inches='tight')
    

if __name__ == "__main__":
    main()