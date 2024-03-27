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
    
    dope_files = sorted(glob("DecodedDOPEProteinHRun1.csv"))
    
    #assign dataset label
    df_input_sets = [pd.read_csv(file, header=None).assign(Row_Index=lambda x: x.index, Dataset='input') for file in input_files]    
    df_dope_trains = [pd.read_csv(file, header=None).assign(Row_Index=lambda x: x.index, Dataset='decoded') for file in dope_files]
    
    # Concatenate the DataFrames vertically
    result_df = pd.concat(df_input_sets + df_dope_trains, ignore_index=True)
    
    # Rename columns for clarity
    result_df.columns = ['DOPE score', 'time', 'Dataset']
    
    # Filter out rows with 'dope_abs_test' or 'dope_abs_train' in the 'DOPE score' column
    result_df = result_df.loc[~result_df['DOPE score'].isin(['dope_abs_test', 'dope_abs_train'])]
    
    # Drop rows with NaN values in the 'DOPE score' column
    result_df = result_df.dropna(subset=['DOPE score'])
    
    # Convert 'DOPE score' column to numeric, handling errors with 'coerce'
    result_df['DOPE score'] = pd.to_numeric(result_df['DOPE score'], errors='coerce')
    
    
    # Plotting two separate lineplots for "input" and "training"
    plt.figure(figsize=(10,10))
    f=sns.lineplot(data=result_df, y='DOPE score', x='time', hue='Dataset', style='Dataset', markers=None, dashes=False)
    
    #insert line for average RMSD of extended conformation
    decoded_scores_extended = result_df[result_df['Dataset'] == 'decoded']['DOPE score'].iloc[:188]
    mean_score_extended = np.mean(decoded_scores_extended)
    plt.hlines(y=mean_score_extended, xmin=result_df['time'].min(), xmax=188, ls='--', color='dimgrey')
    plt.scatter([result_df['time'].min(), 188], [mean_score_extended, mean_score_extended], color='dimgrey', zorder=5)  # Adding ticks at the extremes

    #insert line for average RMSD of compact conformation
    decoded_scores_compact = result_df[result_df['Dataset'] == 'decoded']['DOPE score'].iloc[188:]
    mean_score_compact = np.mean(decoded_scores_compact)
    plt.hlines(y=mean_score_compact, xmin=188, xmax=result_df['time'].max(), ls='--', color='lightgrey')
    plt.scatter([188, result_df['time'].max()], [mean_score_compact, mean_score_compact], color='lightgrey', zorder=5)  # Adding ticks at the extremes
    
    plt.ylabel('DOPE score')
    plt.xlabel('time [ns]')
    f.set_ylim(ymin=-4000, ymax=35000)
    
    # Manually setting legend labels and colors
    legend_labels = ['Input', 'Decoded']
    legend_colors = ['blue', 'orange']
    legend = plt.legend(labels=legend_labels, fontsize=14, loc='upper right', title='Dataset', title_fontsize='14', fancybox=True, framealpha=1, facecolor='white', edgecolor='black', markerscale=1.5)
    for i, (text, color) in enumerate(zip(legend_labels, legend_colors)):
        legend.legendHandles[i].set_color(color)

    # Removing the legend title
    legend.get_title().set_visible(False)
    
    plt.savefig('DOPEVStProteinH4Report.jpg', dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    main()
