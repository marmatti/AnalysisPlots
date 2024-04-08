import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob


def main():
    
    # Define file patterns
    err_test_files = sorted(
    glob("../run*/analysis/err_test*.csv"),
    key=lambda x: ("alignToOneHelix" not in x, x))
    
    # Here sorted is used to allow consistency in how the files are read (in ascending order)
    # Lambda function used to ensure file paths containing "alignToOneHelix" are read in first
    err_train_files = sorted(
    glob("../run*/analysis/err_train*.csv"),
    key=lambda x: ("alignToOneHelix" not in x, x))
    
    df_err_tests = [pd.read_csv(file, header=None) for file in err_test_files]
    df_err_trains = [pd.read_csv(file, header=None) for file in err_train_files]
  
    width = 2
    
    # Here X is used to correctly space the two sectors (local alignment, global alignment) on the x axis
    df_err_test_combined = pd.concat([df.assign(**{'Data split': 'validation set', 'X': i + 2*width if i > 4 else i + width}) for i, df in enumerate(df_err_tests)], ignore_index=True)
    df_err_train_combined = pd.concat([df.assign(**{'Data split': 'training set', 'X': i + 2*width if i > 4 else i + width}) for i, df in enumerate(df_err_trains)], ignore_index=True)
 
    # Concatenate the DataFrames vertically
    result_df = pd.concat([df_err_test_combined, df_err_train_combined], ignore_index=True)
    
    result_df.columns = ['RMSD', 'Data split', 'X']
    
    # In this block we are taking care of possible residual strings in the data
    result_df = result_df.loc[result_df['RMSD'] != 'err_test']
    result_df = result_df.loc[result_df['RMSD'] != 'err_train']
    
    color = {'training set':'#66c2a5', 'validation set':'#fc8d62'}

    custom_order_hue = ['training set', 'validation set'] # Setting different entries for different datasets in the legend
    
    result_df['RMSD'] = pd.to_numeric(result_df['RMSD'], errors='coerce') # Fixes NaN problem
    
    result_df.to_csv('CheckingResultRMSD.csv')

    #y_max = result_df['RMSD'].max()  # Set the limit to the maximum absolute value
    y_max = 10
    y_min = result_df['RMSD'].min()
    
    # Here hue is for the different distributions in the violins, split is used to correctly set the sectors in the x axis
    violin = sns.catplot(
    data=result_df, x="X", y="RMSD", hue="Data split", palette = color,
    kind="violin", split=True, hue_order=custom_order_hue, height=6, aspect=2, width=0.9, native_scale=True
    )
    ax = violin.ax
   
    ax.set_ylim(ymax=y_max, ymin=min(y_min, 0))
    
    ax.set_xlabel('') # Getting rid of numbers on ticks
    
    ax.set_ylabel('RMSD [$\AA$]', fontsize=22)
    
    violin._legend.remove()

    # Add custom legend for "input set" and "input set2"
    custom_legend = [plt.Line2D([0], [0], color='#66c2a5', marker='o', linestyle='', markersize=8),
                     plt.Line2D([0], [0], color='#fc8d62', marker='o', linestyle='', markersize=8)]
    custom_legend_labels = ['training set', 'validation set']
    ax.legend(custom_legend, custom_legend_labels, fontsize=22, bbox_to_anchor=(1.05, 0.5), loc='center left')

    
    
    labels = ['','','local alignment','','','','','global alignment','',''] # Should be same number of objects as labels
    ax.set_xticks([2,3,4,5,6,9,10,11,12,13],labels, fontsize=22) # This is to position labels
    ax.axvline(x=6+(9-6)/2, ls='--', color='black')
    ax.tick_params(axis=u'both', which=u'both',length=0,labelsize=22)

    
    violin.savefig('ViolinRMSDSector.jpg', dpi=300)
main()

