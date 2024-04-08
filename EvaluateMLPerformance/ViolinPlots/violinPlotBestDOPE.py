import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob


def main():
    

    # Define file patterns
    input_files = sorted(glob("Input*.csv")) #here sorted is probably not completely necessary
    
    # Here sorted is used to allow consistency in how the files are read (in ascending order)
    # Lambda function used to ensure file paths containing "alignToOneHelix" are read in first
    dope_test_files = sorted( 
    glob("../run*/analysis/dope_test*.csv"),
    key=lambda x: ("alignToOneHelix" not in x, x))

    dope_train_files = sorted(
    glob("../run*/analysis/dope_train*.csv"),
    key=lambda x: ("alignToOneHelix" not in x, x))
    
    df_input_sets = [pd.read_csv(file, header=None) for file in input_files]
    df_dope_tests = [pd.read_csv(file, header=None) for file in dope_test_files]
    df_dope_trains = [pd.read_csv(file, header=None) for file in dope_train_files]
    
    # This line is placing the data in the correctly labeled column, where X is used to correctly space the three sectors (input, local alignment, global alignment) on the x axis
    # The input set label is differet for the two types of alignment although the data is the same because we want to plot a symmetric violin and this is not possible if there are two input sets labelled the same way
    df_input_sets_combined = pd.concat([df.assign(**{'Data split': f'input se{"t" if i == 0 else "t2"}', 'X': 1}) for i, df in enumerate(df_input_sets)], ignore_index=True)    
    
    width = 2
    
    # Here X is used to correctly space the three sectors (local alignment, global alignment) on the x axis
    df_dope_test_combined = pd.concat([df.assign(**{'Data split': 'validation set', 'X': i + 2*width if i > 4 else 1+i + width}) for i, df in enumerate(df_dope_tests)], ignore_index=True)
    df_dope_train_combined = pd.concat([df.assign(**{'Data split': 'training set', 'X': i + 2*width if i > 4 else 1+i + width}) for i, df in enumerate(df_dope_trains)], ignore_index=True)
 
    # Concatenate the DataFrames vertically
    result_df = pd.concat([df_input_sets_combined, df_dope_test_combined, df_dope_train_combined], ignore_index=True)
    
    result_df.columns = ['DOPE score', 'Data split', 'X']
    
    # In this block we are taking care of possible residual strings in the data
    result_df = result_df.loc[result_df['DOPE score'] != 'dope_abs_test']
    result_df = result_df.loc[result_df['DOPE score'] != 'dope_abs_train']
    result_df = result_df.dropna(subset=['DOPE score']) # Not sure if this is necessary
    
    result_df.to_csv('CheckingResult.csv') # Good practice to check the resulting dataframe
    
    color = {'training set':'#66c2a5', 'validation set':'#fc8d62', 'input set':'#8da0cb', 'input set2':'#8da0cb'}

    custom_order_hue = ['training set', 'validation set', 'input set', 'input set2'] # Setting different entries for different datasets in the legend
    
    result_df['DOPE score'] = pd.to_numeric(result_df['DOPE score'], errors='coerce') #fixes NaN problem
    
    y_max = result_df['DOPE score'].max()  # Set the limit to the maximum absolute value
    
    y_min = result_df['DOPE score'].min()
    
    #print(sns.color_palette("Pastel1").as_hex()) # This line is used to print out hexadecimal values for a specific colour scheme so that it can be assigned to the datasets
    
    # Here hue is for the different distributions in the violins, split is used to correctly set the sectors in the x axis
    violin = sns.catplot(
    data=result_df, x="X", y="DOPE score", hue="Data split", palette = color,
    kind="violin", split=True, height=6, aspect=2, width=1.5, native_scale=True, hue_order=custom_order_hue)
        
    ax = violin.ax

    ax.set_ylim(ymax=y_max, ymin=y_min)
    
    ax.set_xlabel('')
    
    ax.set_ylabel('DOPE score', fontsize=22)
    #ax.legend(fontsize=14, loc='upper right')
    
    # Create a combined legend entry for "input set" and "input set2"
    legend_labels = {'input set2': 'input set'}
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(set(labels))
    combined_labels = [legend_labels.get(label, label) for label in unique_labels]
    combined_handles = [handles[labels.index(label)] for label in unique_labels]

    # Update the legend
    ax.legend(combined_handles, combined_labels, fontsize=22, bbox_to_anchor=(1.05, 0.5), loc='center left')

    violin._legend.remove()

    # Add custom legend for "input set" and "input set2"
    custom_legend = [plt.Line2D([0], [0], color='#66c2a5', marker='o', linestyle='', markersize=8),
                     plt.Line2D([0], [0], color='#fc8d62', marker='o', linestyle='', markersize=8),
                     plt.Line2D([0], [0], color='#8da0cb', marker='o', linestyle='', markersize=8)]
    custom_legend_labels = ['training set', 'validation set','input set']
    ax.legend(custom_legend, custom_legend_labels, fontsize=22, bbox_to_anchor=(1.05, 0.5), loc='center left')

    
    labels = ['input','','','local alignment','','','','','global alignment','',''] # Should be same number of objects as labels
    ax.set_xticks([1.35,2.75,3.75,4.75,5.75,6.75,8.75,9.75,10.75,11.75,12.75],labels, fontsize=22) #this is to position labels
    ax.axvline(x=1.25+(2.75-1.25)/2, ls='--', color='black')
    ax.axvline(x=6.75+(8.75-6.75)/2, ls='--', color='black')
    ax.tick_params(axis=u'both', which=u'both',length=0,labelsize=22)

    
    violin.savefig('ViolinDOPESector.jpg', dpi=300)
main()


    