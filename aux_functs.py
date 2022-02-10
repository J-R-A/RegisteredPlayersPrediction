import pandas as pd
import numpy as np

# Function to bootsrap correlation coefficients 
def boot_corrs(df, feat_1, feat_2,n_its):
   
   # inputs: 
   # df: df with features
   # feat_1: name of first feature
   # feat_2: name of second feature
   # n_its: number or resamples 
   # outputs:
   # corrs: correlation coefficient per sample
    
    corrs = np.zeros(n_its) # vector to store correlations
    for i in range(n_its):
        # get sample with replacement from selected features
        new_df = df[[feat_1,feat_2]].sample(df[feat_1].shape[0],replace=True,random_state = i).copy()
        # get correlation coefficient between selected features
        corrs[i] = np.corrcoef(new_df[feat_1],new_df[feat_2])[0,1]
        
    return corrs