from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pandas as pd

def get_eer_table(results_df, bitrate=None):
    """
    results_df (pd DataFrame) : dataframe from calculate_tamper_score function
    bitrate (str) : audio bitrate out of [64k, 128k, 256k]
    
    returns
        df (pd DataFrame) : equal error rates split by tamper type and tamper length
    """
    labels = ['INS','DEL','REP','aggregate']
    tamperlens = [4, 2, 1, 0.5, 0.25, 'aggregate']
    
    if bitrate is not None:
        results_df = results_df[results_df["bitrate"] == bitrate].copy()
    
    # add truth labels
    results_df.loc[results_df['type'] != 'NONE', 'truth'] = 1
    results_df.loc[results_df['type'] == 'NONE', 'truth'] = 0

    total = {'tamper_len':tamperlens, 'INS':[] ,'DEL':[] ,'REP':[] ,'aggregate':[]}
    for label in labels:
        cols = []
        if label != 'aggregate':
            lab = results_df[(results_df["type"] == label) | (results_df["type"] == 'NONE')]
        else:
            lab = results_df
        
        for lens in tamperlens:
            if lens != 'aggregate':
                len_lab = lab[(lab['len'] == lens) | (lab['len'] == 0)]
            else:
                len_lab = lab
            
            fpr, tpr, thresholds = roc_curve(len_lab['truth'], len_lab['score'])
            eer = calculate_eer(fpr, tpr)
            cols.append(eer)
            
        total[label]=cols
        
    df = results_df.astype(str)
    df = pd.DataFrame(data=total)
    display(df) 
    
    return df


def calculate_eer(fpr, tpr):
    '''
    requires fpr, tpr output from roc_curve (sklearn.metrics)
    Returns the equal error rate for a binary classifier output.
    '''
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    return eer*100
