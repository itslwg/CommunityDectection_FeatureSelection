import numpy as np
from numba import njit


@njit(fastmath=True)
def utility_score(date, weight, resp, action):
    """Calculate the utility score.
    
    Source: 
        User Calibrator.
        December 7th, 2020.
        Avaliable at: https://www.kaggle.com/c/jane-street-market-prediction/discussion/201257
    """
    Pi = np.bincount(date, weight * resp * action)
    count_i = len(Pi)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = min(max(t, 0), 6) * np.sum(Pi)
    return u