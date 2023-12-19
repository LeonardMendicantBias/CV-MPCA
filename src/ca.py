#%%
import numpy as np
from itertools import combinations as CWOR
from itertools import combinations_with_replacement as CWR
import copy

def path_gain(dist, freq):      # dist in meters, freq in GHz
    c = 3e8
    return 10**(-(32.4+17.3*np.log10(dist)+20*np.log10(freq))/10), \
           10**(-(32.4+31.9*np.log10(dist)+20*np.log10(freq))/10), \
           10**(-(31.84+21.50*np.log10(dist)+19*np.log10(freq))/10), \
           (c / (4*np.pi*dist*freq*1e9))**2

def get_rate(channel, ant_num, noise_level):
    snr = np.sum(np.abs(channel)**2, axis=0)/noise_level
    return np.log2(1+snr)

def cell_association(distance, vis, freq, V_max, ant_num, noise_level):  # freq=100e9
    M, K = distance.shape
    channel = path_gain(distance, freq/1e9)[-1]*vis
    
    all_combs = []
    for i in range(V_max, V_max + 1):
        all_combs += list(CWOR(np.arange(K),i))
    
    possible_asso_num = len(all_combs)
    all_asso = np.zeros([possible_asso_num, K])
    for i in range(possible_asso_num):
        all_asso[i, all_combs[i]] = 1

    count = np.zeros(M, np.int)

    cell_asso = np.zeros([M, K])
    
    ca = CWR(possible_asso_num, M)

    for index in range(possible_asso_num**M):
        count = ca[index]
        for m in range(M):
            cell_asso[m] = all_combs[count[m]]
        channel_ = cell_asso*channel
        R = sum_rate(channel_, ant_num, noise_level)
        if R > max_rate:
            max_cell_asso = copy.deepcopy(cell_asso)
            max_rate = R

    return max_cell_asso

def cell_association(required_rate, distance, vis, freq, V_max, ant_num, noise_level):  # freq=100e9
    M, K = distance.shape
    channel = path_gain(distance, freq/1e9)[-1]*vis

    _temp = np.zeros(K)
    # association options for each SBS given V_max and K
    options = []
    for ids in CWOR(np.arange(K), 2):
        temp = _temp.copy()
        temp[list(ids)] = 1
        options.append(temp)

    # among all association options among M SBSs, find the best one
    best_ca = None
    max_rate = -1
    for ca in CWR(options, M):
        channel_ = np.array(ca)*channel
        rates = get_rate(required_rate, channel_, ant_num, noise_level)
        if (R:=rates.sum()) > max_rate and (rates>=required_rate).all():
            max_rate = R
            best_ca = ca

    return best_ca


#%%
if __name__ == '__main__':
    M = 4
    K = 5
    max_asso = 2
    
    all_combs = []
    for i in range(max_asso+1):
        all_combs += list(CWOR(np.arange(K),i))
    
    
    possible_asso_num = len(all_combs)
    all_asso = np.zeros([possible_asso_num, K])
    for i in range(possible_asso_num):
        all_asso[i, all_combs[i]] = 1

    