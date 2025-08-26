import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def get_uplift_model_aucc(t, y_reward, y_cost, roi_pred, quantile=20, title='AUCC'):

    sorted_index = np.argsort(roi_pred)[::-1]

    t = t[sorted_index]
    y_reward = y_reward[sorted_index]
    y_cost = y_cost[sorted_index]
    roi_pred = roi_pred[sorted_index]

    n_t = np.sum(t)
    n_c = np.sum(~t)
    n = n_t + n_c

    
    nt_list = [0]
    nc_list = [0]

    if n_c > 0: 
        delta_reward = y_reward[t].mean() - y_reward[~t].mean()
        delta_cost = y_cost[t].mean() - y_cost[~t].mean()
        delta_cost_quantile = delta_cost / quantile
    else:
        n_c = 1
        delta_reward = y_reward[t].mean()
        delta_cost = y_cost[t].mean()
        delta_cost_quantile = delta_cost / quantile

    delta_cost_list = [0]
    delta_reward_list = [0]

    t_roi_pred_avg_list = [0]
    c_roi_pred_avg_list = [0]

    cost_t = 0
    reward_t = 0
    cost_c = 0
    reward_c = 0

    i = 0
    j = 1
    while i < n:

        if t[i]:
            cost_t += y_cost[i]
            reward_t += y_reward[i]
        else:
            cost_c += y_cost[i]
            reward_c += y_reward[i]

        if i >= n-1 or (j < quantile and cost_t / n_t - cost_c / n_c >= delta_cost_quantile * j):
            delta_cost_list.append(cost_t / n_t - cost_c / n_c)
            delta_reward_list.append(reward_t / n_t - reward_c / n_c)
            j += 1

            nt_list.append(np.sum(t[:i + 1]))
            nc_list.append(np.sum(~t[:i + 1]))

            t_roi_pred_avg_list.append(np.mean(roi_pred[:i + 1][t[:i + 1]]))
            c_roi_pred_avg_list.append(np.mean(roi_pred[:i + 1][~t[:i + 1]]))

        i += 1

    
    delta_cost_list = np.array(delta_cost_list)
    delta_reward_list = np.array(delta_reward_list)
    nt_list = np.array(nt_list)
    nc_list = np.array(nc_list)

    aucc = np.sum(delta_reward_list) / (delta_reward * (quantile + 1))

    plt.plot(delta_cost_list, delta_reward_list, color='r')
    plt.plot(delta_cost_list, delta_reward / delta_cost * delta_cost_list, color='b')

    plt.xlabel('delta cost')
    plt.ylabel('delta reward')
    plt.title(title)

    plt.show()

    df_delta_cost = pd.DataFrame(delta_cost_list)
    df_delta_reward = pd.DataFrame(delta_reward_list)
    df_nt = pd.DataFrame(nt_list)
    df_nc = pd.DataFrame(nc_list)
    df_t_roi_pred_avg = pd.DataFrame(t_roi_pred_avg_list)
    df_c_roi_pred_avg = pd.DataFrame(c_roi_pred_avg_list)

    df_delta_cost.rename(columns={0: 'delta_cost'}, inplace=True)
    df_delta_reward.rename(columns={0: 'delta_reward'}, inplace=True)
    df_nt.rename(columns={0: 'n_treatment'}, inplace=True)
    df_nc.rename(columns={0: 'n_control'}, inplace=True)
    df_t_roi_pred_avg.rename(columns={0: 'roi_pred_treatment'}, inplace=True)
    df_c_roi_pred_avg.rename(columns={0: 'roi_pred_control'}, inplace=True)

    df_aucc = pd.concat([df_delta_cost, df_delta_reward, df_nt, df_nc, df_t_roi_pred_avg, df_c_roi_pred_avg], axis=1)
    display(df_aucc)
    print("{} = ".format(title), aucc)

    return aucc, delta_cost_list, delta_reward_list, nt_list, nc_list, t_roi_pred_avg_list, c_roi_pred_avg_list


def get_model_mt_aucc(t, y_reward, y_cost, t_roi_pred, t_minus_1_roi_pred, quantile=20):

    t = np.array(t)
    y_reward = np.array(y_reward)
    y_cost = np.array(y_cost)
    t_roi_pred = np.array(t_roi_pred)
    t_minus_1_roi_pred = np.array(t_minus_1_roi_pred)

    n = len(t)

    unique_t, count_t = np.unique(t, return_counts=True)
    n_dict = dict(zip(unique_t, count_t))

    t_max = np.max(unique_t)
    t_min = np.min(unique_t)
    

    alpha = np.array([n / n_dict[t_i] for t_i in t])

    CG_t = np.zeros(shape=(np.sum(t < t_max)), dtype=bool)
    CG_y_reward = (y_reward * alpha)[t < t_max]
    CG_y_cost = (y_cost * alpha)[t < t_max]
    CG_roi_pred = t_roi_pred[t < t_max]

    TG_t = np.ones(shape=(np.sum(t > t_min)), dtype=bool)
    TG_y_reward = (y_reward * alpha)[t > t_min]
    TG_y_cost = (y_cost * alpha)[t > t_min]
    TG_roi_pred = t_minus_1_roi_pred[t > t_min]

    new_t = np.concatenate((TG_t, CG_t), axis=0)
    new_y_reward = np.concatenate((TG_y_reward, CG_y_reward), axis=0)
    new_y_cost = np.concatenate((TG_y_cost, CG_y_cost), axis=0)
    new_roi_pred = np.concatenate((TG_roi_pred, CG_roi_pred), axis=0)

    return get_uplift_model_aucc(new_t, new_y_reward, new_y_cost, new_roi_pred, quantile=quantile, title='MT-AUCC')
