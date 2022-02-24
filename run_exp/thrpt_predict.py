import numpy as np
import bisect
import sys
import matplotlib.pyplot as plt
import pandas


def is_ho_near(start_t, curr_t, ho_array):
    elapsed_t = curr_t - start_t
    next_ho_index = bisect.bisect_left(ho_array[:, 0], elapsed_t)
    is_HO = False
    for i in range(2):
        if next_ho_index + i < len(ho_array[:, 0]):
            if ho_array[next_ho_index + i, 1] != 0:
                ho_type = ho_array[next_ho_index + i, 1]
                next_ho_time = ho_array[next_ho_index + i][0]
                is_HO = True
                break
    if is_HO:
        if next_ho_time - elapsed_t < 2 and ho_type <= -1:
            return ho_type
        elif next_ho_time - elapsed_t < 2 and ho_type == 1:
            return ho_type
    return 0

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def get_error(pred_ts, pred, gt, ho_array):
    err = []
    predict, predict_near_ho = [], []
    gt_trace, gt_near_ho = [], []
    for i in range(len(pred_ts)):
        pred_tput, gt_tput = pred[i], gt[int(pred_ts[i])]
        err.append(abs(gt_tput-pred_tput)/gt_tput)
        ho_type = is_ho_near(0, pred_ts[i], ho_array)
        if ho_type == 0:
            predict.append(pred_tput)
            gt_trace.append(gt_tput)
        else:
            predict_near_ho.append(pred_tput)
            gt_near_ho.append(gt_tput)
    return np.array(err), np.array(predict), np.array(gt_trace), np.array(predict_near_ho), np.array(gt_near_ho)

def mae(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred)))


# selected_traces = os.listdir('')
selected_traces = ['BNL-WLOOP-IPERFM1-RUN2_8_239', 'BNL-DLOOP-IPERFM1-RUN3_2_239', 'BNL-DLOOP-IPERFM1-RUN3_4_239', 'BNL-WLOOP-IPERFM1-RUN2_5_239'] # , 'BNL-WLOOP-IPERFM1-RUN2_5_239'

# trace_name = sys.argv[1]

fastMPC_original, fastMPC_ho_original, fastMPC_ho = [], [], []
fastMPCHO_original = []
robustMPC_original, robustMPC_ho = [], []

for trace in selected_traces:
    original_pred, ho_pred = np.loadtxt('throughputs/thrpt_robustMPC_'+trace), np.loadtxt('throughputs/thrpt_robustMPCHO_'+trace)
    gt_thrpt = pandas.read_csv('../../tput-traces/'+trace+'.csv')
    ho_trace = pandas.read_csv('../handover_predict_traces/'+trace+'.csv')
    # ho_trace.columns = ['time', 'ho']
    ho_array = ho_trace.to_numpy()[10:]
    ho_array[:, 0] -= 10
    gt_thrpt.columns = ['time', 'tput']
    ts_original, ts_ho = original_pred[:, 0] - np.min(original_pred[:, 0]), ho_pred[:, 0] - np.min(ho_pred[:, 0])
    thrpt_pred_original, thrpt_pred_ho = original_pred[:, 1] * 8.0, ho_pred[:, 1] * 8.0
    gt_ts = gt_thrpt['time'].to_numpy()[10:] - 10 # sync
    gt_tput = gt_thrpt['tput'].to_numpy()[10:]
    original_err, original_pred, original_gt, original_near_ho, original_gt_near_ho = get_error(ts_original[:30], thrpt_pred_original[:30], gt_tput, ho_array)
    ho_pred_err, ho_pred, ho_gt, near_ho, gt_near_ho = get_error(ts_ho[:30], thrpt_pred_ho[:30], gt_tput, ho_array)
    # ho_pred, ho_gt = ho_pred[ho_pred_err < 2], ho_gt[ho_pred_err < 2]
    
    fastMPC_original.append(mae(original_gt, original_pred))
    fastMPC_ho_original.append(mae(original_gt_near_ho, original_near_ho))
    fastMPC_ho.append(mae(gt_near_ho, near_ho))
    fastMPCHO_original.append(mae(ho_gt, ho_pred))
    print(mae(original_gt, original_pred), mae(ho_gt, ho_pred))

    # original_pred, ho_pred = np.loadtxt('throughputs/thrpt_robustMPC_'+trace), np.loadtxt('throughputs/thrpt_robustMPCHO_'+trace)
    # gt_thrpt = pandas.read_csv('../../tput-traces/'+trace+'.csv')
    # gt_thrpt.columns = ['time', 'tput']
    # ts_original, ts_ho = original_pred[:, 0] - np.min(original_pred[:, 0]), ho_pred[:, 0] - np.min(ho_pred[:, 0])
    # thrpt_pred_original, thrpt_pred_ho = original_pred[:, 1] * 8.0, ho_pred[:, 1] * 8.0
    # gt_ts = gt_thrpt['time'].to_numpy()[10:] - 10 # sync
    # gt_tput = gt_thrpt['tput'].to_numpy()[10:]
    # original_err, original_pred, original_gt = get_error(ts_original[:30], thrpt_pred_original[:30], gt_tput)
    # ho_pred_err, ho_pred, ho_gt = get_error(ts_ho[:30], thrpt_pred_ho[:30], gt_tput)
    # # ho_pred, ho_gt = ho_pred[ho_pred_err < 2], ho_gt[ho_pred_err < 2]
    
    # robustMPC_original.append(mae(original_gt, original_pred))
    # robustMPC_ho.append(mae(ho_gt, ho_pred))

print(np.mean(fastMPC_original), np.std(fastMPC_original), np.mean(fastMPC_ho), np.std(fastMPC_ho), np.mean(fastMPC_ho_original), np.std(fastMPC_ho_original), np.mean(fastMPCHO_original), np.std(fastMPCHO_original))
# print(np.mean(robustMPC_original), np.std(robustMPC_original), np.mean(robustMPC_ho), np.std(robustMPC_ho),  np.mean(robustMPC_ho)/np.mean(robustMPC_original))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar([0], [np.mean(fastMPC_original)], 0.35, color='none', edgecolor='darkred', ecolor='darkred', hatch='/', linewidth=2, yerr=np.std(fastMPC_original), error_kw=dict(lw=2.5, capsize=5, capthick=2))
plt.bar([0.5], [np.mean(fastMPC_ho_original)], 0.35, color='none', edgecolor='darkblue', ecolor='darkblue', hatch='o', linewidth=2, yerr=np.std(fastMPC_ho_original), error_kw=dict(lw=2.5, capsize=5, capthick=2))
plt.bar([1], [np.mean(fastMPC_ho)], 0.35, color='none', edgecolor='darkgreen', ecolor='darkgreen', hatch='x', linewidth=2, yerr=np.std(fastMPC_ho), error_kw=dict(lw=2.5, capsize=5, capthick=2))
plt.ylabel('Mean Avg. Error (Mbps)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(['no HO\n(fastMPC)', 'during HO\n(fastMPC)', 'during HO\n(fastMPCHO)'], fontsize=18)
plt.tight_layout()
plt.grid(linestyle='--', axis='y')
plt.show()

summary_dict = {'w/o HO\n(fastMPC)': [np.mean(fastMPC_original), np.std(fastMPC_original)], 'w/ HO\n(fastMPC)': [np.mean(fastMPC_ho_original), np.std(fastMPC_ho_original)],
    'w/ HO\n(fastMPC-PR)': [np.mean(fastMPC_ho), np.std(fastMPC_ho)], 'w/o HO\n(fastMPC-PR)': [np.mean(fastMPCHO_original), np.std(fastMPCHO_original)],
    }

# import json
# with open('predict_summary.json', 'w') as f:
#     json.dump(summary_dict, f)