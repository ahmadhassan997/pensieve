import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import scipy.stats

font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)

# matplotlib.use('Agg')

RESULTS_FOLDER = './results_ho_no_selected/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 61
VIDEO_BIT_RATE = [4900, 8200, 11700, 32800, 152400, 260000]  # Kbps
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired 
SIM_DP = 'sim_dp'
SCHEMES = ['fastMPCHO', 'robustMPCHO', 'robustMPC', 'fastMPC', 'RB', 'fastMPCGT', 'robustMPCGT', 'RBHO', 'RBHOGT'] # 'RB', 'BOLA', 'FESTIVE',
# SCHEMES = ['RL']

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def main():
    time_all = {}
    bit_rate_all = {}
    buff_all = {}
    stall_all = {}
    bw_all = {}
    raw_reward_all = {}
    summay_all = {}
    for scheme in SCHEMES:
        time_all[scheme] = {}
        raw_reward_all[scheme] = {}
        bit_rate_all[scheme] = {}
        buff_all[scheme] = {}
        stall_all[scheme] = {}
        bw_all[scheme] = {}

    stall_total = 0.0
    log_files = os.listdir(RESULTS_FOLDER)
    for log_file in log_files:

        time_ms = []
        bit_rate = []
        buff = []
        bw = []
        stall = []
        reward = []

        print log_file

        with open(RESULTS_FOLDER + log_file, 'rb') as f:
            if SIM_DP in log_file:
                for line in f:
                    parse = line.split()
                    if len(parse) == 1:
                        reward = float(parse[0])
                    elif len(parse) >= 6:
                        time_ms.append(float(parse[3]))
                        bit_rate.append(VIDEO_BIT_RATE[int(parse[6])])
                        buff.append(float(parse[4]))
                        bw.append(float(parse[5]))

            else:
                for line in f:
                    try:
                        parse = line.split()
                        if len(parse) <= 1:
                            break
                        time_ms.append(float(parse[0]))
                        bit_rate.append(int(parse[1]))
                        buff.append(float(parse[2]))
                        stall.append(float(parse[3]))
                        stall_total += float(parse[3])
                        bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                        reward.append(float(parse[6]))
                    except ZeroDivisionError:
                        bw.append(float(parse[4]) / (float(parse[5])+1) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                        reward.append(float(parse[6]))

        if SIM_DP in log_file:
            time_ms = time_ms[::-1]
            bit_rate = bit_rate[::-1]
            buff = buff[::-1]
            bw = bw[::-1]
        
        time_ms = np.array(time_ms)
        time_ms -= time_ms[0]
        
        # print log_file

        for scheme in SCHEMES:
            if scheme in log_file :
                if scheme == 'fastMPC' and ('HO' in log_file or 'GT' in log_file):
                    continue
                if scheme == 'robustMPC' and ('HO' in log_file or 'GT' in log_file):
                    continue
                if scheme == 'RB' and ('HO' in log_file or 'GT' in log_file):
                    continue
                if scheme == 'RBHO' and ('GT' in log_file):
                    continue
                time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
                bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
                buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
                stall_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = stall
                bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
                raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
                break
    
    print(raw_reward_all)
    # calculate mean bitrate and stall rate
    scheme_reward = {}
    for scheme in SCHEMES:
        reward = []
        summay_all[scheme] = {}
        # avg bitrate
        bitrate_avg_arr = []
        bitrate_total = 0
        total_chunk = 0
        summay_all[scheme]['all_br'] = []
        for log in bit_rate_all[scheme]:
            reward += raw_reward_all[scheme][log][:VIDEO_LEN]
            if len(bit_rate_all[scheme][log]) >= VIDEO_LEN:
                bitrate_avg_arr.append(np.mean(bit_rate_all[scheme][log][:VIDEO_LEN])/1000)
                bitrate_total += np.sum(np.array(bit_rate_all[scheme][log][:VIDEO_LEN])/1000) #Mbis
                total_chunk += VIDEO_LEN
            else:
                bitrate_avg_arr.append(np.mean(bit_rate_all[scheme][log])/1000)
                bitrate_total += np.sum(np.array(bit_rate_all[scheme][log])/1000) #Mbis
                total_chunk += len(bit_rate_all[scheme][log])
        # print(scheme)
        # print(bitrate_total)
        # print(total_chunk)
        # print(float(bitrate_total)/total_chunk)
        # print(bitrate_avg_arr)
        print(scheme, 'reward:', np.mean(reward))
        scheme_reward[scheme] = np.mean(reward)
        summay_all[scheme]['avg_br'] = np.mean(bitrate_avg_arr)
        summay_all[scheme]['all_br'] = bitrate_avg_arr

        # avg stall rate
        stall_time_all = float(0)
        total_playback_time = 0
        avg_stall_arr = []
        summay_all[scheme]['all_stalled_rate'] = []
        for log in stall_all[scheme]:
            if len(stall_all[scheme][log]) >= VIDEO_LEN:
                avg_stall_arr.append(np.sum(np.array(stall_all[scheme][log][:VIDEO_LEN]))/(158+np.sum(np.array(stall_all[scheme][log][:VIDEO_LEN]))))
                stall_time_all += float(np.sum(np.array(stall_all[scheme][log][:VIDEO_LEN], dtype=float))) # seconds
                total_playback_time = total_playback_time + 158 + np.sum(np.array(stall_all[scheme][log][:VIDEO_LEN]))
            else:
                avg_stall_arr.append(np.sum(np.array(stall_all[scheme][log]))/(len(stall_all[scheme][log])+np.sum(np.array(stall_all[scheme][log]))))
                stall_time_all += float(np.sum(np.array(stall_all[scheme][log], dtype=float))) # seconds
                total_playback_time = total_playback_time + len(stall_all[scheme][log]) + np.sum(np.array(stall_all[scheme][log]))
        print("total stall time %f" % stall_time_all)
        print("total playback time %f" % total_playback_time)
        # print("pure video time %f" % pure_video_time)
        summay_all[scheme]['time_stalled_rate'] = np.mean(avg_stall_arr) * 100 # percentage
        summay_all[scheme]['all_stalled_rate'] = np.array(avg_stall_arr) * 100
        # print(summay_all[scheme]['time_stalled_rate'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(2, 0)
    plt.xticks(np.arange(2.0, 0.0, -0.5))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    result_summary = {}
    for scheme in SCHEMES:
        # ax.scatter(summay_all[scheme]['time_stalled_rate'], summay_all[scheme]['avg_br'])
        print("mean for scheme %s are %f, %f" % (scheme, np.mean(summay_all[scheme]['all_stalled_rate']), np.mean(summay_all[scheme]['all_br'])))
        # print("confidence interval scheme %s are %f, %f" % (scheme, m_x-left_m_x, m_y-left_m_y))
        # print("stderr for scheme %s are %f, %f" % (scheme, np.std(summay_all[scheme]['all_stalled_rate']), np.std(summay_all[scheme]['all_br'])))
        m_x, left_m_x, right_m_x = mean_confidence_interval(summay_all[scheme]['all_stalled_rate'])
        # if (left_m_x != right_m_x):
        #     print("not equal!")
        #     print(m_x-left_m_x)
        #     print(right_m_x-m_x)
        #     print(m_x)
        m_y, left_m_y, right_m_y = mean_confidence_interval(summay_all[scheme]['all_br'])
        print("confidence interval scheme %s are %f, %f" % (scheme, m_x-left_m_x, m_y-left_m_y))
        print("variance scheme %s are %f, %f" % (scheme, np.std(summay_all[scheme]['all_stalled_rate']), np.std(summay_all[scheme]['all_br'])))
        ax.errorbar(summay_all[scheme]['time_stalled_rate'], summay_all[scheme]['avg_br'], xerr=m_x-left_m_x, yerr=m_y-left_m_y, capsize=3)
        ax.scatter(summay_all[scheme]['time_stalled_rate'], summay_all[scheme]['avg_br'])
        result_summary[scheme] = [summay_all[scheme]['time_stalled_rate'], summay_all[scheme]['avg_br']/260, m_x-left_m_x, (m_y-left_m_y)/260]
        if scheme == "interfaceMPC":
            ax.annotate(scheme + ' (no overhead)', (summay_all[scheme]['time_stalled_rate']-0.1, summay_all[scheme]['avg_br']+0.1), fontsize=15)
        elif scheme == "interface":
            ax.annotate(scheme, (summay_all[scheme]['time_stalled_rate']+1.3, summay_all[scheme]['avg_br']-0.6), fontsize=15)
        elif scheme == "truthMPC":
            ax.annotate(scheme, (summay_all[scheme]['time_stalled_rate']-0.1, summay_all[scheme]['avg_br']+0.1), fontsize=15)
        elif scheme == "BB":
            ax.annotate("BBA", (summay_all[scheme]['time_stalled_rate']+0.1, summay_all[scheme]['avg_br']+0.4), fontsize=15)
        elif scheme == 'RL':
            ax.annotate("Pensieve", (summay_all[scheme]['time_stalled_rate']-0.1, summay_all[scheme]['avg_br']+0.6), fontsize=15)
        elif scheme == 'robustMPC':
            ax.annotate("robustMPC", (summay_all[scheme]['time_stalled_rate']+1.2, summay_all[scheme]['avg_br']+0.8), fontsize=15)
        elif scheme == 'fastMPC':
            ax.annotate("fastMPC", (summay_all[scheme]['time_stalled_rate']-0.1, summay_all[scheme]['avg_br']-4.8), fontsize=15)
        elif scheme == 'FESTIVE':
            ax.annotate("FESTIVE", (summay_all[scheme]['time_stalled_rate']+1.7, summay_all[scheme]['avg_br']+0.8), fontsize=15)
        else:
            ax.annotate(scheme, (summay_all[scheme]['time_stalled_rate']-0.1, summay_all[scheme]['avg_br']+0.8), fontsize=15)
    
    import json
    print(result_summary)
    with open('summary.json', 'w') as outfile:
        json.dump(result_summary, outfile)

    # colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    # for i,j in enumerate(ax.lines):
    #     j.set_color(colors[i])	
    plt.xlabel('Time Spent on Stall (%)', fontsize=20)
    plt.ylabel('Average Bitrate (Mbps)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.show()
    # plt.savefig('results.pdf')

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # i = 0
    # for scheme in SCHEMES:
    #     ax.bar(i, scheme_reward[scheme], label=scheme)
    #     i += 1
    # plt.legend()
    # plt.show()
    return

if __name__ == '__main__':
	main()