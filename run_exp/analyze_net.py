import numpy as np
import bisect
import sys
import matplotlib.pyplot as plt
import pandas

def get_capacity(time, timestamps, throughputs, granularity, window=60):
    if time - timestamps[0] < window:
        index_right = bisect.bisect_right(timestamps, time)
        capacity = np.sum(throughputs[:index_right]) * granularity
    else:
        index_left = bisect.bisect_left(timestamps, time - 60)
        index_right = bisect.bisect_right(timestamps, time)
        capacity = np.sum(throughputs[index_left:index_right]) * granularity
    return capacity



net_trace, video_trace_name = sys.argv[1], sys.argv[2]

net_tputs = pandas.read_csv(net_trace, delimiter=',')
net_tputs.columns = ['time', 'tput']
net_tputs['tput'] = net_tputs['tput'].to_numpy()
video_trace = np.loadtxt(video_trace_name)[:61]
print(video_trace.shape)
timestamps = video_trace[:, 0] - np.min(video_trace[:, 0])
video_bitrates = video_trace[:, 4] * 8.0 / 1000000.0 / 2

tput_capacity, video_capacity = [], []

for i in range(120):
    tput_capacity.append(get_capacity(i, net_tputs['time'].to_numpy(), net_tputs['tput'].to_numpy(), 1))
    video_capacity.append(get_capacity(i, timestamps, video_bitrates, 2))


plt.plot(np.arange(len(tput_capacity)),tput_capacity, label='network capacity')
plt.plot(np.arange(len(video_capacity)),video_capacity, label='used by video')
plt.ylabel('Capacity (Mbits)')
plt.xlabel('Time (Sec)')
plt.legend()
plt.show()