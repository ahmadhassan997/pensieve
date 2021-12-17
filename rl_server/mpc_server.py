#!/usr/bin/env python
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import base64
import urllib
import sys
import os
import json
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import time
import itertools

######################## FAST MPC #######################

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
MPC_FUTURE_CHUNK_COUNT = 5
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300, 6144, 17408]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20, 25, 30]
BITRATE_REWARD_MAP = {0: 0, 95: 1, 150: 2, 276: 3, 750: 12, 2048: 15, 4096: 20, 6144: 25, 17408: 30}
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 60.0
TOTAL_VIDEO_CHUNKS = 60
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size download_time reward
NN_MODEL = None

CHUNK_COMBO_OPTIONS = []

# video chunk sizes
size_video1 = [8984252, 8812499, 8207776, 8360161, 8605612, 8719429, 6829108, 8375548, 10569172, 6029581, 11249173,
               4139129, 13581050, 11763286, 12048267, 10367711, 10417649, 9217558, 9599223, 6710001, 5114103, 4851074,
               8270843, 7639317, 5784803, 8106221, 8989096, 6803489, 8612841, 8084730, 9850664, 7434155, 9040566,
               13507628, 11097191, 5320379, 9946759, 9323424, 5961885, 5127326, 7304721, 7179033, 12687276, 10207337,
               11911044, 14335586, 10376870, 7035694, 10449867, 12966473, 6313654, 7219588, 7727482, 10945677, 7740836,
               7408441, 8399356, 8814231, 6328902, 4995662, 366883]
size_video2 = [2973395, 3216338, 2923304, 2991183, 3068026, 3079690, 2596636, 3627557, 3500919, 2017353, 4069880,
               1451161, 4466516, 4093130, 3923791, 3246824, 3439277, 3151552, 3449206, 2849448, 1806035, 1506152,
               3139646, 2743841, 2918123, 3196667, 3582019, 2124905, 2979569, 2663315, 2892353, 2341279, 3016363,
               4532494, 3800892, 1795524, 3592102, 3172099, 1939284, 2031102, 2720491, 2385364, 4518933, 3407953,
               4031712, 5623731, 3697103, 2426202, 3334027, 4233552, 2526030, 2755094, 3052783, 3933755, 2927851,
               2808021, 2996183, 3103916, 2091032, 1540102, 154758]
size_video3 = [2026874, 2282622, 2045115, 2078942, 2149025, 2173872, 1756976, 2412694, 2464913, 1448736, 2978791,
               1053082, 3100445, 2911278, 2780758, 2333980, 2432248, 2176277, 2434200, 1898618, 1250449, 1045211,
               2193109, 1856435, 1862612, 2153393, 2499274, 1520022, 2138602, 1945277, 2035411, 1626990, 2082955,
               3185501, 2695408, 1260675, 2500537, 2234789, 1364394, 1350392, 1852865, 1664143, 3227678, 2407657,
               2880554, 4038757, 2649872, 1671868, 2384352, 2986638, 1770744, 1881609, 2039295, 2796603, 2048545,
               1990648, 2147194, 2223990, 1480811, 1080469, 109544]
size_video4 = [1196619, 1505626, 1341083, 1329507, 1395656, 1402782, 1257823, 1712055, 1553636, 984034, 2009347, 783864,
               2101627, 1850158, 1859476, 1533260, 1595053, 1360348, 1530079, 1189500, 808171, 749214, 1493407, 1261340,
               1312352, 1525448, 1699108, 1081087, 1498050, 1253424, 1386312, 1050765, 1254697, 2072603, 1706319,
               844645, 1474916, 1398867, 935215, 1023261, 1276221, 1162066, 2209113, 1416748, 1707709, 2599568, 1759463,
               1120791, 1611226, 1899599, 1191077, 1188536, 1327714, 1606567, 1391626, 1385995, 1484775, 1528801,
               1030878, 777915, 81052]
size_video5 = [769132, 982320, 847125, 867101, 922201, 927794, 841747, 1071429, 961027, 666674, 1440223, 558530,
               1301622, 1164089, 1215250, 1003362, 1042025, 852339, 975010, 691399, 497360, 499075, 1004708, 797124,
               842834, 985112, 1118262, 738603, 987316, 845712, 910957, 670041, 772068, 1341856, 1124973, 541852,
               861806, 862454, 611995, 675632, 807225, 775145, 1422398, 940377, 1137107, 1773256, 1124436, 717447,
               1005260, 1305479, 795921, 836956, 884241, 1001960, 905824, 905273, 987581, 1003667, 677789, 491097,
               55837]
size_video6 = [496029, 626184, 563912, 594231, 585818, 609315, 552068, 648719, 623398, 456346, 954901, 403887, 823760,
               755391, 794371, 643906, 668853, 538913, 628136, 396929, 304195, 327404, 663908, 527868, 516798, 664368,
               706359, 466439, 679058, 573279, 601218, 431436, 493402, 869953, 739963, 336555, 562135, 529952, 395600,
               424921, 556671, 487562, 927147, 604572, 778018, 1135471, 724487, 457125, 640249, 901806, 519913, 543084,
               600545, 637925, 588923, 600035, 651092, 665307, 454456, 327876, 37648]
size_video7 = [306487, 403283, 363863, 371640, 376465, 383921, 330657, 395247, 394371, 302714, 623862, 275667, 489860,
               463956, 476300, 399873, 408170, 346900, 400265, 227233, 178793, 179173, 434704, 328229, 291825, 407310,
               448442, 299424, 417363, 379063, 378277, 260820, 306637, 554729, 475069, 209683, 346581, 318551, 241126,
               255383, 342913, 319241, 625204, 406821, 519272, 689005, 450747, 278181, 426803, 561508, 340810, 357010,
               394244, 381111, 369153, 392169, 429721, 430076, 290579, 200591, 24700]
size_video8 = [133569, 160130, 145265, 150440, 148305, 159788, 126264, 154522, 162333, 130870, 266367, 129491, 178679,
               177194, 178960, 162651, 164555, 139096, 162192, 75789, 66072, 74207, 179936, 134925, 99636, 154189,
               176032, 125445, 170625, 158231, 157038, 106413, 119272, 219290, 196021, 77430, 144427, 129417, 101688,
               100328, 143850, 131157, 244529, 166772, 213175, 280143, 189589, 116167, 169327, 206082, 145785, 144460,
               160326, 144275, 144142, 167355, 188003, 183946, 120403, 85396, 10033]


def get_chunk_size(quality, index):
    if index < 0 or index > TOTAL_VIDEO_CHUNKS:
        return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    sizes = {7: size_video1[index], 6: size_video2[index], 5: size_video3[index], 4: size_video4[index],
             3: size_video5[index], 2: size_video6[index], 1: size_video7[index], 0: size_video8[index]}
    return sizes[quality]


def make_request_handler(input_dict):
    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.log_file = input_dict['log_file']
            # self.saver = input_dict['saver']
            self.s_batch = input_dict['s_batch']
            # self.a_batch = input_dict['a_batch']
            # self.r_batch = input_dict['r_batch']
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            print post_data

            if 'pastThroughput' in post_data:
                # @Hongzi: this is just the summary of throughput/quality at the end of the load
                # so we don't want to use this information to send back a new quality
                print "Summary: ", post_data
            else:
                # option 1. reward for just quality
                # reward = post_data['lastquality']
                # option 2. combine reward for quality and rebuffer time
                #           tune up the knob on rebuf to prevent it more
                # reward = post_data['lastquality'] - 0.1 * (post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])
                # option 3. give a fixed penalty if video is stalled
                #           this can reduce the variance in reward signal
                # reward = post_data['lastquality'] - 10 * ((post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) > 0)

                # option 4. use the metric in SIGCOMM MPC paper
                rebuffer_time = float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])

                # --linear reward--
                reward = VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                         - REBUF_PENALTY * rebuffer_time / M_IN_K \
                         - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                                                   self.input_dict['last_bit_rate']) / M_IN_K

                # --log reward--
                # log_bit_rate = np.log(VIDEO_BIT_RATE[post_data['lastquality']] / float(VIDEO_BIT_RATE[0]))   
                # log_last_bit_rate = np.log(self.input_dict['last_bit_rate'] / float(VIDEO_BIT_RATE[0]))

                # reward = log_bit_rate \
                #          - 4.3 * rebuffer_time / M_IN_K \
                #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

                # --hd reward--
                # reward = BITRATE_REWARD[post_data['lastquality']] \
                #         - 8 * rebuffer_time / M_IN_K - np.abs(BITRATE_REWARD[post_data['lastquality']] - BITRATE_REWARD_MAP[self.input_dict['last_bit_rate']])

                self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]
                self.input_dict['last_total_rebuf'] = post_data['RebufferTime']

                # retrieve previous state
                if len(self.s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(self.s_batch[-1], copy=True)

                # compute bandwidth measurement
                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                video_chunk_size = post_data['lastChunkSize']

                # compute number of video chunks left
                video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.input_dict['video_chunk_coount']
                self.input_dict['video_chunk_coount'] += 1

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                try:
                    state[0, -1] = VIDEO_BIT_RATE[post_data['lastquality']] / float(np.max(VIDEO_BIT_RATE))
                    state[1, -1] = post_data['buffer'] / BUFFER_NORM_FACTOR
                    state[2, -1] = rebuffer_time / M_IN_K
                    state[3, -1] = float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K  # kilo byte / ms
                    state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
                        CHUNK_TIL_VIDEO_END_CAP)
                except ZeroDivisionError:
                    # this should occur VERY rarely (1 out of 3000), should be a dash issue
                    # in this case we ignore the observation and roll back to an eariler one
                    if len(self.s_batch) == 0:
                        state = [np.zeros((S_INFO, S_LEN))]
                    else:
                        state = np.array(self.s_batch[-1], copy=True)

                # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
                self.log_file.write(str(time.time()) + '\t' +
                                    str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                                    str(post_data['buffer']) + '\t' +
                                    str(rebuffer_time / M_IN_K) + '\t' +
                                    str(video_chunk_size) + '\t' +
                                    str(video_chunk_fetch_time) + '\t' +
                                    str(reward) + '\n')
                self.log_file.flush()

                # pick bitrate according to MPC           
                # first get harmonic mean of last 5 bandwidths
                past_bandwidths = state[3, -5:]
                while past_bandwidths[0] == 0.0:
                    past_bandwidths = past_bandwidths[1:]
                # if ( len(state) < 5 ):
                #    past_bandwidths = state[3,-len(state):]
                # else:
                #    past_bandwidths = state[3,-5:]
                bandwidth_sum = 0
                for past_val in past_bandwidths:
                    bandwidth_sum += (1 / float(past_val))
                future_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))

                # future chunks length (try 4 if that many remaining)
                last_index = int(post_data['lastRequest'])
                future_chunk_length = MPC_FUTURE_CHUNK_COUNT
                if TOTAL_VIDEO_CHUNKS - last_index < 4:
                    future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

                # all possible combinations of 5 chunk bitrates (9^5 options)
                # iterate over list and for each, compute reward and store max reward combination
                max_reward = -100000000
                best_combo = ()
                start_buffer = float(post_data['buffer'])
                # start = time.time()
                for full_combo in CHUNK_COMBO_OPTIONS:
                    combo = full_combo[0:future_chunk_length]
                    # calculate total rebuffer time for this combination (start with start_buffer and subtract
                    # each download time and add 2 seconds in that order)
                    curr_rebuffer_time = 0
                    curr_buffer = start_buffer
                    bitrate_sum = 0
                    smoothness_diffs = 0
                    last_quality = int(post_data['lastquality'])
                    for position in range(0, len(combo)):
                        chunk_quality = combo[position]
                        index = last_index + position + 1  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                        download_time = (get_chunk_size(chunk_quality,
                                                        index) / 1000000.) / future_bandwidth  # this is MB/MB/s --> seconds
                        if (curr_buffer < download_time):
                            curr_rebuffer_time += (download_time - curr_buffer)
                            curr_buffer = 0
                        else:
                            curr_buffer -= download_time
                        curr_buffer += 4

                        # linear reward
                        # bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                        # smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])

                        # log reward
                        # log_bit_rate = np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                        # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0]))
                        # bitrate_sum += log_bit_rate
                        # smoothness_diffs += abs(log_bit_rate - log_last_bit_rate)

                        # hd reward
                        bitrate_sum += BITRATE_REWARD[chunk_quality]
                        smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])

                        last_quality = chunk_quality
                    # compute reward for this combination (one reward per 5-chunk combo)
                    # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

                    # linear reward 
                    # reward = (bitrate_sum/1000.) - (4.3*curr_rebuffer_time) - (smoothness_diffs/1000.)

                    # log reward
                    # reward = (bitrate_sum) - (4.3*curr_rebuffer_time) - (smoothness_diffs)

                    # hd reward
                    reward = bitrate_sum - (8 * curr_rebuffer_time) - (smoothness_diffs)

                    if (reward > max_reward):
                        max_reward = reward
                        best_combo = combo
                # send data to html side (first chunk of best combo)
                send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
                if best_combo != ():  # some combo was good
                    send_data = str(best_combo[0])

                end = time.time()
                # print "TOOK: " + str(end-start)

                end_of_video = False
                if post_data['lastRequest'] == TOTAL_VIDEO_CHUNKS:
                    send_data = "REFRESH"
                    end_of_video = True
                    self.input_dict['last_total_rebuf'] = 0
                    self.input_dict['last_bit_rate'] = DEFAULT_QUALITY
                    self.input_dict['video_chunk_coount'] = 0
                    self.log_file.write('\n')  # so that in the log we know where video ends

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', len(send_data))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data)

                # record [state, action, reward]
                # put it here after training, notice there is a shift in reward storage

                if end_of_video:
                    self.s_batch = [np.zeros((S_INFO, S_LEN))]
                else:
                    self.s_batch.append(state)

        def do_GET(self):
            print >> sys.stderr, 'GOT REQ'
            self.send_response(200)
            # self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', 20)
            self.end_headers()
            self.wfile.write("console.log('here');")

        def log_message(self, format, *args):
            return

    return Request_Handler


def run(server_class=HTTPServer, port=8333, log_file_path=LOG_FILE):
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # make chunk combination options
    for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
        CHUNK_COMBO_OPTIONS.append(combo)

    with open(log_file_path, 'wb') as log_file:

        s_batch = [np.zeros((S_INFO, S_LEN))]

        last_bit_rate = DEFAULT_QUALITY
        last_total_rebuf = 0
        # need this storage, because observation only contains total rebuffering time
        # we compute the difference to get

        video_chunk_count = 0

        input_dict = {'log_file': log_file,
                      'last_bit_rate': last_bit_rate,
                      'last_total_rebuf': last_total_rebuf,
                      'video_chunk_coount': video_chunk_count,
                      's_batch': s_batch}

        # interface to abr_rl server
        handler_class = make_request_handler(input_dict=input_dict)

        server_address = ('localhost', port)
        httpd = server_class(server_address, handler_class)
        print 'Listening on port ' + str(port)
        httpd.serve_forever()


def main():
    if len(sys.argv) == 2:
        trace_file = sys.argv[1]
        run(log_file_path=LOG_FILE + '_fastMPC_' + trace_file)
    else:
        run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print "Keyboard interrupted."
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
