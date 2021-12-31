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
VIDEO_BIT_RATE = [1500, 4900, 8200, 11700, 32800, 152400]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 1500: 1, 4900: 2, 8200: 3, 11700: 12, 32800: 15, 152400: 20}
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 120.0
TOTAL_VIDEO_CHUNKS = 120
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
size_video1 = [41313924, 43161530, 38770457, 38678481, 37277126, 33509174, 35948395, 35963190, 35897710, 38850953,
               35232459, 40377414, 36661154, 21661223, 38243607, 34315159, 46734361, 44608844, 39241003, 10727101,
               43706094, 40316410, 20139165, 9343803, 66507577, 62868918, 59311729, 50124272, 61510011, 57651229,
               56681300, 49778904, 49959115, 45998715, 45118932, 37910009, 38462390, 41208932, 23867242, 20388860,
               22787863, 22113847, 24427264, 24282513, 30630072, 38771586, 34252822, 32825116, 14274907, 18673322,
               20556312, 44446394, 41123209, 28820268, 35291176, 30886005, 31903199, 39952228, 42904936, 42784546,
               46608285, 45010010, 44464999, 25988503, 32961159, 47678836, 63845161, 56713615, 50587484, 48990504,
               22053122, 27935566, 30146330, 65496652, 60079112, 28880920, 29113499, 27603917, 20863745, 25165459,
               35134773, 28986990, 31176301, 29906005, 47221581, 52256612, 44088758, 42754507, 44119047, 59032584,
               43372441, 60148321, 44142600, 41551976, 32061806, 34720562, 35234670, 86828312, 79224651, 79258834,
               28008108, 25178660, 30519618, 28659493, 28507956, 29881334, 43759388, 46740663, 37420196, 25837678,
               25908268, 30123693, 32188086, 33389177, 37885492, 33134487, 36397117, 19584030, 21926110, 26104154,
               2314642]
size_video2 = [8903603, 8837308, 8184926, 8333356, 8035965, 7472510, 7787243, 7915183, 7890352, 8333922, 7812610,
               8637653, 7916380, 4667135, 7926995, 7725063, 10093577, 9555963, 8485435, 3067292, 10617957, 8867905,
               5033350, 2070274, 13782054, 12997095, 12434874, 10365312, 11335137, 11985798, 11189123, 9398903, 9888648,
               9988354, 9081436, 8055840, 8486252, 9049659, 5858500, 5453966, 5267456, 4243230, 4752713, 4946914,
               6213863, 9137058, 7652044, 7075706, 4582266, 5156914, 5063274, 10131487, 10146738, 6332301, 7330827,
               6220437, 6760132, 9273821, 7928395, 8003793, 10150263, 8907487, 8256131, 6133010, 7499921, 10283207,
               13704075, 12213939, 10947211, 10148924, 4327336, 6095847, 6332443, 13076788, 12358407, 6093971, 5887529,
               5635988, 4458957, 5354433, 7351026, 6439166, 7015573, 6651002, 11070788, 12308770, 9872083, 9654065,
               9598944, 12962799, 11414296, 13778920, 10078075, 9095852, 6647013, 7238394, 7293967, 13034697, 13920685,
               12896854, 6124976, 5671500, 6704945, 6448171, 6460450, 6906705, 9942007, 10156374, 8512040, 5772294,
               6129325, 7346682, 7621233, 7976969, 8662750, 7765753, 8047001, 3871639, 4359653, 5535895, 631217]
size_video3 = [2757324, 3129734, 3036279, 3014195, 2878492, 2735308, 2869830, 2841699, 2869777, 2993761, 2794001,
               3054197, 2878010, 1920243, 3298222, 3154837, 3497744, 3238606, 2827897, 1154959, 3846880, 3222357,
               1807260, 788615, 4594386, 4309310, 4309911, 3623863, 3365305, 4168278, 3761162, 2758621, 3076972,
               3592799, 3227631, 2847457, 3184879, 3259966, 2670845, 2601191, 2256201, 1375606, 1557987, 1678511,
               2199894, 3543686, 2810290, 2588564, 2563348, 2542036, 2205235, 3650063, 4202731, 2346245, 2356568,
               1950218, 2251216, 3376681, 2522294, 2614019, 3098820, 2767350, 2538259, 2144732, 2532043, 3517362,
               4648287, 4246121, 3820651, 3484789, 1497311, 2134879, 2360003, 4776391, 4404207, 1934275, 1875399,
               1983265, 1745206, 2067378, 2821926, 2167840, 2340713, 2323077, 3999467, 4389152, 3339656, 3402140,
               3275036, 4501386, 4758516, 5191170, 3690819, 3237203, 2340906, 2566660, 2711386, 4260443, 4350366,
               4321828, 2459360, 2237778, 2597143, 2534429, 2637949, 2720917, 3782932, 3835162, 3277578, 2231843,
               2368465, 2743443, 2646816, 2782226, 2990049, 2634341, 2710945, 1169139, 1289457, 1839743, 246914]
size_video4 = [1774505, 2248151, 2125471, 2168577, 2012563, 1911036, 2004550, 1992790, 2029952, 2077618, 1958841,
               2178892, 2011918, 1241761, 2188776, 2068817, 2470084, 2352448, 2035254, 823723, 2781063, 2347664,
               1313901, 553955, 3192970, 2990666, 3051671, 2599389, 2454938, 3030710, 2744087, 2037195, 2225014,
               2543657, 2229353, 1947934, 2247295, 2344855, 1722042, 1627599, 1506222, 993804, 1120201, 1175238,
               1543273, 2432593, 1877434, 1742991, 1554722, 1587620, 1455628, 2531464, 2845384, 1668794, 1706380,
               1438667, 1635408, 2402722, 1819990, 1979765, 2219755, 1969711, 1859322, 1532930, 1841196, 2470728,
               3249288, 2940551, 2668630, 2472337, 1068208, 1467884, 1693864, 3303464, 3075746, 1404832, 1350477,
               1417928, 1200795, 1411473, 1849241, 1529944, 1659614, 1637139, 2810056, 3102278, 2362392, 2386365,
               2344286, 3206958, 3287458, 3679870, 2646825, 2300645, 1611356, 1765195, 1875696, 3151593, 3186205,
               3141123, 1716627, 1542441, 1787520, 1730166, 1837041, 1895681, 2627701, 2651664, 2291287, 1563473,
               1664641, 1941034, 1894490, 1996073, 2133540, 1908855, 1940335, 836205, 929034, 1273849, 171677]
size_video5 = [923208, 1296332, 1279196, 1302011, 1175444, 1193765, 1195093, 1160612, 1216073, 1255162, 1171477,
               1213522, 1223741, 832171, 1397677, 1366003, 1413216, 1308922, 1152822, 547774, 1765050, 1443223, 830756,
               350996, 1941919, 1767924, 1770966, 1503265, 1410741, 1760173, 1586763, 1150572, 1244690, 1519770,
               1291389, 1108583, 1296539, 1330965, 1100587, 1061129, 919957, 592872, 680092, 724124, 911761, 1488353,
               1128186, 1090146, 1118768, 1079072, 1024052, 1537611, 1812032, 1040076, 1055155, 848462, 985512, 1505625,
               1033982, 1063818, 1287131, 1133835, 1026837, 876905, 1030428, 1305427, 1895354, 1689841, 1544979,
               1426980, 640411, 863010, 1012189, 1768932, 1737294, 828844, 808433, 863943, 834867, 911483, 1171862,
               947299, 988138, 1014609, 1686819, 1864188, 1375629, 1271187, 1225853, 1913521, 2092099, 2204246, 1570135,
               1371470, 956984, 1059238, 1097761, 1758422, 1933059, 1888533, 1026161, 894091, 954786, 1036897, 1064195,
               1043301, 1441708, 1408849, 1400903, 972009, 1023569, 1204277, 1163915, 1222078, 1293365, 1151259,
               1190667, 496405, 557846, 806233, 113238]
size_video6 = [262937, 402355, 396718, 401644, 362635, 371755, 378047, 369310, 377731, 376830, 361137, 392228, 393409,
               270051, 402588, 382535, 387627, 390276, 376130, 208832, 635339, 515017, 326855, 127719, 542090, 497231,
               497483, 458954, 424537, 542430, 477972, 353202, 385506, 454212, 392477, 315649, 379767, 395964, 278865,
               262783, 243440, 170011, 197467, 213372, 291081, 489469, 367884, 329244, 318545, 339084, 318960, 484818,
               532442, 325974, 344930, 280827, 309167, 476460, 358161, 345682, 412341, 339391, 317706, 231708, 284053,
               386946, 581043, 512468, 487744, 441686, 187087, 243323, 330139, 438359, 452135, 256994, 247405, 265930,
               252942, 284718, 353564, 301358, 323026, 320693, 548466, 604459, 430394, 415348, 434092, 602379, 627967,
               656100, 474269, 407294, 295991, 317387, 335363, 523151, 619261, 583660, 346766, 298038, 339019, 336995,
               356745, 346229, 428094, 395523, 412505, 308061, 320001, 397385, 384781, 397764, 411511, 378302, 369322,
               184943, 172081, 245799, 38806]


def get_chunk_size(quality, index):
    if index < 0 or index > TOTAL_VIDEO_CHUNKS:
        return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index],
             1: size_video5[index], 0: size_video6[index]}
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
