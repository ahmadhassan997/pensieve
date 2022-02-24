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
import ho_utils

######################## RB HO #######################

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
MPC_FUTURE_CHUNK_COUNT = 5
VIDEO_BIT_RATE = [4900, 8200, 11700, 32800, 152400, 260000]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 1500: 1, 1850: 2, 4900: 3, 8200: 12, 11700: 15, 32800: 20}
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 60.0
TOTAL_VIDEO_CHUNKS = 60
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 260  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
THRPT_FILE = './throughputs/thrpt'
# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size download_time reward
NN_MODEL = None
HO_FACTOR = 1.4

start_time = np.loadtxt('start_time.txt', dtype=float)
print(start_time)

if len(sys.argv) == 3:
    ho_trace = np.loadtxt(open(sys.argv[2], "rb"), delimiter=",", skiprows=0)
    print(ho_trace)

CHUNK_COMBO_OPTIONS = []

# video chunk sizes
size_video1 =  [3898997, 3272180, 2913853, 2910421, 139588203, 108247856, 133590859, 145202943, 128913706, 121653710, 99247241, 100875488, 87415056, 89427267, 85214169, 79066131, 78221353, 81447023, 76954670, 72930723, 73522396, 82545700, 77551456, 71776349, 78162704, 71735089, 74436996, 71945755, 63114763, 73443022, 72984047, 65164657, 75370586, 66117771, 69280544, 72192300, 65447397, 74404067, 70058753, 64080637, 76565788, 65777254, 72423677, 73669436, 66042480, 77362737, 68541110, 67798081, 79191012, 65709626, 76807742, 76692980, 68051292, 80912997, 69802285, 70441275, 85328815, 69184868, 60370749, 51134093, 1009591]
size_video2 =  [3480775, 3272180, 2913853, 2910421, 69724395, 64544235, 68238787, 74122001, 65781674, 59325910, 49489151, 43382003, 36400102, 37808545, 38393683, 35775273, 35791444, 37812996, 36656065, 34260477, 36539495, 34934633, 37472480, 33960930, 37045410, 34761833, 37172626, 36674174, 30526756, 34452372, 35592699, 32282906, 37733948, 32736567, 35135749, 36820652, 32183695, 38477711, 36046409, 31839501, 37331773, 32433344, 36247037, 38145910, 32204109, 38691474, 34760375, 32706773, 38512633, 32012550, 37596161, 38571337, 32377462, 39982696, 35166655, 35445687, 42787799, 36129677, 33509837, 30320113, 801652]
size_video3 =  [1101519, 2871053, 2627764, 2317007, 11816317, 14671998, 15594921, 15833627, 14174027, 11733096, 9580050, 7692609, 6502714, 6806824, 7216259, 7492657, 7426827, 7150035, 7286507, 7394315, 7435833, 6486352, 6498652, 6280812, 6457313, 6517453, 6664140, 6977807, 6585659, 6883384, 6966632, 6652469, 7066540, 7017975, 7131901, 7822083, 7107849, 7392330, 7794418, 7232282, 7459615, 7359251, 7603000, 7967593, 7470103, 7728465, 7825920, 7364094, 8156901, 7573801, 7967670, 8257179, 7756837, 8766074, 8524286, 8100868, 9138236, 8597720, 8610278, 8859240, 291579]
size_video4 =  [427088, 1102047, 1111083, 1066380, 3929689, 4658506, 5308892, 4879776, 5465230, 4419693, 3390038, 2621842, 2250224, 2396841, 2489895, 2647781, 2677548, 2616054, 2643911, 2695131, 2677597, 2341766, 2437210, 2403729, 2518938, 2415843, 2436487, 2472325, 2436672, 2389025, 2420818, 2399724, 2451183, 2485412, 2521240, 2620947, 2528167, 2657381, 2605698, 2520434, 2632140, 2625445, 2710856, 2762874, 2628851, 2686664, 2713275, 2691207, 2788080, 2591805, 2657541, 2753887, 2655797, 2852144, 2793004, 2760403, 2872082, 2829935, 2986295, 3129785, 140096]
size_video5 =  [385365, 790650, 911646, 844970, 2969508, 3365094, 3919884, 3745938, 3431194, 2832778, 2078200, 1615473, 1571858, 1699150, 1782255, 1899687, 1887174, 1858888, 1873405, 1903096, 1899497, 1665126, 1745472, 1740096, 1791801, 1726833, 1758432, 1765517, 1737274, 1686029, 1737778, 1719887, 1759121, 1754600, 1772258, 1813747, 1753351, 1868400, 1801372, 1756670, 1828249, 1780560, 1869204, 1886999, 1815848, 1912505, 1855534, 1857416, 1934818, 1775665, 1843405, 1877606, 1888345, 2073424, 1924796, 1882535, 2006234, 1978624, 2053440, 2063846, 96354]
size_video6 =  [546328, 425156, 613681, 607418, 2007311, 2133418, 2136401, 2246644, 1895081, 1575580, 1143128, 934186, 936528, 1009038, 1057676, 1107721, 1130807, 1095030, 1117429, 1126196, 1166161, 980320, 1007766, 1004743, 1052718, 1020291, 1037792, 1059926, 1060401, 1045758, 1057164, 1030184, 1092837, 1053983, 1047095, 1095862, 1070638, 1117208, 1075777, 1044993, 1108411, 1067549, 1115300, 1119172, 1088201, 1147726, 1099728, 1112580, 1169933, 1065418, 1078575, 1117020, 1145031, 1219157, 1184532, 1162025, 1238364, 1129622, 1080811, 1150968, 60222]

def get_chunk_size(quality, index):
    if index < 0 or index > TOTAL_VIDEO_CHUNKS:
        return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index],
             1: size_video5[index], 0: size_video6[index]}
    return sizes[quality]

def get_next_quality(bw_est):
    next_quality = 0
    for i in range(6):
        if bw_est >= VIDEO_BIT_RATE[i]:
            next_quality = i
    return next_quality


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

                # compute throughput est.
                est_throughput = float(video_chunk_size) / float(video_chunk_fetch_time) * 8.0 # kbits/ms

                # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
                self.log_file.write(str(time.time()) + '\t' +
                                    str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                                    str(post_data['buffer']) + '\t' +
                                    str(rebuffer_time / M_IN_K) + '\t' +
                                    str(video_chunk_size) + '\t' +
                                    str(video_chunk_fetch_time) + '\t' +
                                    str(reward) + '\n')
                self.log_file.flush()

                ## TODO: check the handover prediction results
                nearest_ho_type = ho_utils.is_ho_near(start_time, time.time(), ho_trace)
                if nearest_ho_type != -1:
                    # there is a nearby HO here
                    print("Nearby HO, ", time.time() - start_time)
                if nearest_ho_type == 1:
                    # increase throughput
                    est_throughput *= 3
                elif nearest_ho_type == -1:
                    est_throughput /= HO_FACTOR

                next_quality =  get_next_quality(est_throughput)    

                est_throughput = est_throughput / 8.0
                with open(THRPT_FILE+'_RBHO_'+sys.argv[1], 'a') as thrpt_file:
                    thrpt_file.write(str(time.time()) + ' ' + str(est_throughput) + '\n')
                    thrpt_file.flush()
                    thrpt_file.close()

                # send data to html side (first chunk of best combo)
                # send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
                # if best_combo != ():  # some combo was good
                send_data = str(next_quality)

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
    if len(sys.argv) == 2 or len(sys.argv) == 3:
        trace_file = sys.argv[1]
        run(log_file_path=LOG_FILE + '_RBHO_' + trace_file)
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
