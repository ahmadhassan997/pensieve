import numpy as np
import bisect

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
    # if next_ho_index != len(ho_array[:, 0]):
    #     next_ho_time = ho_array[next_ho_index][0]
    #     print(next_ho_time)
    #     ho_type = ho_array[next_ho_index, 1]
    #     print(ho_type)
    if is_HO:
        if next_ho_time - elapsed_t < 2 and ho_type <= -1:
            return ho_type
        elif next_ho_time - elapsed_t < 2 and ho_type == 1:
            return ho_type
    return 0