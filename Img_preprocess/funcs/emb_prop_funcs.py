import numpy as np


def cal_dist(x, y):
    return np.sqrt(x**2 + y**2)


def cal_layer_thickness(cnt_out, cnt_in, center):
    distances = []
    # fig, ax = plt.subplots()
    if cnt_in is None:  # no lumen
        for idx in range(len(cnt_out[:, 1])):
            distances.append(np.sqrt((cnt_out[idx, 1]-center[1])**2+(cnt_out[idx, 0]-center[0])**2))
    else:
        for idx in range(len(cnt_out[:, 1])):
            # ax.imshow(img)
            thick = [cnt_out[idx, 1], cnt_out[idx, 0], 0, 0]
            if cnt_out[idx, 0]-center[0]!=0 and cnt_out[idx, 1]-center[1]!=0:
                slope = (cnt_out[idx, 1] - center[1]) * 1.0 / (cnt_out[idx, 0] - center[0])
                diff_min = 999999
                for idx2 in range(len(cnt_in[:, 1])):
                    if cnt_in[idx2, 0]-center[0]!=0 and cnt_in[idx2, 1]-center[1]!=0:
                        diff = abs(slope - (cnt_in[idx2, 1]-center[1])*1.0/(cnt_in[idx2, 0]-center[0]))
                        if diff < diff_min and (cnt_out[idx, 1]-center[1])*1.0/(cnt_in[idx2, 1]-center[1]) > 0 and (cnt_out[idx, 0]-center[0])*1.0/(cnt_in[idx2, 0]-center[0]) > 0:
                            diff_min = diff
                            thick[2] = cnt_in[idx2, 1]; thick[3] = cnt_in[idx2, 0]
                distances.append(np.sqrt((thick[1]-thick[3])**2+(thick[0]-thick[2])**2))
    return distances