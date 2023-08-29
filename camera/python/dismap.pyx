import cv2
import numpy as np

def find_Disparity(left, right, window_size, median_size):
    cost_volume = generate_census(left, right, window_size, median_size)
    aggregated_cost = np.empty_like(cost_volume)
    for d in range(31):
        aggregated_cost[:, window_size//2:, d] = cv2.medianBlur(cost_volume[:,window_size//2:,d], median_size)
    return np.argmin(aggregated_cost, axis=2)

# @jit(uint8[:,:,:](uint8[:, :], uint8[:, :], uint8, uint8), nopython=True, fastmath=True)
def generate_census(left, right, unsigned char window_size, unsigned char median_size):
    cdef unsigned char max_disparity = 31
    left_census = np.empty((left.shape[0], left.shape[1], window_size**2), dtype=np.uint8)
    right_census = np.empty((right.shape[0], right.shape[1], window_size**2), dtype=np.uint8)
    cdef int i,j 
    for i in range(window_size // 2, left.shape[0] - window_size // 2):
        for j in range(window_size // 2, left.shape[1] - window_size // 2):
            left_window = left[i - window_size // 2:i + window_size // 2 + 1,
                          j - window_size // 2:j + window_size // 2 + 1]
            right_window = right[i - window_size // 2:i + window_size // 2 + 1,
                           j - window_size // 2:j + window_size // 2 + 1]
            left_census[i, j] = (left_window >= left[i, j]).ravel()
            right_census[i, j] = (right_window >= right[i, j]).ravel()

    cost_volume = np.empty((left.shape[0], left.shape[1], max_disparity),dtype=np.uint8)

    cost_volume[:, :, 0] = np.sum(left_census[:, :] != right_census[:, :], axis=2)
    cdef int d
    for d in range(1, max_disparity):
        cost_volume[d:, :, d] = np.sum(left_census[:-d, :] != right_census[d:, :], axis=2)

    return cost_volume