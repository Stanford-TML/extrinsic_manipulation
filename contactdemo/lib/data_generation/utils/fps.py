import numpy as np


def furthest_point_sample(pc, n_samples):
    n_pc = pc.shape[0]
    p_first_idx = np.random.randint(0, n_pc)
    sampled_idxs = [p_first_idx]

    pairwise_dist_mat = ((pc[None] - pc[:, None]) ** 2).sum(axis=-1)
    min_dist_arr = pairwise_dist_mat[p_first_idx]

    for i in range(n_samples - 1):
        p_far_idx = min_dist_arr.argmax()
        sampled_idxs.append(p_far_idx)
        for j in range(n_pc):
            min_dist_arr[j] = min(min_dist_arr[j], pairwise_dist_mat[p_far_idx, j])

    return pc[sampled_idxs]
