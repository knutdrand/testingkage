import numpy as np
from itertools import product

bit_count_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint32)

def _get_combined_matrix_for_offset(masks, offset):
    helpermasks = masks[:-offset]
    mainmasks = masks[offset:]
    tmp = np.empty((len(helpermasks), 3, 3))
    for h, m in product(range(3), range(3)):
        tmp[:, h, m] = np.sum(bit_count_table[(helpermasks[:, h] & mainmasks[:, m])], axis=-1)    
    return tmp

def get_combined_matrix_for_offset(masks, offset):
    helpermasks = masks[:-offset][:, :, None, :]
    mainmasks = masks[offset:][:, None, :, :]
    return np.sum(bit_count_table[(helpermasks & mainmasks)], axis=-1)

def create_combined_matrices(genotype_matrix, window_size):
    genotype_matrix = np.asanyarray(genotype_matrix)
    masks = np.packbits(genotype_matrix[:, None, :]  == np.arange(3).reshape(1, 3, 1), axis=-1)
    return (get_combined_matrix_for_offset(masks, offset)
            for offset in range(1, min(window_size, genotype_matrix.shape[0])))
