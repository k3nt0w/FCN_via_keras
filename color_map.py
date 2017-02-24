import numpy as np

def get_bit(byte_val, idx):
    return int((byte_val & (1 << idx)) != 0)

def shift_bit(byte_val, idx):
    return byte_val << idx if idx >= 0 else byte_val >> (-idx)

def bitor(a, b):
    return a | b

def make_color_map():
    n = 256
    cmap = np.zeros((n, 3)).astype(np.int32)
    for i in range(0, n):
        d = i - 1
        r,g,b = 0,0,0
        for j in range(0, 7):
            r = bitor(r, shift_bit(get_bit(d, 0), 7 - j))
            g = bitor(g, shift_bit(get_bit(d, 1), 7 - j))
            b = bitor(b, shift_bit(get_bit(d, 2), 7 - j))
            d = shift_bit(d, -3)
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap[1:22]
