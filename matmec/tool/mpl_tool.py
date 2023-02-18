import numpy as np


def regular_ticks(ax, axis: str):
    '''
    Regular the xticks or yticks, to make it increase from 0, from bottom to top, from left to right
    Parameters:
        ax: matplotlib.axes attribute
        axis: x or y ticks
    '''
    tickslabel_font = {'family': 'arial', 'size': 12}
    if axis == 'x':
        ori_xticks = ax.get_xticks()
        ori_xticks = np.array(ori_xticks, dtype=np.float16)
        if ori_xticks[0] < 0:
            xticks = np.array(ori_xticks+np.abs(ori_xticks[0]), dtype=np.float16)
        else:
            xticks = np.array(ori_xticks, dtype=np.float16)
        ax.set_xticks(ori_xticks)
        ax.set_xticklabels(xticks, fontdict=tickslabel_font)
        return 
    elif axis == 'y':
        ori_yticks = ax.get_yticks()
        ori_yticks = np.array(ori_yticks, dtype=np.float16)
        if ori_yticks[0] < 0:
            yticks = np.array(ori_yticks+np.abs(ori_yticks[0]), dtype=np.float16)
        else:
            yticks = np.array(ori_yticks, dtype=np.float16)
        ax.set_yticks(ori_yticks)
        ax.set_yticklabels(yticks, fontdict=tickslabel_font)
        return
    else:
        raise ValueError('axis should be x or y')