import numpy as np
# from .cell import Cell



def simi_judge(vec, tolerence_type: int =1, tolerence:float =1.5e-2):
    '''
    judge the similarity of the input vector, return True of False
    tolerence_type: 1 for relative, 2 for absolute
    '''
    if not isinstance(vec, (list, np.array)):
        raise ValueError('input parameter vec should be of list type')
    else:
        vec = np.array(vec)
        id1, id2 = np.triu_indices(len(vec), 1)
        max_diff = np.abs(vec[id1]-vec[id2]).max()
        if tolerence_type == 1:
            max_value = np.abs(vec).max()
            return (max_diff/max_value) <= tolerence
        elif tolerence_type == 2:
            return max_diff <= tolerence
        else:
            raise ValueError('tolerence_type should be 1 for relateive or 2 for absolute')

