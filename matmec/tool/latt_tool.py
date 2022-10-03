from multiprocessing.sharedctypes import Value
from matmec.core.cell import Cell
import numpy as np
import json
import os

periodic_table_json = 'periodic_table.json'
detailed_periodic_table_json = 'periodic_table_detailed.json'

pt_file_path = os.path.join(os.path.dirname(__file__), periodic_table_json)


with open(pt_file_path, 'r') as f:
    periodic_table = json.load(f)

metallic_elements_list = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr',
       'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs',
       'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
       'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
       'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
       'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
       'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds ', 'Rg ',
       'Cn ', 'Nh', 'Fl', 'Mc', 'Lv']

nonmetallic_elements_list = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Si', 'P', 'S', 'Cl',
       'Ar', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Sb', 'Te', 'I', 'Xe', 'Po',
       'At', 'Ts', 'Og']

def easy_get_distance(p1, p2=None, isDirect = True, cell=None):
    '''
    Return the distance between p1 and p2, with boundary condition applied for direction coordinates condition
    '''
    if p2 is None:
        p2 = p1
    p1 = p1[:, None, :]
    p2 = p2[None, :, :]
    dis = abs(p1 - p2)
    if isDirect:
        dis[np.where( dis >= 0.5 )] -= 1
        if cell is not None:
            cell = Cell()
        dis = np.linalg.norm(np.matmul(dis, cell.lattvec*cell.scale), axis=1)
        return dis
    else:
        return np.sum(dis ** 2, axis=-1) ** 0.5


def get_distances(p1, p2=None, cell=None):
    '''
    Return the distance in [x,y,z] and cartesian distance with different positions.\n
    1) p1: position list 1. If p2 is not implemented, p1 is calculated with itself
    2) p2: position list 2, default is None.
    3) Cell object.
    '''
    from matmec.core.cell import Cell
    # make p1 and p2 2d matrix in case they contain only one element
    # consider the following operation as matrix operation but with each element to be a vector(position, length 3)
    p1 = np.atleast_2d(p1)
    assert(p1.shape[-1] == 3), "Each element in p1 should be position like, such as [0, 0, 0]"
    pbc = True
    # p2 = None
    if p2 is None:
        # use the feature of upper triangular matrix where the coordinate of each element can represent 
        # all different match of p1 itself like 0-1, 0-2, 1-2 (example of (3, 3) case)
        id1, id2 = np.triu_indices(len(p1), 1)
        D = (p1[id2] - p1[id1]).reshape(-1, 3)
    else:
        # if a column subtract a row, the every element will subtract every element in the row
        # for example, (1, 3) - (3, 1) in numpy.array form, will hence create a (3, 3) array
        # the new matrix starts with the first dimension of the subtracted matrix
        p2 = np.atleast_2d(p2)
        assert(p2.shape[-1] == 3), "Each element in p2 should be position like, such as [0, 0, 0]"
        D = (p2[np.newaxis, :, :] - p1[:, np.newaxis, :]).reshape(-1, 3)

    # apply periodic boundary condition
    if pbc == True:
        large_mask = D > 0.5
        while large_mask.any():
            _toSubtract = np.zeros(D.shape)
            _toSubtract[large_mask] = 1.0
            D -= _toSubtract
            large_mask = D > 0.5
        small_mask = D < -0.5
        while small_mask.any():
            _toAdd = np.zeros(D.shape)
            _toAdd[small_mask] = 1.0
            D += _toAdd
            small_mask = D < -0.5
    if cell is None:
        cell = Cell()
    # calculate the cartesian distance of the distance 
    D_len = np.linalg.norm(np.matmul(D, cell.lattvec*cell.scale), axis=1)

    if p2 is None:
        Dout = np.zeros((len(p1), len(p1), 3))
        Dout[id1, id2] = D
        Dout[id2, id1] = -D
        Dout_len = np.zeros((len(p1), len(p1)))
        Dout_len[id1, id2] = D_len
        Dout_len[id2, id1] = D_len
        return Dout, Dout_len
    else:
        D = D.reshape(-1, len(p2), 3)
        D_len = D_len.reshape(len(p1), len(p2))
        return D, D_len

def get_ele_num_list(ele_list):
    pass

def complete_arr(arr1: np.array, to_length: int, dtype=None, shape: tuple=(-1,)):
    assert(isinstance(arr1, (list, np.ndarray))), 'The input arr1 is not of list or np.array object'
    if dtype is None:
        dtype = type(arr1[0])
    arr1 = np.array(arr1, dtype=dtype).reshape(shape)
    to_length = int(to_length)
    if len(arr1) < to_length:
        lack = to_length - len(arr1)
        arr1 = np.append(arr1, [arr1[-1]]*lack, axis=0)
    return arr1

def get_diff_index(arr1, arr2, tolerence: float=1E-6):
    '''
    Return the index array of the different elements when comparing arr1 and arr2
    '''
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    assert(arr1.shape == arr2.shape), 'The shape of compared array should be same'
    if not arr1.dtype in ['bool', '<U2']:
        id = np.where(np.abs(arr1-arr2)>=tolerence)[0]
        id = np.unique(id)
    else:
        id = np.where(arr1!=arr2)[0]
        id = np.unique(id)
    return id


def get_numbers_afterIndex(name: str, index: int) -> int:
    '''
    give a name, return the numbers following the given index
    '''
    count = 0 
    leng = len(name) - 1 
    if index > leng:
        raise ValueError('Index out of range!')
    if index == leng:
        return 0

    while index < leng:
        index += 1
        if 48 <= ord(name[index]) <=57:
            count += 1
        else:
            return count
    
    return count

def check_formula(formula_name: str) -> dict:
    '''
    Return a tuple, 1st is whether the formula_name is a formula, 2nd is the formula dictionary\n
    I know this function is prolix, but I wrote this a long time ago, I really don't want to modify it...

    '''
    upperList = []
    for i in range(len(formula_name)):
        if formula_name[i].isupper():
            upperList.append(i)

    formula = dict()

    for i in range(len(upperList)):
        j = upperList[i]
        # for the indexes before the last one
        if i < len(upperList) - 1:
            if formula_name[j] in periodic_table:
                if formula_name[j:j+2] in periodic_table:
                    numbers_after_ele = get_numbers_afterIndex(formula_name, j+1)
                    if numbers_after_ele == 0:
                        formula['%s' % formula_name[j:j+2]] = 1
                        continue
                    elif numbers_after_ele < 3:
                        formula['%s' % formula_name[j:j+2]] = int(formula_name[j+2:j+2+numbers_after_ele])
                        continue
                    else:
                        continue
                else:
                    numbers_after_ele = get_numbers_afterIndex(formula_name, j)
                    print(numbers_after_ele)
                    if numbers_after_ele == 0:
                        formula['%s' % formula_name[j]] = 1
                        continue
                    elif numbers_after_ele < 3:
                        formula['%s' % formula_name[j]] = int(formula_name[j+1:j+1+numbers_after_ele])
                        continue
            if formula_name[j:j+2] in periodic_table:
                numbers_after_ele = get_numbers_afterIndex(formula_name, j+1)
                if numbers_after_ele == 0:
                    formula['%s' % formula_name[j:j+2]] = 1
                    continue
                elif numbers_after_ele < 3:
                    formula['%s' % formula_name[j:j+2]] = int(formula_name[j+2:j+2+numbers_after_ele])
                    continue
                else:
                    continue

        # for the last item in the upperList
        if i == len(upperList) - 1:
            j = upperList[i]
            if j == len(formula_name) - 1:
                if formula_name[j] in periodic_table:
                    formula['%s' % formula_name[j]] = 1
            else:
                if formula_name[j] in periodic_table:
                    if formula_name[j:j+2] in periodic_table:
                        numbers_after_ele = get_numbers_afterIndex(formula_name, j+1)
                        if numbers_after_ele == 0:
                            formula['%s' % formula_name[j:j+2]] = 1
                            continue
                        elif numbers_after_ele < 3:
                            formula['%s' % formula_name[j:j+2]] = int(formula_name[j+2:j+2+numbers_after_ele])
                            continue
                        else:
                            continue
                    else:
                        numbers_after_ele = get_numbers_afterIndex(formula_name, j)
                        if numbers_after_ele == 0:
                            formula['%s' % formula_name[j]] = 1
                            continue
                        elif numbers_after_ele < 3:
                            formula['%s' % formula_name[j]] = int(formula_name[j+1:j+1+numbers_after_ele])
                            continue
                elif formula_name[j:j+2] in periodic_table:
                    numbers_after_ele = get_numbers_afterIndex(formula_name, j+1)
                    if numbers_after_ele == 0:
                        formula['%s' % formula_name[j:j+2]] = 1
                        continue
                    elif numbers_after_ele < 3:
                        formula['%s' % formula_name[j:j+2]] = int(formula_name[j+2:j+2+numbers_after_ele])
                        continue
                    else:
                        continue

    
    if float(len(formula)) == float(len(upperList)):
        isFormula = True
    else:
        isFormula = False

    return isFormula, formula

def get_formula(elements_arr):
    assert(isinstance(elements_arr, (tuple, list, np.ndarray))), 'The input elements_arr should belong to tuple or list or np.ndarray'
    formula = {}
    for ele in elements_arr:
        if ele not in formula:
            formula[ele] = 1
        else:
            formula[ele] += 1
    metals_string = ''
    nonmetals_string = ''
    for ele in formula:
        if ele in metallic_elements_list:
            if formula[ele] == 1:
                metals_string += '%s' % ele
            else:
                metals_string += '%s%d' % (ele, formula[ele])
        else:
            if formula[ele] == 1:
                metals_string += '%s' % ele
            else:
                nonmetals_string += '%s%d' % (ele, formula[ele])
    return metals_string + nonmetals_string, formula

def get_elements_list_fromformula(formula: str):
    assert(isinstance(formula, str)), 'Input formula should be of str type'
    isformula, formula_dict = check_formula(formula)
    if isformula:
        elements_list = np.array([], dtype=str)
        for ele in formula_dict:
            elements_list = np.append(elements_list, [ele]*formula_dict[ele])
        return elements_list
    else:
        raise ValueError('The implemented formula doesnt has correct form')
    
def get_elements_list_from_poscarString(elementsNames, elementCounts):
    '''
    For the form of elementsNames and elementsCounts in POSCAR, postprocess the 
    list into the desired elements_list
    '''
    assert(isinstance(elementsNames, (tuple, list, np.ndarray))), 'Input elementsNames should be of list type.'
    assert(isinstance(elementCounts, (tuple, list, np.ndarray))), 'Input elementsCounts should be of list type.'
    elements_list = np.array([], dtype=str)
    for i, ele in enumerate(elementsNames):
        assert(ele in periodic_table), 'Please check the input elements name, %s is not in periodic table' % ele
        temp_elements_list = [ele]*int(elementCounts[i])
        elements_list = np.append(elements_list, temp_elements_list)
        del temp_elements_list
    return elements_list

def get_poslist_from_poscarstring(isSelectiveDynamic: bool, posString: str):
    '''
    input whether this poscar is selectiveDynamic, we can give you 
    the relative postions and fix according to this line
    '''
    assert(isinstance(posString, str)), 'The posliststring should be of str type.'
    temp = posString.strip().split()
    if isSelectiveDynamic:
        if len(temp) > 5:
            pos = np.array(temp[:3], dtype=float)
            fix = get_fix_from_string(fixString=temp[3:6])
        else:
            pos = np.array(temp[:3], dtype=float)
            fix = np.array([True, True, True], dtype=bool)
    else:
        pos = np.array(temp[:3], dtype=float)
        fix = np.array([True, True, True], dtype=bool)
    return pos, fix

def get_fix_from_string(fixString: list=None, fix: list=None, inverse=False):
    '''
    forward direction is to return the fix list from the splited one line in poscar
    when inverse is set to True, we give the string from fix list, the string will 
    later be written into poscar
    '''
    if not inverse:
        fixlist = np.array([], dtype=bool)
        for i in fixString:
            if i in ['T', 't', 'True', '.True.', ".TRUE."]:
                fixlist = np.append(fixlist, True)
            elif i in ['F', 'f', 'False', '.False.', '.FALSE.']:
                fixlist = np.append(fixlist, False)
        return fixlist
    else:
        s = ' '
        for i in fix:
            if i:
                s += 'T '
            else:
                s += 'F '
        return s

