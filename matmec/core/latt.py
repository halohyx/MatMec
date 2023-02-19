from crypt import methods
from dis import dis
from multiprocessing.sharedctypes import Value
from turtle import pos
import warnings
from time import time
import os
import numpy as np
from .cell import Cell
from .atom import Atom, UPPDATE_ITEM
from copy import deepcopy
from typing import Union
from matmec.tool.latt_tool import get_distances, complete_arr, get_diff_index, \
                                periodic_table, check_formula, get_formula, \
                                get_elements_list_fromformula, get_elements_list_from_poscarString,\
                                get_poslist_from_poscarstring, get_fix_from_string,\
                                easy_get_distance, simplified_get_distance
                                

atoms_propdict = {'elements':'element', 'poslist':'pos', 'fix':'fix', \
                  'velocity': 'velocity'}
latt_prop = ['cell', 'name', '__direct__', 'propdict', 'natom', 'formula']


# BUG methods lacking: 1) merge_sites; 

class Atomlist:
    __name__ = 'matmec.core.Atomlist'
    def __init__(self, atomlist: Union[list, np.ndarray]=None) -> None:
        '''
        To pack the atomlist to provide it with more function
        '''
        if atomlist is None:
            self.atomlist = np.array([], dtype=Atom)
        else:
            self.atomlist = np.array(atomlist, dtype=object)
    
    @property
    def atomlist(self):
        return self._atomlist
    @atomlist.setter
    def atomlist(self, newatomlist: Union[list, np.ndarray]):
        if newatomlist is None:
            self._atomlist = None
        else:
            for atom in newatomlist:
                assert(atom.__name__ == 'matmec.core.Atom'), 'The newatomlist should only contain Atom object!'
            self._atomlist = newatomlist
    
    def get_atom_property(self, atom_property):
        atom_property_list = []
        for atom in self.atomlist:
            atom_property_list.append(atom)

    def append(self, atomlist):
        self.atomlist = np.append(self.atomlist, atomlist).reshape(-1)
        return self
    
    @staticmethod
    def static_move(atomlist, shiftVector: list or np.ndarray):
        '''
        A staticmethod to help move atoms.
        The most RUDE way to move atoms, be careful about the coordinates, we dont change the coordinates here
        '''
        for i in atomlist:
            i.pos += shiftVector
    
    def move(self, idx, shiftVector: list or np.ndarray):
        '''
        A built-in method to help move atoms. Move self.atomlist[idx] with shiftVector.
        The most RUDE way to move atoms, be careful about the coordinates, we dont change the coordinates here
        Args:
            idx: see in self.__getitem__
            shiftVector: a vector that will be directly added onto the atoms selected
        '''
        for i in self[idx]:
            i.pos += shiftVector
            
    def __add__(self, atomlist):
        return np.append(self.atomlist, atomlist).reshape(-1)

    def __getitem__(self, idx: int or str):
        '''
        You can get part of the atomlist by three ways:
        1. give the element such as a.atomlist["Al"], then you can get the Al atoms. Or give elements in a list
        2. give the slices like a numpy array
        3. give the bool mask
        '''
        if isinstance(idx, (list, np.ndarray)) and idx != []:
            if isinstance(idx[0], str):
                mask = [ i.element in idx for i in self.atomlist]
                return self.atomlist[mask]
            else:
                '''
                Including the Bool and int type of list
                '''
                return self.atomlist[idx]
        else:
            if isinstance(idx, str):
                mask = [ i.element == idx for i in self.atomlist]
                return self.atomlist[mask]
            else:
                return self.atomlist[idx]
    
    def __len__(self) -> int:
        '''Length of the atomlist'''
        return len(self.atomlist)
    
    def __repr__(self):
        s = ''
        for atom in self.atomlist:
            s += '%s\n' % atom
        return s

class Latt:

    __name__ = 'matmec.core.Latt'
    isSelectiveDynamic = False

    def __init__(self, 
                 formula: str=None, 
                 cell: Cell=None, 
                 pos= [0., 0., 0.],
                 fix: bool =[True, True, True],
                 velocity=[0., 0., 0.], 
                 isDirect: bool=True,
                 name: str ='MatMecLatt'):
        '''
        The Latt class.\n
        1) formula: the formula of the system. can be like Ce12H3
        2) cell: the cell of this latt
        3) pos: the positions of the atoms you give as formula, should be 
        at least longer than the number of elements given in formula
        4) isDirect: the given pos is in cartesian or direct coordinate
        5) fix: the freedom of each atom, defined as the selectdynamic of \
        atoms in VASP
        6) velocity: the velocity of each atom

        Trick: the formula can be of several types:
        1) normal formula form
        2) implemented elements list
        3) list of Atom
        '''
        # initilize the propdict, name, __direct__ and atomlist
        self.propdict = {}
        self.atomlist = np.array([], dtype=Atom)
        self.name = name
        self.__direct__ = isDirect

        # set cell
        if cell is None:
            self.cell = None
        else:
            if isinstance(cell, Cell):
                self._set_cell(cell)
            else:
                cell = Cell(lattvec=cell)
                self._set_cell(cell)

        # last step of initilization is to add atoms
        if isinstance(formula, str):
            isformula, formula_dict = check_formula(formula)
            if isformula:
                # elements_list = get_elements_list_fromformula(formula=formula)
                self.addatom(formula, pos, self.__direct__, fix, velocity)
            else:
                raise ValueError('The implemented formula doesnt has correct form')
        elif isinstance(formula, (tuple, list, np.ndarray)):
            # if implemented a list of atoms, just add them
            self.addatom(formula, pos, self.__direct__, fix, velocity)
        elif isinstance(formula, Latt):
            self.set_cell(formula.cell)
            self.addatom(formula.atomlist)


    def addatom(self, element: str, pos=[0., 0., 0.], isDirect: bool=True, \
                fix: bool =[True, True, True], velocity=[0., 0., 0.]):
        '''
        Several situations in providing input:
        1) single Atom instance in the 1st place
        2) single Latt instance in the 1st place (cell will be first set as the cell of current Latt instance,\
            and the direct set to the self.get_direct())
        3) a list of Atom instances in the first place
        4) a list of string instances indicating what elements will be added, and the pos, isDirect, fix, velocity\
            can be given in following
        5) single string instance, will be regarded as formula and resolved by built-in algorism and call in 4)
        '''
        # one object given, in Atom or Latt form
        if hasattr(element, '__name__'):
            if element.__name__ == 'matmec.core.Atom' or element.__name__ == 'matmec.core.Atomlist':
                self.atomlist.append(element)
                return 0
            elif element.__name__ == 'matmec.core.Latt':
                '''
                If a Latt object a1 is given, we first a1.set_direct() to cartesian coordinate, and set the cell\
                as self.cell, and a1.set_direct() to the self.__direct__
                '''
                element.cell = self.cell
                element.set_direct(self.get_direct())
                self.atomlist.append(element.atomlist)
                return 0
            else:
                raise ValueError('What the fuck you inputted?')
        # A list object given, a list of str or Atom, will determine later
        elif isinstance(element, (tuple, list, np.ndarray)):
            # if the object in the list is of Atom class
            if isinstance(element[0], Atom):
                for atom in element:
                    assert(atom.__name__ == 'matmec.core.Atom'), 'Object %s in the list is not Atom object.' % atom
                self.atomlist.append(element)
                return 0
            # if the object in the list is of str class
            elif isinstance(element[0], str):
                pos = np.array(pos, dtype=float).reshape(-1, 3)
                to_length = len(pos)
                element = complete_arr(arr1=element, to_length=to_length, dtype=str, shape=(-1,))
                assert(len(element) == len(pos)), 'Not enough pos input!'
                fix = complete_arr(arr1=fix, to_length=to_length, dtype=bool, shape=(-1, 3))
                velocity = complete_arr(arr1=velocity, to_length=to_length, dtype=float, shape=(-1, 3))
                assert(isinstance(isDirect, bool)), 'isDirect input should be of bool type'
                created_atoms = []
                for i in range(len(element)):
                    atom = Atom(element=element[i], pos=pos[i], isDirect=isDirect, cell=self.cell,\
                                fix=fix[i], velocity=velocity[i])
                    created_atoms.append(atom)
                    del atom
                self.addatom(created_atoms)
            # no other form supported
            else:
                raise ValueError('The first offered parameter should be one or list of Atom or str type, pls check the form!')
        # a single string object is given as element, will use the number of offered pos to created atoms
        elif isinstance(element, str):
            element = get_elements_list_fromformula(element)
            self.addatom(element, pos, isDirect, fix, velocity)
        else:
            raise ValueError('Pls provide correct type of input.')

    def __add_single_atom(self, atom):
        pass
    
    # periodic boundary condition
    def wrap(self):
        '''
        Apply boundary condition in 3 diretions
        Do this only in direct coordinates
        '''
        # in case the poslist is not updated, if the length of poslist is not equal to atomlist
        # update the poslist with current atomlist first
        eps = 1E-7
        if len(self.poslist) < self.natom:
            self._update_atom_propdict('poslist')
        if self.__direct__:
            poslist = np.array(self.poslist)
            poslist += eps
            poslist %= 1.0
            poslist -= eps
            self.poslist = poslist
        else:
            # if current coordinate is cartesian coordinate, \
            # then calculate the corresponding direct coordinate and apply boundary conditions
            poslist = np.array(self.poslist)
            transformMattrix = np.linalg.inv(self.cell.lattvec*self.cell.scale)
            poslist = np.matmul(np.array(poslist), transformMattrix)
            poslist += eps
            poslist %= 1.0
            poslist -= eps
            poslist = np.matmul(poslist, self.cell.lattvec*self.cell.scale)
            self.poslist = poslist

    #latt_method: to merge sites using the cluster algorism in scipy
    def merge_sites(self, 
                    tolerence: float=1E-6):
        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import fcluster, linkage
        poslist = np.array(self.poslist)
        if self.get_direct() == True:
            _, dis_arr = get_distances(poslist, cell=self.cell)
        else:
            _, dis_arr = get_distances(poslist, cell=None)
        # use the clustering algorism in scipy to find atoms to merge
        clusters = fcluster(linkage(squareform(dis_arr)), 
                            t=tolerence, 
                            criterion="distance")
        newAtomlist = Atomlist()
        for i in np.unique(clusters):
            cluster = self.atomlist[np.where(clusters == i)]
            if len(cluster) == 1:
                newAtomlist.append(cluster)
            else:
                if len(set([ atom.element for atom in cluster])) != 1:
                    raise ValueError('Atoms to be merged are not with same element, pls check!')
                else:
                    # now it's like when those atoms are overlapped at the same site, we will randomly
                    # choose one to represent them. But all these should be the same element
                    newAtomlist.append(cluster[0])
        self.atomlist = newAtomlist
        for key in self.propdict.keys():
            if key in atoms_propdict.values():
                self._update_atom_propdict(key)
        return self.atomlist

    # check overlap
    def check_overlap(self, 
                      tolerence: float=1E-6):
        '''
        Check if some atoms are overlaped
        1) if no atoms overlapped, return False
        2) if some atoms overlap with others, return two index list, atoms in the first index list \
        overlap with the corresponding atoms in the second index list
        '''
        poslist = np.array(self.poslist)
        if self.get_direct() == True:
            _, dis_arr = get_distances(poslist, cell=self.cell)
        else:
            _, dis_arr = get_distances(poslist, cell=None)
        id1, id2 = np.triu_indices(len(poslist), 1)
        overlap_id = np.where(dis_arr[id1, id2] < tolerence)[0]
        if len(overlap_id) == 0:
            return False
        else:
            return id1[overlap_id], id2[overlap_id]


    # sort
    def sort(self):
        elements = self.elements
        ele_num_list = np.array([periodic_table[ele] for ele in elements])
        sorted_index = ele_num_list.argsort()
        self.atomlist = self.atomlist[sorted_index]
        for atom_prop in atoms_propdict.keys():
            self._update_atom_propdict(atom_prop)


    def newpropdict(self, name, value):
        self.propdict[name] = value

        
    # @latt_property: name
    def get_name(self):
        return self._get_propdict_value('name')
    def set_name(self, name: str):
        assert(isinstance(name, str)), 'The name should be of str type'
        self._set_propdict('name', name)
    name = property( get_name, set_name, doc='The name of the system')

    @property
    def atomlist(self):
        return self._atomlist
    @atomlist.setter
    def atomlist(self, newatomlist):
        self._atomlist = Atomlist(newatomlist)
        nowKeys = list(self.propdict.keys())
        for key in nowKeys:
            if key in atoms_propdict.keys():
                del self.propdict[key]
    

    # @latt_property: cell
    def _get_cell(self):
        return self._inplace_get_propdict_value('cell')
    def _set_cell(self, cell: Cell):
        '''
        When set cell, the atoms in latt will be changed into cartesian coordinate,\
            and the cell can be directly attached. Then the coordinate is changed back.
        '''
        if cell is not None:
            assert(cell.__name__ == 'matmec.core.Cell'), 'The cell should be of Cell type'
            isDirect = self.__direct__
            if self.cell is not None:
                self.set_direct(False)
                self._set_propdict('cell', cell)
                self.set_direct(isDirect)
            else:
                self._set_propdict('cell', cell)
        else:
            self._set_propdict('cell', None)
    def set_cell(self, 
                 lattvec: np.ndarray =None, 
                 scale: float =1.0, 
                 scaleatoms=False):
        '''
        Set the cell of current latt instance.
        Args:
            lattvec: the lattice vectors of new cell. Or can be given as another Cell instance.
            scale: only works when lattvec is given as lattive vectors not the Cell instance. Scale of the cell.
            scaleatoms: whether to change the cartesian coordinates of atoms, or scale the position \
                of atoms using the direct coordinates of atoms in the new cell.
        '''
        if isinstance(lattvec, Cell):
            cell = lattvec
        else:
            cell = Cell()
            cell.scale = scale
            if lattvec is None:
                lattvec = np.array([[1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0]])
            else:
                lattvec = np.array(lattvec, dtype=float)
                assert(lattvec.shape == (3, 3)), ValueError("The lattice vector should be (3, 3) matrix")
            cell.lattvec = lattvec
        if not scaleatoms or self.cell == None:
            # don't change the cartesian coordinates of the atoms
            self.cell = cell
        else:
            # apply cell onto the direct positions of atoms, will scale them
            isDirect = self.__direct__
            self.set_direct(True)
            self._set_propdict('cell', cell)
            self.set_direct(isDirect)
    cell = property(_get_cell, _set_cell, doc='The cell that current atom belongs to')

    def del_cell(self):
        self._set_propdict('cell', None)

    # @latt_property: propdict
    def _get_propdict_value(self, name):
        '''
        Return the deepcopy of the property
        '''
        return deepcopy(self.propdict.get(name, None))
    def _inplace_get_propdict_value(self, name):
        '''
        Return the property, and that can be inplacely changed
        '''
        return self.propdict.get(name, None)
    def _set_propdict(self, name, value=None):
        if name in atoms_propdict:
            self._update_atom_propdict(name)
        elif name in latt_prop:
            # the properties directly from the latt
            self.propdict[name] = value
        else:
            raise ValueError('Provided name is not one of the latt properties.')
    def _update_atom_propdict(self, name):
        assert(name in atoms_propdict), 'The property doesnt belong to atoms_prop, pls call set_propdict'
        exec('prop_value = [ atom.%s for atom in self.atomlist]' % atoms_propdict[name])
        self.propdict[name]=np.array(locals()['prop_value'])
        if name in UPPDATE_ITEM:
            del UPPDATE_ITEM[name]
    
    # the relative method in setting atoms's properties in the atomlist
    def _atom_prop_getter(self, name):
        '''
        The getter for the attributes of atom_prop
        '''
        if name in self.propdict:
            if UPPDATE_ITEM.get(name):
                self._update_atom_propdict(name)
            return self._get_propdict_value(name)
        else:
            self._update_atom_propdict(name)
            return self._get_propdict_value(name)

    def _atom_prop_setter(self, name, newarr):
        '''
        The setter for the attributes of atom_prop.\n
        1) when newarr has same length as curren atomlist. Only the indexes of atoms with different element in the newarr and atom_prop will be modified
        2) when newarr has shorter length than current atomlist, extra atoms in atomlist will be deleted. Then follow 1)
        3) when newarr has longer length than current atomlist, extra atoms will be created following the copy of the last atom in atomlist. Then follow 1)
        '''
        newarr = np.atleast_2d(newarr)
        if len(newarr) != len(self.atomlist):
            # 创建新的原子或删除现在的部分原子
            if len(newarr) < len(self.atomlist):
                warnings.warn('The newarr has shorter length than current atomlist, extra atoms will be deleted. pls make sure you know what you are doing!')
                self.atomlist = self.atomlist[:len(newarr)]
                self._update_atom_propdict(name)
                modify_index = get_diff_index(newarr, self._get_propdict_value(name))
                self._set_part_atom_propdict(name, newarr[modify_index], modify_index)
            elif len(newarr) > len(self.atomlist):
                warnings.warn('The newarr has shorter length than current atomlist, extra atoms will be created follow the last atom in atomlist. pls make sure you know what you are doing!')
                created_atom_num = len(newarr) - len(self.atomlist)
                created_atoms = self.atomlist[-1]*created_atom_num
                self.atomlist.append(created_atoms)
                self._update_atom_propdict(name)
                modify_index = get_diff_index(newarr, self._get_propdict_value(name))
                self._set_part_atom_propdict(name, newarr[modify_index], modify_index)
        else:
            modify_index = get_diff_index(self._get_propdict_value(name), newarr)
            self._set_part_atom_propdict(name, newarr[modify_index], modify_index)

    def _set_part_atom_propdict(self, name, value, modify_index):
        '''
        When the user is modifying the properties of atom_prop, we call this func,\
        which modify the atoms by the given modify_index, and then update the propdict of given name
        '''
        assert(len(value) == len(modify_index)),'The length of the value to modify and the index to modify shoud be same!'
        for i, atom in enumerate(self.atomlist[modify_index]):
            atom.set_propdict(atoms_propdict[name], value[i])
        self._update_atom_propdict(name)

    # @atom_property: poslist
    def _get_poslist(self):
        return self._atom_prop_getter('poslist')
    def _set_poslist(self, newposlist):
        self._atom_prop_setter('poslist', newposlist)
        direct = 'direc' if self.__direct__ else 'cartes'
        # print('Current coordinate is %s' % direct)
    poslist = property(_get_poslist, _set_poslist, doc='Position list of this Latt')

    # @atom_property: velocity
    def _get_velocity(self):
        return self._atom_prop_getter('velocity')
    def _set_velocity(self, newvelocity):
        self._atom_prop_setter('velocity', newvelocity)
    velocity = property(_get_velocity, _set_velocity, doc='Velocity list of this Latt')

    # @atom_property: fix
    def _get_fix(self):
        return self._atom_prop_getter('fix')
    def _set_fix(self, newfix):
        self._atom_prop_setter('fix', newfix)
    fix = property(_get_fix, _set_fix, doc='Fix list of this Latt')

    # @atom_property: elements
    def _get_elements(self):
        return self._atom_prop_getter('elements')
    def _set_elements(self, elements):
        self._atom_prop_setter('elements', elements)
    elements = property(_get_elements, _set_elements, doc='elements list of this Latt')


    # @latt_property: direct
    def get_direct(self):
        return self.__direct__
    def set_direct(self, isDirect: bool =True):
        assert(isinstance(isDirect, bool)), 'The isDirect should be of bool type'
        if self.cell:
            transferNeed = self.__direct__ == isDirect
            if not transferNeed:
                if not len(self.poslist) == 0: 
                    if self.__direct__:
                        self.__DtoC()
                    else:
                        self.__CtoD()
                    for atom in self.atomlist:
                        atom.__direct__ = isDirect
                    self.__direct__ = isDirect
                else:
                    self.__direct__ = isDirect
        else:
            self.__direct__ = isDirect

    # @latt_property: natom
    def _get_natom(self):
        return len(self.atomlist)
    def _set_natom(self):
        raise ValueError('natom cannot be set!')
    natom = property(_get_natom, _set_natom, doc='The number of atoms in this system.')


    # method to change the direct and cartesian coordinate
    def __CtoD(self):
        assert(self.cell is not None), 'cell should be defined ahead of changing coordinate type'
        transformMattrix = np.linalg.inv(self.cell.lattvec*self.cell.scale)
        newposlist = np.matmul(np.array(self.poslist), transformMattrix)
        # check if some atom is outside the box defined by cell, if so, then raise error
        # outBoundary = newposlist > 1
        # if outBoundary.any():
        #     raise ValueError('Some atoms are outside the box defined by cell, pls check!')
        self.poslist = newposlist
    def __DtoC(self):
        assert(self.cell is not None), 'cell should be defined ahead of changing coordinate type'
        transformMattrix = self.cell.lattvec*self.cell.scale
        
        self.poslist = np.matmul(np.array(self.poslist), transformMattrix)

    def __getitem__(self, idx: int):
        return self.atomlist[idx]

    def __repr__(self) -> str:
        s = ''
        s += 'name: %s\n' % self.name
        s += 'formula: %s \n' % get_formula(self.elements)[0]
        # s += 'natoms: %d\n' % self.natom
        s += 'cell: %s' % self.cell
        return s

    def copy(self):
        return deepcopy(self)

    def get_supercell(self,
                      supercell: Union[list, np.ndarray]):
        if len(supercell) != 3:
            raise ValueError('Check the input supercell, should be length of 3 and all integers')
        for i in supercell:
            assert (np.floor(i) == i), 'Check the input supercell, should be length of 3 and all integers'
        tmpLatt = self.copy()
        if not tmpLatt.__direct__:
            tmpLatt.set_direct(True)
        newcell = np.array(tmpLatt.cell.lattvec.T*np.array(supercell)).T# [tmpLatt.cell.lattvec[i]*supercell[i] for i in range(3)]
        tmpLatt.set_cell(newcell)

        newatomlist = Atomlist(tmpLatt.atomlist)
        for i, dimen in enumerate(supercell):
            oldlen = len(newatomlist)
            # duplicate the newatomlist in desired dimention
            tmpatomlist = Atomlist()
            for _ in range(dimen):
                tmpatomlist.append(deepcopy(newatomlist))
            newatomlist = deepcopy(tmpatomlist)
            del tmpatomlist
            # prepare the toadd_list
            toadd_list = []
            for j in range(2, dimen+1):
                toadd_list.append([(j-1)/dimen]*oldlen)
            toadd_list = np.array(toadd_list).reshape(-1)
            # for each new created atom, add corresponding value onto it
            for atom, toadd in zip(newatomlist[oldlen:], toadd_list):
                atom.pos[i] += toadd
        tmpLatt.atomlist = newatomlist
        for key in tmpLatt.propdict.keys():
            if key in atoms_propdict:
                tmpLatt._update_atom_propdict(key)
        # tmpLatt.wrap()
        return tmpLatt

    def get_neighbor_matrix(self,
                          max_neigh = 3,
                          verbose = True):
        '''
        Return the neighbor list of current system.
        Parameters:
        max_neigh: default: 3, define which the max nearest neighbor shell
        verbose: default: True, whether to output the information in a verbose way.
        '''

        if verbose:
            print(f"Generating neighbor matrix for total {self.natom} atoms", end=' ... \n')
            t0 = time()

        ndim = 27
        max_n = max_neigh + 1

        if verbose:
            print(f"Generating distance matrix for total {self.natom} atoms", end=' ... ')

        direct_translation_vectors = [[ 0, 0, 0],[ 1, 0, 0],[-1, 0, 0],[ 0, 1, 0],[ 0,-1, 0],[ 1, 1, 0],[-1,-1, 0],[ 1,-1, 0],[-1, 1, 0],
                                    [ 0, 0, 1],[ 1, 0, 1],[-1, 0, 1],[ 0, 1, 1],[ 0,-1, 1],[ 1, 1, 1],[-1,-1, 1],[ 1,-1, 1],[-1, 1, 1],
                                    [ 0, 0,-1],[ 1, 0,-1],[-1, 0,-1],[ 0, 1,-1],[ 0,-1,-1],[ 1, 1,-1],[-1,-1,-1],[ 1,-1,-1],[-1, 1,-1]]
        
        # the ndim number of matrix for translate the current cell to all the periodic equivalent position, for 3D there are 27
        translation_matrix = np.array([ vec*self.natom for vec in direct_translation_vectors[:ndim] ]).reshape(ndim, self.natom, 3)

        # the translated position, for 3D there are 27 sets
        allPos = np.tile(self.poslist.flatten(), ndim).reshape(ndim, -1, 3)
        allPos = allPos - translation_matrix

        # get the cartesian positions of all periodic equivalent cells
        cell = self.cell.lattvec*self.cell.scale
        allCartPos = np.matmul(allPos, cell)

        # get the distance matrixes for the original cell and all the equivalent cells, and take the minimum at the same position
        # of all the distance matrixes
        large_distance_mat = np.array([ simplified_get_distance(allCartPos[0], CardPos) for CardPos in allCartPos])
        dis_mat = np.minimum.reduce(large_distance_mat, axis=0)
        dis_mat = np.round(dis_mat, 3)

        if verbose:
            print(f"Done in {time()-t0:.3f} sec")

        # all the type of distances in the system. Use this to judge which nearest neighbor shell the atom pairs belong to
        neigh_dis_list = np.unique(dis_mat.ravel())[:max_n]
        if len(neigh_dis_list)<max_n: max_n=len(neigh_dis_list)

        neigh_mat = np.zeros((self.natom, self.natom))
        for m in range(1, max_n):
            neigh_indice = np.where(np.logical_and(dis_mat <= neigh_dis_list[m], neigh_mat == 0))
            neigh_mat[neigh_indice] = m

        neigh_mat[np.diag_indices(self.natom)] = 0
        self.neigh_mat = neigh_mat

        if verbose:
            print(f"Done generating neighbor matrix in {time()-t0:.3f} sec")
            
        return neigh_mat

    @classmethod
    def read_from_poscar(cls, 
                         file: os.path='./POSCAR'
                         ):
        '''
        Read structure from POSCAR format file
        Parameters:
        file: default: './POSCAR', a path like string pointing to the location of the POSCAR file.
        return: a Latt class containg the information of the POSCAR file.
        '''
        if os.path.isfile('%s' % file):
            poscar = cls()
            with open(file, 'r') as f:
                count = 1
                line = f.readline()
                poscar.set_name(line.strip()) # First line indicating the system name
                while line:
                    line = f.readline()
                    count += 1
                    if count == 2:
                        scale = float(line.strip())
                        continue
                    elif count == 3:
                        single_lattceVec = np.array(line.strip().split(), dtype=float)
                        if len(single_lattceVec) != 3:
                            raise ValueError('length of lattice vector not equals 3!')
                        LatticeVecs = single_lattceVec
                        continue
                    elif count==4 or count==5:
                        single_lattceVec = np.array(line.strip().split(), dtype=float)
                        if len(single_lattceVec) != 3:
                            raise ValueError('length of lattice vector not equals 3!')
                        LatticeVecs = np.append(LatticeVecs, single_lattceVec)
                        continue
                    elif count == 6:
                        LatticeVecs = LatticeVecs.reshape(3, 3)
                        # BUG pls make sure the set_cell func is correct here
                        poscar.set_cell(lattvec=LatticeVecs, scale=scale)
                        elementsNames = line.strip().split()
                        continue
                    elif count == 7:
                        elementCounts = [int(i) for i in line.strip().split()]
                        elementList = get_elements_list_from_poscarString(elementsNames, elementCounts)
                        natoms = len(elementList)
                        continue
                    elif count == 8:
                        '''Selective dynamics or Coordinate specification'''
                        if line[0] in ['S', 's']:
                            # when SelectiveDynamic is true, we don't count this in Counts, 
                            # so the starting Count of the poslist will always be 9
                            isSelectiveDynamic = True
                            line = f.readline()
                            if line[0] in ['D', 'd']:
                                poscar.__direct__ = True
                            elif line[0] in ['C', 'c']:
                                poscar.__direct__ = False
                        elif line[0] in ['D', 'd']:
                            isSelectiveDynamic = False
                            poscar.__direct__ = True
                        elif line[0] in ['C', 'c']:
                            isSelectiveDynamic = False
                            poscar.__direct__ = False
                        poslist = np.array([], dtype=float)
                        fixlist = np.array([], dtype=bool)
                        velocitylist = np.array([], dtype=float)
                        continue
                    elif count >= 9 and count < 9 + natoms:
                        # only loop in the length defined in elementCounts
                        temppos, tempfix = get_poslist_from_poscarstring(isSelectiveDynamic, line)
                        poslist = np.append(poslist, temppos)
                        fixlist = np.append(fixlist, tempfix)
                        del temppos, tempfix
                        continue
                    elif count >= 9 + natoms:
                         tempvelocity= np.array(line.strip().split()[:3], dtype=float)
                         velocitylist = np.append(velocitylist, tempvelocity)
                         del tempvelocity
                # after the loop of the file, addatoms to the poscar
                poslist = poslist.reshape(-1, 3)

                fixlist = fixlist.reshape(-1, 3)
                velocitylist = velocitylist.reshape(-1, 3)
                if len(velocitylist) != len(poslist):
                    velocitylist = [0., 0., 0.]
                poscar.addatom(elementList, poslist, poscar.get_direct(), fixlist, velocitylist)
                # set poscar to current latt
                poscar.isSelectiveDynamic = isSelectiveDynamic
                poscar.wrap()
                return poscar
        else:
            raise ValueError('Please ensure the validity of file path.')

    def write_to_poscar(self, 
                        file: str='POSCAR'):
        '''
        To output the current system into a POSCAR file
        '''
        s = self.to_poscar_string()
        print('wrote POSCAR into %s' % file)
        with open(file, 'w') as f:
            f.write(s)
    
    def to_poscar_string(self):
        '''
        Return a string with VASP POSCAR format
        '''
        # isOverlap = self.check_overlap()
        isOverlap = False
        if  isOverlap is False:
            # This means there is no overlap
            pass
        else:
            raise ValueError('Atom %s overlap with Atom %s' % isOverlap)
        if self.get_direct():
            self.wrap()
        s = '%s \n' % self.name
        s += '%f \n' % self.cell.scale
        for i in self.cell.lattvec:
            for j in i:
                s += '%.10f\t' % j
            s += '\n'
        formula_dict = get_formula(self.elements)[1]
        for i in formula_dict.keys():
            s += '    %s  ' % i
        s += '\n'

        for i in formula_dict.values():
            s += '    %s  ' % i
        s +='\n'
        
        if self.isSelectiveDynamic:
            s += 'SelectiveDynamics \n'
            if self.get_direct():
                s += 'Direct \n'
            else:
                s += 'Cartesian \n'
            poslist = self.poslist.reshape(-1,)
            fixlist = self.fix.reshape(-1,)
            for i in range(self.natom):
                for j in poslist[3*i:3*i+3]:
                    s += '\t%.10f\t' % j
                s += '%s' % get_fix_from_string(fix=fixlist[3*i:3*i+3], inverse=True)
                s += '%s%s\n' % (self.elements[i], i+1)
        else:
            if self.get_direct():
                s += 'Direct \n'
            else:
                s += 'Cartesian \n'
            poslist = self.poslist.reshape(-1,)
            for i in range(self.natom):
                for j in poslist[3*i:3*i+3]:
                    s += '\t%.10f\t' % j
                s += '%s%s\n' % (self.elements[i], i+1)
        # if velocity is not all zero, then write velocity as well
        velocity_bool = np.array(self.velocity, dtype=bool)
        write_velocity = velocity_bool.any()
        if write_velocity:
            s += '\n'
            velocitylist = self.velocity.reshape(-1,)
            for i in range(self.natom):
                for j in velocitylist[3*i:3*i+3]:
                    s += '%.10f\t' % j
                s += '\n'
        
        return s
        
        
    def set_fix_TopBottom(self, fix: bool =[True, True, False], element: str or list =None):
        '''
        Can be used to fix bottom and top atoms. 
        You can use the element options to specify which type of elements to fix
        '''
        self.wrap() # 这里有BUG,如果现在是cartesian坐标，那就不能做PBC了，但是这里
        if element is not None:
            if isinstance(element, str):
                eleMask = self.elements == element
            elif isinstance(element, (tuple, list, np.ndarray)):
                eleMask = [ ele in element for ele in self.elements ]
            eleIndexes = np.where(eleMask)[0]
            zlist = self.poslist[eleIndexes][:, 2]
        else:
            # if element is not assigned, use all atoms
            eleIndexes = np.arange(self.natom)
            zlist = self.poslist[:, 2]

        top = zlist.max()
        bottom = zlist.min()
        topIndexes = np.where(np.abs(zlist - top) < 0.01)[0]
        bottomIndexes = np.where(np.abs(zlist - bottom) < 0.01)[0]

        atoms_toFix = np.append(topIndexes, bottomIndexes)
        print('The TOP and Bottom atoms lists are: \n%s' % self.atomlist[eleIndexes][atoms_toFix])
        for atom in self.atomlist[eleIndexes][atoms_toFix]:
            atom.fix = fix
        self.isSelectiveDynamic = True

    def set_mobility(self, fix=[True, True, True], element=None, topBottom=False, topBottom_fix=[True, True, False]):
        '''
        Give the exiting POSCAR, export the mobility fixed POSCAR, 
        you can specify if the Top atoms are fixed or not
        '''
        assert(isinstance(fix, (tuple, list, np.ndarray))), 'Input fix has a wrong form! should be like [True, True, True]'
        fixlist = self.fix
        if element is not None:
            if isinstance(element, str):
                eleMask = self.elements == element
            elif isinstance(element, (tuple, list, np.ndarray)):
                eleMask = [ ele in element for ele in self.elements ]
            eleIndexes = np.where(eleMask)[0]
        else:
            eleIndexes = np.arange(self.natom)
        fixlist[eleIndexes] = fix
        self.fix = fixlist
        if topBottom:
            self.set_fix_TopBottom(topBottom_fix, element)
        self.isSelectiveDynamic = True
    

    def move_Part(self, DirectionVec: list, MoveDistance: float, lowlimit: float =-1E100, highlimit: float =1E100):
        '''To move the part (lowlimit< c < highlimit) toward the direction to somedistance(MoveDistance)\n
            Notice:
            1. the DirectionVec is the real vector, similar as the latticevector,  the atoms will
            move toward that direction
            2. And this step doesn't fix the selective mobility, please use the set_mobility function to fix the 
            mobility after the this function
            3. The MoveDistance is defined the cartesian distance after the shortest lattice vector is set to 1 by
            divide the three lattice vectors by the shortest lattice vector length.\n
            Example: move a part higher than 0.32 of the crystal to <a+b> direction with distance 0.76\n
                     moveDirec = latt.cell.lattvec[0]+latt.cell.lattvec[1]\n
                     latt.movePart(moveDirec, 0.7621023553, 0.32)
            '''
        # first get the length in each direction
        xlength, ylength, zlength, _, _, _ = self.cell.get_len_angle()
        minLength = min(xlength, ylength, zlength)
        # using the latticevectors to tranform the DirectionVec into current system
        transform_Matrix = np.linalg.inv(self.cell.lattvec/minLength)
        '''This is the transform matrix for coordinate transformation'''
        normalized_directionVec = DirectionVec/np.linalg.norm(DirectionVec)
        final_DirectionVec = np.matmul(MoveDistance*normalized_directionVec, transform_Matrix)
        
        # when discussing about the z coordinate with the low and high limit, do pbc first
        self.wrap()
        self.set_direct(True)
        newposlist = self.poslist.copy()
        zlist = self.poslist[:, 2]
        move_indexes = np.where((lowlimit<zlist) == (zlist<highlimit))
        newposlist[move_indexes] += final_DirectionVec
        self.poslist = newposlist
        self.wrap()

    def uniaxial_tensile(self, Direction: int, start_elongation: float, end_elongation: float, steps: int, relax:bool =True):
        '''
        Note that the tensile direction should be perpendicular to the plane made of other two lattice vectors,
        and relaxation can be applied on all atoms in the other two direction to minimize the system energy
        Direction: the tensile direction, can only be one of the lattice vectors. 0, 1, 2 represent the lattice vectors
        start_elongation: the starting total elongation rate of the tensile test (from 1, thus > 1 means tensile, < 1 means compress)
        end_elongation: the ending total elongation rate of the tensile test (from 1, thus > 1 means tensile, < 1 means compress)
        steps: how many steps to go to reach the required elongation
        relax: whether to relax the atoms after each step
        There are two types of tensile: (1) relax atoms perpendicular to the tensile direction after each step
                                        (2) fix all direction after each step.
        And the strain is equal to (x1-x0)/x0
        '''
        assert(Direction in [0, 1, 2]), 'Direction should be 0, 1, 2, indicating the 3 lattice vectors'
        assert(isinstance(steps, int) and steps > 0), 'Steps should be of int type and larger than 0'
        if start_elongation <= 0 or end_elongation <= 0:
            raise ValueError('Check the elongation')
        else:
            self.set_direct(True)
            mobility = [True, True, True]
            # mobility[Direction] = False
            if relax:
                self.set_mobility(fix=mobility)
            else:
                self.set_mobility(fix=[False, False, False])
            each_step = self.cell.lattvec[Direction]*(end_elongation - start_elongation)/steps
            self.cell.lattvec[Direction] = self.cell.lattvec[Direction]*start_elongation # set the cell to start position
            self.write_to_poscar('./Tensile_%s_%5.3f.vasp' % (str(0), start_elongation))

            for i in range(steps):
                self.cell.lattvec[Direction] += each_step
                print(self.cell.lattvec[Direction])
                self.write_to_poscar('./Tensile_%s_%5.3f.vasp' % (str(i+1), (i+1)*(end_elongation - start_elongation)/steps+start_elongation ))


    def moniclinic_shear(self, tilt_axis: int, toward_axis: int, initialStrain: float, endStrain: float, steps: int, relax=True):
        '''
        This function tilt axis 1 to axis 2 with a shear strain.
        tilt_axis: the lattice vector which is tilted
        toward_axis: the lattice vector which is the direction for shearing
        strain: the final strain
        steps: how many steps to reach the strain
        relax: whether to relax the atoms after each step
        There are two types of tensile: (1) relax atoms perpendicular to the tensile direction after each step
                                        (2) fix all direction after each step.
        Example: tilt zone axis c to a with 0.5 strain
                    strain = Proj[a]C/Proj[c0]C
        '''
        assert(isinstance(steps, int) and steps > 0), 'Steps should be of int type and larger than 0'
        if tilt_axis not in [0, 1, 2] or toward_axis not in [0, 1, 2]:
            raise ValueError('tilt_axis and toward_axis should be 0, 1, 2, indicating the 3 lattice vectors')
        else:
            self.set_direct(True)
            if relax:
                self.set_mobility([True, True, True])
            else:
                self.set_mobility(False, False, False)
            tilt_Axis_length = np.linalg.norm(self.cell.lattvec[tilt_axis])
            toward_Axis_length = np.linalg.norm(self.cell.lattvec[toward_axis])
            initialVector = (initialStrain*tilt_Axis_length/toward_Axis_length)*self.cell.lattvec[toward_axis]
            self.cell.lattvec[tilt_axis] += initialVector
            dispVector = (endStrain-initialStrain)*(tilt_Axis_length/toward_Axis_length)*self.cell.lattvec[toward_axis]
            self.write_to_poscar('./Shear_%s_%5.3f.vasp' % (str(0), initialStrain) )
            if steps != 0:
                each_step = dispVector/steps
                for i in range(steps):
                    self.cell.lattvec[tilt_axis] += each_step
                    self.write_to_poscar('./Shear_%s_%5.3f.vasp' % (str(i+1), (initialStrain + (i+1)*(endStrain-initialStrain)/steps) ))
