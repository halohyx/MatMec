
import numpy as np
from matmec.tool.latt_tool import periodic_table
from matmec.core.cell import Cell
from copy import deepcopy
import warnings

atom_propdict = ['element', 'pos', 'fix', 'velocity']

global UPPDATE_ITEM
UPPDATE_ITEM = {}

'''
BUG one problem that needs to be fixed is that when the property of one atom is updated, then
a loop will need to be run to update the property of all the atoms. Which is a waste of time.
'''

class Atom:

    __name__ = 'matmec.core.Atom'

    def __init__(self, 
                 element: str, 
                 pos=[.0, .0, .0], 
                 isDirect: bool=True,
                 fix: bool =[True, True, True], 
                 velocity=[.0, .0, .0]):
        '''
        Atom: an atom class contains index, element, pos, isDirect and cell\n
        Args:
            element: the element symbol of the atom, should be in the periodic table, str type
            pos: the position of the atom, list type
            isDirect: determine the coordinate type of input position, bool type
        '''
        self.propdict = {}
        self._set_element(element)
        self._set_pos(pos)
        self.__direct__ = isDirect
        self._set_fix(fix)
        self._set_velocity(velocity)
 
    def __repr__(self) -> str:
        s= ''
        if self.index is not None:
            s += '%03d' % self.index
        s += '%s ' % self.element
        for i in self.pos:
            s += '%05f ' % float(i)
        direct = 'direc' if self.get_direct() else 'cartes'
        s += '%s ' % direct
        return 'Atom( %s)' % s

    # element
    def _get_element(self):
        return self.get_propdict_value('element')
    def _set_element(self, element: str):
        assert(element in periodic_table), 'Illegal input of element'
        self.set_propdict('element', element)
        UPPDATE_ITEM['elements'] = True
    element = property(_get_element, _set_element, doc='The element symbol of this atom')

    # @atom_property: pos
    def _get_pos(self):
        return self.get_propdict_value('pos')
    def _set_pos(self, pos=None):
        assert(isinstance(pos, (list, tuple, np.ndarray)) and len(pos)==3), 'Length of pos should be 3'
        pos = np.array(pos, dtype=float)
        self.set_propdict('pos', pos)
        UPPDATE_ITEM['poslist'] = True
    def set_pos(self, pos=None, isDirect: bool=None):
        '''Default is to set pos in the current coordinate, if isDirect is set, \
            then the pos is in the new coordinate and \
            the self.direct will be changed according to the isDirect'''
        assert(len(pos)==3), 'Length of pos should be 3'
        pos = np.array(pos, dtype=float)
        self.set_propdict('pos', pos)
        UPPDATE_ITEM['poslist'] = True
        if isDirect is not None:
            self.set_direct(isDirect)
    pos = property(_get_pos, _set_pos, doc='The position of this atom')

    # velocity
    def _get_velocity(self):
        return self.get_propdict_value('velocity')
    def _set_velocity(self, v=None):
        assert(isinstance(v, (list, tuple, np.ndarray)) and len(v)==3), 'Length of velocity should be 3'
        v = np.array(v, dtype=float)
        self.set_propdict('velocity', v)
        UPPDATE_ITEM['velocity'] = True
    velocity = property(_get_velocity, _set_velocity, doc='The velocity of this atom')

    # index
    def _get_index(self):
        '''
        BUG need to find a way to solve the index 
        '''
        try: 
            index = np.where(self.latt.atomslist==self)
            self.index = index
        except:
            return None 
    def _set_index(self):
        raise ValueError('The index can not be set!\
                        it is automatically generated according to the position in atomslist of latt')
    index = property(_get_index, _set_index, doc='Index of this atom in the atomslist of latt')

    # direct
    def get_direct(self):
        return deepcopy(self.__direct__)
    def set_direct(self, isDirect: bool =True):
        assert(isinstance(isDirect, bool)), 'The isDirect should be of bool type'
        transferNeed = self.__direct__ == isDirect
        if not transferNeed:
            if self.__direct__:
                self.__DtoC()
            else:
                self.__CtoD()
            self.__direct__ = isDirect

    # fix
    def _get_fix(self):
        return self.get_propdict_value('fix')
    def _set_fix(self, fix: bool =[True, True, True]):
        assert(isinstance(fix, (list, tuple, np.ndarray)) and len(fix)==3), 'Input fix should be of list or tuple type and length should be 3'
        self.set_propdict('fix', fix)
        UPPDATE_ITEM['fix'] = True 
    fix = property(_get_fix, _set_fix, doc='The fix of this atom')

    # propdict
    def get_propdict_value(self, name):
        return self.propdict.get(name)
    def set_propdict(self, name, value):
        self.propdict[name] = value

    # method to change the direct and cartesian coordinate
    def __CtoD(self):
        assert(self.cell is not None), 'cell should be defined ahead of changing coordinate type'
        self.pos = self.cell.get_direct_coords(self.pos)[0]
    def __DtoC(self):
        assert(self.cell is not None), 'cell should be defined ahead of changing coordinate type'
        self.pos = self.cell.get_cartesian_coords(self.pos)[0]
    def move(self, 
             dispvec):
        '''
        Move towards a certain direction, can move in direct coordinate or cartesian coordinate\n
        Args:
            dispvec: displacement vector, can be of list or np.ndarray type, length should be 3
        '''
        # check if the moveVec is of list or np.ndarray type
        assert(isinstance(dispvec, (list, tuple, np.ndarray)) and len(dispvec)==3), 'Input moveVec should be of list or tuple type and length should be 3'
        print("Pos before movement: %s" % self.pos)

        self.pos += dispvec

        print("Pos after movement: %s" % self.pos)
        
    def copy(self):
        '''
        return the copy of current Atom
        '''
        newatom = Atom(self.element, self.pos, self.get_direct(), self.fix)
        return newatom
    
    def get_property(self, prop_name: str):
        return self.get_propdict_value(prop_name)

    def repeat(self, rep):
        atom = np.array(self.copy())
        return atom.repeat(rep)

    def __add__(self, other):
        atom = self.copy()
        other = np.array(other)
        return np.append(other, atom)
    
    def __mul__(self, rep):
        return self.repeat(rep)
    
    def __eq__(self, atom: object) -> bool:
        tol = 1E-6
        for prop in atom_propdict:
            if self.get_propdict_value(prop).dtype != bool:
                if not (abs(self.get_propdict_value(prop) - atom.get_propdict_value(prop)) < tol).all():
                    return False
            else:
                if self.get_propdict_value(prop) != atom.get_propdict_value(prop):
                    return False
        return True
