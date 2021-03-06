
import numpy as np
from matmec.tool.latt_tool import periodic_table
from matmec.core.cell import Cell
import warnings

global UPPDATE_ITEM
UPPDATE_ITEM = {}


class Atom:

    __name__ = 'matmec.core.Atom'

    def __init__(self, element: str, pos=[.0, .0, .0], isDirect: bool=True, \
                cell: Cell=None, latt=None, fix: bool =[True, True, True], velocity=[.0, .0, .0]):
        '''
        Atom: an atom class contains index, element, pos, isDirect and cell\n
        Parameters:
        element: the element symbol of the atom, should be in the periodic table, str type
        pos: the position of the atom, list type
        isDirect: determine the coordinate type of input position, bool type
        cell: the cell that the atom belongs to, Cell type
        '''
        self.propdict = {}
        self._set_element(element)
        self._set_pos(pos)
        self.__direct__ = isDirect
        self._set_fix(fix)
        self._set_velocity(velocity)
        if latt is not None:
            self._set_latt(latt)
            self._set_cell(latt.cell)
            if cell is None:
                raise ValueError('If latt is set, then the cell should not be set at the same time')
        else:
            if cell is not None:
                self._set_cell(cell)
 
    def __repr__(self) -> str:
        s= ''
        if self.index is not None:
            s += '%03d' % self.index
        s += '%s ' % self.element
        for i in self.pos:
            s += '%05f ' % float(i)
        direct = 'direc' if self.__direct__ else 'cartes'
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

    # pos
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
            self.__direct__ = isDirect
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

    # cell
    def _get_cell(self):
        return self.get_propdict_value('cell')
    def _set_cell(self, cell: Cell):
        assert(cell.__name__ == 'matmec.core.Cell'), 'The cell should be of Cell type'
        self.set_propdict('cell', cell)
    cell = property(_get_cell, _set_cell, doc='The cell that current atom belongs to')
    
    # latt
    def _set_latt(self, latt=None):
        assert(latt.__name__ == 'matmec.core.Latt'), 'The latt should be of Latt type'
        self.set_propdict('latt', latt)
    def _get_latt(self):
        return self.get_propdict_value('latt')
    latt = property(_get_latt, _set_latt, doc='The latt that current atom belongs to')

    # index
    def _get_index(self):
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
        return self.__direct__
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
        transfromMattrix = np.linalg.inv(self.cell.lattvec*self.cell.scale)
        self.pos = np.matmul(np.array(self.pos), transfromMattrix)
    def __DtoC(self):
        assert(self.cell is not None), 'cell should be defined ahead of changing coordinate type'
        transfromMattrix = self.cell.lattvec*self.cell.scale
        self.pos = np.matmul(np.array(self.pos), transfromMattrix)

    def move(self, DirectionVec: list, distance: float, move_type: int=1):
        '''
        Move towards a certain direction, can move in direct coordinate or cartesian coordinate\n
        Parameters:
        DirectionVec: defines the direction vector of this move, this vector is defined in the cartesian coordinate
        distance: 
        move_type: 1 for Direct coordinate, 2 for Cartesian coordinate
        '''
        assert(self.cell is not None), 'cell should be defined ahead of moving atom'
        if move_type == 1:
            '''Move in a direct coordinate way'''
            [a, b, c, _, _, _] = self.cell.angle_calc()
            minLength = min(a,b,c)
            transformMatrix = np.linalg.inv(self.cell.lattvec/minLength)
            DirectionVec = DirectionVec/np.linalg.norm(DirectionVec)
            DispVec = distance*DirectionVec
            DispVec = np.matmul(DispVec, transformMatrix)
            self.set_direct(True)
            self.pos += DispVec

        elif move_type == 2:
            '''Move in a cartesian coordinate way'''
            self.set_direct(False)
            DirectionVec = DirectionVec/np.linalg.norm(DirectionVec)
            DispVec = distance*DirectionVec
            self.pos += DispVec

    def copy(self):
        '''
        return the copy of current Atom
        '''
        newatom = Atom(self.element, self.pos, self.__direct__, \
                self.cell, self.latt, self.fix)
        return newatom
    
    def repeat(self, rep):
        atom = np.array(self.copy())
        return atom.repeat(rep)

    def __add__(self, other):
        atom = self.copy()
        other = np.array(other)
        return np.append(other, atom)
    
    def __mul__(self, rep):
        return self.repeat(rep)
    
