import copy
from enum import Flag
from multiprocessing.sharedctypes import Value
from pickle import FALSE
from time import time
from xml.dom.minidom import Element
from matmec.utils import reader, writer
from matmec.core import Latt, Cell
from matmec.tool.latt_tool import periodic_table, get_elements_list_from_poscarString, \
                                  get_formula
import numpy as np

traj_prop = ['cell', "timestep", "timeseries", "poslist", "nframes", "elements", "comment", "prop_dict", "formula"]



class Traj:
    '''
    All the following attribute will be stored in a propdict
    timestep: the unit of time
    timeseries: a 1D numpy array that records the time of each frame
    elements: a 1D numpy array that records the element type of each atom
    cell: be of matmec.core.Cell type, can either be one single Cell attribute (if the cell shape in the Traj doesn't change), \
        or a 1D numpy array that records the cell of each frame (if the cell shape change in the Traj)
    poslist: documented in (nframes, natoms, 3) numpy array, records the position of each atom in each frame
    '''
    __name__ = "matmec.core.Traj"

    def __init__(self, cell=None, elements=None, poslist=None, timestep=1, timeseries=None, isDirect=True, lattList=None):
        self.propdict = {}
        # the priority in trajectory is elements, poslist, cell
        self.elements = elements

        if poslist is None:
            self.poslist = 1e100*np.ones(3).reshape(-1, 1, 3) # we initialize a initial array for poslist
        else:
            self.poslist = poslist
        self.timestep = timestep
        self.timeseries = timeseries

        self.cell = cell
        self.__direct__ = isDirect

    
    # @Traj_property: cell
    def _get_cell(self):
        return self._get_propdict_value('cell')
    def _set_cell(self, cell):
        '''
        If give a set of cells, then the number of cells should be of the length as nframes
        '''
        if cell is None:
            self._set_propdict("cell", None)
        else:
            if isinstance(cell, (list, np.ndarray)):
                assert(self.nframes != 0), "The cell list should be of the same length as nframes, but now nframes = 0"
                for index, c in enumerate(cell):
                    assert(isinstance(c, Cell)), "The %sth cell should be of Cell type" % index
                assert(len(cell) == self.nframes), "The cell list should be of the same length as nframes"
                self.__variedCell = True
                self._set_propdict("cell", cell)
            elif isinstance(cell, Cell):
                self.__variedCell = False
                self._set_propdict("cell", cell)
            else:
                raise ValueError("pls provide the correct form of cell")
    cell = property(_get_cell, _set_cell, doc='The cell that frames belongs to')
    
    # @Traj_property: poslist
    def _get_poslist(self):
        return self._get_propdict_value('poslist')
    def _set_poslist(self, poslist):
        '''
        If you give the poslist directly, then it should satisfy several requirements:
        1. it should be of 3 dimentions
        2. each position in the poslist should be of length 3
        3. each frame should be of the same length with the elements list in Traj class
        '''
        poslist = np.array(poslist, dtype=float)
        assert(len(poslist.shape) == 3), "The poslist supplied should be of 3 dimentions"
        assert(poslist.shape[-1] == 3), "Each position in the poslist should be of length 3"
        if self.elements is None:
            natoms = len(poslist[0])
            for index, pos in enumerate(poslist):
                assert(len(pos) == natoms), "The %sth frame has different atoms the others" % index+1
            self._set_propdict("poslist", poslist)
        else:
            nelements = len(self.elements)
            for index, pos in enumerate(poslist):
                assert(len(pos) == nelements), "The %sth frame has different atoms with the elments list" % index+1
            self._set_propdict("poslist", poslist)
    poslist = property(_get_poslist, _set_poslist, doc='The poslist of all the frames recorded in the trajectory')

    # @Traj_property: elements
    def _get_elements(self):
        return self._get_propdict_value('elements')
    def _set_elements(self, elements):
        '''
        You can set the elements freely (no matter the length), but the poslist should be\
            given whose length should match the length of elements
        '''
        if elements is None:
            self._set_propdict('elements', elements)
        else:
            for element in elements:
                assert(element in periodic_table.keys()), "%s not in periodic table, pls check the supplied elements" % element
            self._set_propdict('elements', elements)
    elements = property(_get_elements, _set_elements, doc='elements list of this trajectory')

    # @Traj_property: formula
    def _get_formula(self):
        if self.elements is not None:
            return get_formula(self.elements)[0]
        else:
            return None
    formula = property(_get_formula, doc='Formula of the system')
    

    # @Traj_property: nframes
    def _get_nframes(self):
        ifposlistEmpty = np.array(self.poslist == 1e100*np.ones(3).reshape(-1, 1, 3)).all()
        if ifposlistEmpty:
            self._set_propdict("nframes", 0)
        elif self.poslist is None:
            self._set_propdict("nframes", 0)
        else:
            self._set_propdict("nframes", len(self.poslist))
        return self._get_propdict_value("nframes")
    nframes = property(_get_nframes, doc='The number of frames in current trajectory')

    # @Traj_property: timestep
    def _get_timestep(self):
        self._get_propdict_value('timestep')
    def _set_timestep(self, timestep):
        try:
            timestep = float(timestep)
        except:
            raise ValueError('The timestep should be of int or float type')
        self._set_propdict("timestep", timestep)
    timestep = property(_get_timestep, _set_timestep, doc="The timestep of the frames")

    # @Traj_property: timeseries
    def _get_timeseries(self):
        self._get_propdict_value('timeseries')
    def _set_timeseries(self, timeseries):
        if timeseries is None:
            timeseries = np.arange(self.nframes)
        else:
            assert(len(timeseries) == self.nframes), "The timeseries should be of the same length as nframes"
        self._set_propdict("timeseries", timeseries)
    timeseries = property(_get_timeseries, _set_timeseries, doc="Timeseries of the frames")

    # @Traj_property: comment
    def get_comment(self):
        return self._get_propdict_value('comment')
    def set_comment(self, comment: str):
        assert(isinstance(comment, str)), 'The comment should be of str type'
        self._set_propdict('comment', comment)
    comment = property( get_comment, set_comment, doc='The comment of the trajectory')

    # @Traj_property: direct
    def get_direct(self):
        return self.__direct__
    def set_direct(self, isDirect: bool =True):
        assert(isinstance(isDirect, bool)), 'The isDirect should be of bool type'
        transferNeed = self.__direct__ == isDirect
        if not transferNeed:
            if not len(self.poslist) == 0: 
                if self.__direct__:
                    self.__DtoC()
                else:
                    self.__CtoD()
                self.__direct__ = isDirect
            else:
                self.__direct__ = isDirect

    # @Traj_method: method to change the direct and cartesian coordinate
    def __CtoD(self):
        assert(self.cell is not None), 'cell should be defined ahead of changing coordinate type'
        if self.__variedCell:
            cellTensor = np.array([ cell.lattvec*cell.scale for cell in self.cell ])
            transformTensor = np.linalg.inv(cellTensor)
            self.poslist = np.matmul(np.array(self.poslist), transformTensor)
        else:
            transformMattrix = np.linalg.inv(self.cell.lattvec*self.cell.scale)
            self.poslist = np.matmul(np.array(self.poslist), transformMattrix)
    def __DtoC(self):
        assert(self.cell is not None), 'cell should be defined ahead of changing coordinate type'
        if self.__variedCell:
            cellTensor = np.array([ cell.lattvec*cell.scale for cell in self.cell ])
            transformTensor = cellTensor
            self.poslist = np.matmul(np.array(self.poslist), transformTensor)
        else:
            transformMattrix = self.cell.lattvec*self.cell.scale
            self.poslist = np.matmul(np.array(self.poslist), transformMattrix)

    # @Traj_property: propdict
    def _get_propdict_value(self, name):
        return copy.deepcopy(self.propdict.get(name))
    def _set_propdict(self, name, value=None):
        # set value as None to delete the current item
        if value is not None:
            self.propdict[name] = value
        else:
            if name in self.propdict:
                del self.propdict[name]
            else:
                self.propdict[name] = None

    # @Traj_method: represent the Traj class
    def __repr__(self) -> str:
        s = 'matmec.Traj  '
        s += 'formula: %s; ' % self.formula
        s += 'nframes: %s' % self.nframes
        if self.comment is not None:
            s += "/n%s" % self.comment
        return s


    # @Traj_method: periodic boundary condition
    def pbc(self):
        '''
        Apply boundary condition in 3 diretions
        '''
        poslist = np.array(self.poslist)
        while (poslist>=1).any():
            id1, id2 = np.where(poslist>=1)
            poslist[id1, id2] -= 1
        while (poslist<0).any():
            id1, id2 = np.where(poslist<0)
            poslist[id1, id2] +=1
        self.poslist = poslist

    # @Traj_method: get trajectory from XDATCAR
    def read_from_XDATCAR(self, file='XDATCAR', skipEvery=1, timestep=1, readConfs=None):
        '''
        In XDATCAR, when the cell shape and volume doen't change, then the system name
        and the cell vectors won't be printed in each configuration
        file: name of the XDATCAR file
        skipEvery: take the trajectory for each skipEvery frames
        timestep: timestep of each frame
        '''
        with open(file, 'r') as f:
            name = f.readline()[:-2] # to get rid of the ending /n
            __writeElements = True
            poslist = []
            celllist = []
            count = 0
            if readConfs is None:
                readConfs = 1E100
            natoms = None
            while name:
                if name[:21] != "Direct configuration=":
                    _sysname = name
                    _scale = float(f.readline().split()[0]) # read the scale factor
                    _cellVec1 = [float(i) for i in f.readline().split()] # read the lattice vector
                    _cellVec2 = [float(i) for i in f.readline().split()] # read the lattice vector
                    _cellVec3 = [float(i) for i in f.readline().split()] # read the lattice vector
                    if __writeElements:
                        _elementsNames = [str(i) for i in f.readline().split()] # read the elements name
                        _elementCounts = [int(i) for i in f.readline().split()] # read the elements counts
                        elementslist = get_elements_list_from_poscarString(_elementsNames, _elementCounts)
                        natoms = len(elementslist)
                        if f.readline()[0] == 'D':
                            _isDirect = True
                        else:
                            _isDirect = False
                        __writeElements = False
                    else:
                        f.readline()
                        f.readline()
                        f.readline()
                    _cell = Cell(_scale, lattvec=[_cellVec1, _cellVec2, _cellVec3])
                    celllist.append(_cell)
                _pos = [ [ float(j) for j in f.readline().split()] for i in range(natoms) ]
                poslist.append(_pos)
                del _pos
                name = f.readline()[:-2]
                count += 1
                if count >= readConfs:
                    break
            if len(celllist) == 1 and len(poslist) != 1:
                celllist = celllist[0]
                poslist = poslist[0::skipEvery]
            elif len(celllist) != 1 and len(celllist) == len(poslist):
                celllist = np.array(celllist[0::skipEvery], dtype=Cell)
                poslist = np.array(poslist[0::skipEvery], dtype=float)
            else:
                raise ValueError("Please report this bug to Yixuan :)")
            timestep = timestep*skipEvery
            self.__init__(cell=celllist, elements=elementslist, poslist=poslist, timestep=timestep, \
                          isDirect=_isDirect)
            return self.__repr__()

    