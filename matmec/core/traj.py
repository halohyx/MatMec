import copy
from matmec.core import Latt, Cell
from matmec.core.slab import abcvector
from matmec.tool.latt_tool import periodic_table, get_elements_list_from_poscarString, \
                                  get_formula
from matmec.tool.mpl_tool import regular_ticks
import numpy as np

traj_prop = ['cell', "timestep", "timeseries", "poslist", "nframes", "elements", "comment", "prop_dict", "formula"]

class Traj:
    '''
    A class that record the trajectory of a molecular dynamic process, curenntly only support the trajectory of AIMD in vasp.
    In most cases you may want to initialize this class by using the read_from_XDATCAR method.
    Args:
        timestep: int type
            the unit of time
        timeseries: list_like
            a 1D numpy array that records the timeseries of each frame
        elements: list_like
            a 1D numpy array that records the element type of each atom
        cell: matmec.core.Cell type
            can either be one single Cell attribute (if the cell shape in the Traj doesn't change), \
            or a 1D numpy array that records the cell of each frame (if the cell shape change in the Traj)
        poslist: (nframes, natoms, 3) numpy array_like
            records the position of each atom in each frame
    Return:
        traj: a Traj object
    '''
    __name__ = "matmec.core.Traj"

    def __init__(self, cell=None, elements=None, poslist=None, timestep=1, timeseries=None, isDirect=True, lattList=None):
        self.propdict = {}
        # the priority in trajectory is elements, poslist, cell
        self.elements = elements

        if poslist is None:
            self.poslist = 1E100*np.ones(3).reshape(-1, 1, 3) # we initialize a initial array for poslist
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
    
    # @Traj_property: cellarray
    def _get_cellarray(self):
        if self.__variedCell:
            cellarray = [ cell.scale*cell.lattvec for cell in self.cell ]
        else:
            cellarray = np.array(self.cell.scale*self.cell.lattvec).reshape(1, 3, 3)
        return np.array(cellarray, dtype=float)
    cellarray = property(_get_cellarray, doc="The arrays of cell lattice vectors")

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
    def _set_nframes(self):
        raise RuntimeError('The nframes cannot be set! It can only be get!')
    nframes = property(_get_nframes, doc='The number of frames in current trajectory')

    # @Traj_property: timestep
    def _get_timestep(self):
        return self._get_propdict_value('timestep')
    def _set_timestep(self, timestep):
        try:
            timestep = float(timestep)
        except:
            raise ValueError('The timestep should be of int or float type')
        self._set_propdict("timestep", timestep)
    timestep = property(_get_timestep, _set_timestep, doc="The timestep of the frames")

    # @Traj_property: timeseries
    def _get_timeseries(self):
        return self._get_propdict_value('timeseries')
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
    # check whether this function can run properly later%%%%%%%%%%%%%%%%%%
    def wrap(self):
        '''
        Apply boundary condition in 3 diretions
        '''
        eps = 1E-7
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

    # @Traj_method: get trajectory from XDATCAR
    def read_from_XDATCAR(self, file='XDATCAR', skip_every=1, timestep=1, read_confs=None):
        '''
        In XDATCAR, when the cell shape and volume doen't change, then the system name
        and the cell vectors won't be printed in each configuration
        Args:
            file: Path-like or string
                file path of the XDATCAR file. Default is 'XDATCAR'
            skip_every: int type
                take the trajectory for each skip_every frames. Default is 1, which means take all frames.
            timestep: int type
                timestep of each frame. Default is 1 fs.
            read_confs: int type
                read the first read_confs configurations. Default is None, which means read all configurations.
        Return:
            a Traj object
        '''
        with open(file, 'r') as f:
            name = f.readline()[:-2] # to get rid of the ending /n
            __writeElements = True
            poslist = []
            celllist = []
            count = 0
            if read_confs is None:
                read_confs = 1E100
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
                    print([_cellVec1, _cellVec2, _cellVec3])
                    _cell = Cell(lattvec=[_cellVec1, _cellVec2, _cellVec3], scale=_scale)
                    celllist.append(_cell)
                _pos = [ [ float(j) for j in f.readline().split()] for i in range(natoms) ]
                poslist.append(_pos)
                del _pos
                name = f.readline()[:-2]
                count += 1
                if count >= read_confs:
                    break
            if len(celllist) == 1 and len(poslist) != 1:
                celllist = celllist[0]
                poslist = poslist[0::skip_every]
            elif len(celllist) == 1 and len(poslist) == 1:
                celllist = celllist[0]
                poslist = [poslist[0]]
            elif len(celllist) != 1 and len(celllist) == len(poslist):
                celllist = np.array(celllist[0::skip_every], dtype=Cell)
                poslist = np.array(poslist[0::skip_every], dtype=float)
            else:
                raise ValueError("Please report this bug to Yixuan :)")
            timestep = timestep*skip_every
            self.__init__(cell=celllist, elements=elementslist, poslist=poslist, timestep=timestep, \
                          isDirect=_isDirect)
            return self.__repr__()

    # Traj_method: projection on some 2D plane
    def get_projection(self, ob_direc, x_direc):
        r"""
        Project the coordinates on a given lattice plane
        Args: 
            ob_direc: 
                the normal vector of the projection lattice plane, \
                can be given in the tradition of hkl in str: "hkl" or int number: hkl\
                or a real vector

            x_direc: 
                the X direction of the projection lattice plane, \
                should be given in the tradition of hkl in str: "hkl" or \
                int number: hkl, and notice that this x_direct \
                vector should be perpendicular of ob_direct. Thus, the 3rd basis \
                vector of the camera coordinate is then defined as the cross of \
                ob_direc and x_direc, and all of them will then be normalized.
        
        Return: 
            the 2D cartesian coordinates on the projection plane
        """
        ob_direc = abcvector(ob_direc, self.cell)
        x_direc = abcvector(x_direc, self.cell)
        # the 3rd vector comes from the cross of the given two vectors
        y_direc = np.cross(x_direc, ob_direc)

        # normalize x,y and ob_direc
        if len(ob_direc.shape) == 1:
            ob_direc = ob_direc/np.linalg.norm(ob_direc)
            x_direc = x_direc/np.linalg.norm(x_direc)
            if np.abs(np.sum(np.dot(ob_direc, x_direc.T))) > 1.5e-2:
                raise ValueError("The x_direc vector doesnt lie on the normal plane of ob_direc")
            y_direc = y_direc/np.linalg.norm(y_direc)
            camera_coord = np.array([x_direc, y_direc, ob_direc])
        else:
            ob_direc = ob_direc/np.linalg.norm(ob_direc, axis=1)
            x_direc = x_direc/np.linalg.norm(x_direc, axis=1)
            if np.abs(np.sum(np.dot(ob_direc, x_direc.T))) > 1.5e-2:
                raise ValueError("The x_direc vector doesnt lie on the normal plane of ob_direc")
            y_direc = y_direc/np.linalg.norm(y_direc, axis=1)
            camera_coord = np.array([[x_direc[i], y_direc[i], ob_direc[i]] for i in range(len(ob_direc))])
        # the transform_tensor will be the original basis vector multiplied by the inverse of camera_coord
        # transform_tensor = np.matmul(self.cellarray, np.linalg.inv(camera_coord))
        # the transform_tensor should be current_coordinate*np.linalg.inv(camera_coord), but in the case of 
        # cartesian coordinates, the current_coordinate is unit matrix, so it changes to below
        transform_tensor = np.linalg.inv(camera_coord)
        self.set_direct(False)
        # the coordinates on the projection plane
        newcoordinates = np.matmul(self.poslist, transform_tensor)[:, :, :2]
        return newcoordinates
        
    # Traj_method: plotting projection
    def plot_projection(self, ob_direc, 
                              x_direc, 
                              xlim=None, 
                              ylim=None, 
                              selected=None, 
                              name='projection', 
                              cmap=None, 
                              style='seaborn-whitegrid'):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.ticker as mticker

        '''
        For plotting the projection along the direction specified. The projection definition can be found in Traj.get_projection()
        Args:
            ob_direc: str: "hkl" or int number: hkl or a list type real vector
                see Traj.get_projection(). Default: 100 (Hope in your cell the 100 are vertical to 001, otherwise the default setting may cause error!)
            x_direc: str: "hkl" or int number: hkl or a list type real vector
                see Traj.get_projection() Deafult: 001
            xlim: list type. Optional
                the xlim of the plot. Default: None
            ylim: list type. Optional
                the ylim of the plot. Default: None
            selected: list type. Optional
                the selected elements to be plotted. Default: None, all will be plotted.
            name: str type. Optional
                by default we will save the figure, the name of the saved figure file. Default: 'projection'
            cmap: list like. Optional
                list of single colormap. A list of colormap will be used for each element. Default: None
            style: str type. Optional.
                style in matplotlib, call the plt.style.use(style). Default: 'seaborn-whitegrid'
        Return:
            None
        '''

        default_cmap_list = ['winter', 'cool','magma','cool', 'Blues', 'Greens', 'Oranges', 'Reds','cool', 'hyx1']

        default_cmap_list = [
    "winter",
    'cool',
    'magma',
    'cividis',
    'coolwarm',
    'twilight',
    'seismic',
    'twilight_shifted',
    'hsv',
    'Pastel1',
    'Pastel2',
    'tab10',
    'tab20',
    'tab20b',
    'tab20c',
    'rainbow',
]

        projection = self.get_projection(ob_direc, x_direc)
        elements = np.unique(self.elements)
        num_of_cmap = len(elements)

        if cmap is None:
            cmap = default_cmap_list

        if not isinstance(cmap, (str, mpl.colors.Colormap, list, np.ndarray)):
            raise ValueError('Pls provide correct type of cmap, could be of str or colormap type.')

        fig = plt.figure(figsize=(15, 9))
        plt.style.use(style)
        gs = GridSpec(1, 15)

        norm = mpl.colors.Normalize(vmin=self.timeseries[0]*self.timestep, vmax=self.timeseries[-1]*self.timestep)
        c = np.arange(len(self.timeseries))/len(self.timeseries)

        ax = fig.add_subplot(gs[:15-num_of_cmap])

        # the interaction words
        print(f'Plotting figure ...')

        if selected == None:
            for index, ele in enumerate(np.unique(self.elements)):
                # for i in [273, 73, 65, 153, 74]:
                whereEle = np.where(self.elements == ele)[0]
                # for i in whereEle:
                _c = c.repeat(len(whereEle)).ravel()
                ax.scatter(projection[:, whereEle, 0], projection[:, whereEle, 1], s=70, c=_c, alpha=0.5, cmap=default_cmap_list[index], marker='.')
        else:
            selected = np.array(selected)
            elementslist = self.elements[selected]
            selected_projection = projection[:, selected, :]
            for index, ele in enumerate(np.unique(elementslist)):
                num_of_cmap = len(np.unique(elementslist))
                # for i in [273, 73, 65, 153, 74]:
                whereEle = np.where(elementslist == ele)[0]
                # for i in whereEle:
                _c = c.repeat(len(whereEle)).ravel()
                ax.scatter(selected_projection[:, whereEle, 0], selected_projection[:, whereEle, 1], s=70, c=_c, alpha=0.8, cmap=default_cmap_list[index], marker='.')

        # set the properties of the plotting ax
        # hide all spines, set facecolor black, turn of grid
        # set fonts, set xlabel and ylabel and higher ylim
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        ax.set_facecolor('k')
        ax.grid(False)
        title_font = {'family': 'calibri', 'size': 16}
        tickslabel_font = {'family': 'arial', 'size': 14}

        ax.set_xlabel(r"$Distance\ (\AA)$", fontdict=title_font)
        ax.set_ylabel(r"$Distance\ (\AA)$", fontdict=title_font)

        # make the ticks increase from 0, from bottom to top, from left to right
        regular_ticks(ax, 'x')
        regular_ticks(ax, 'y')
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        for i in range(num_of_cmap):
            ax = fig.add_subplot(gs[15-num_of_cmap+i])
            fig.subplots_adjust(wspace=0.1)
            cbar = mpl.colorbar.Colorbar(ax, mpl.cm.ScalarMappable(cmap=default_cmap_list[i], norm=norm), \
                                        ticklocation='right',\
                                        orientation='vertical'\
                                        )
            # for the last colorbar, set the ticks
            if i == num_of_cmap-1:
                # cbar.set_label('Time, 60 ps', loc='top', fontdict=title_font)
                yticks = cbar.get_ticks()
                cbar.set_ticks(yticks[:-1], fontdict=title_font)
                pass
            # for the middle colorbar set the title
            elif i == int(num_of_cmap/2):
                tickslabel_font['size'] = 18
                title = ax.set_title(r"$Time\ (fs)$", fontdict=tickslabel_font)
                cbar.set_ticks([])
            else:
                cbar.set_ticks([])
            ax.set_xlabel(elements[i], fontdict=title_font)
        print(f'Saving figure...')
        plt.savefig('%s.jpeg' % name)
        print(f'Figure saved as {name}.jpeg')