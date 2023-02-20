import numpy as np
from copy import deepcopy
from matmec.core import Latt

class SQS:
    '''
    Special Quasirandom Structures (SQS)
    '''
    def __init__(self,
                 latt = None,
                 maxNeigh: int = 2,
                 conv_thr: float =1e-5) -> None:
        # BUG 这里加功能，要么直接给latt，或者sqs也可以直接read from poscar然后返回一个sqs对象
        if np.isscalar(conv_thr):
            self.conv_thr = np.ones(maxNeigh)*conv_thr
        else:
            self.conv_thr = np.array(conv_thr, dtype=float)
            assert len(self.conv_thr) == maxNeigh, "The length of given conv_thr should be same as maxNeigh"

        self.maxNeigh = maxNeigh
        if latt is not None:
            self.natom = latt.natom
            self.oriLatt = latt
            self.sublat_indice = np.array(range(self.natom))
        self.latt = latt
        # sqsLatt is the sqs that will be output

    def read_structure(self, 
                       filePath = "./POSCAR",
                       supercell = [1, 1, 1]):
        '''
        Read structure from file.
        Parameters:
        filePath: the path of the structure file.
        supercell: whether to make it a supercell
        '''
        tmplatt = Latt.read_from_poscar(filePath)
        if not supercell == [1, 1, 1]:
            tmplatt = tmplatt.get_supercell(supercell=supercell)
        self.latt = tmplatt
        self.oriLatt = tmplatt

        self.natom = self.latt.natom
        self.sublat_indice = np.array(range(self.natom))

    def select_sublattice(self,
                          sublattice):
        '''
        Selecting the sublattice that will be used as sites for elements mixing.
        Parameter:
        sublattice: element, all the sites that this element occupy will be regarded as the sublattice you choose
        Return:
        0
        '''
        if len(np.unique(self.latt.elements)) == 1: 
            print("**Bro, there is only one element, no need to select sublattice")
            return 0 
        # 两个点，第一个是把原始的结构找个地方存起来: 通过前面的oriLatt存起来了
        # 第二个是把sublattice选走的位置要记住，后面需要给定elements 找个通过存了一个sublat_indice存起来了
        
        if sublattice not in self.latt.elements:
            raise ValueError("Selected sublattice not in supplemented lattice.")

        sublat_indice = (self.latt.elements == sublattice).nonzero()[0]
        self.sublat_indice = sublat_indice

        self.natom = len(sublat_indice)

        newAtomlist = self.latt.atomlist[sublattice]
        selectedLatt = Latt(newAtomlist, cell=self.latt.cell)
        self.latt = selectedLatt

        return 0

    def generate_neighbor_matrix(self):
        '''
        Generate the neighbor matrix of the corresponding latt class
        '''
        self.neigh_mat = deepcopy(self.latt.get_neighbor_matrix())

    def _get_natom(self):
        '''
        Generate the neighbor matrix of the corresponding latt class
        '''
        self.natom = int(self.latt.natom)
        return self.natom

    def input_concentration(self,
                            comp_dict: dict, 
                            verbose:bool =True):
        '''
        Using the given concentration to calculate the real concentration, and give the real number of atoms in the system.
        Parameters:
        comp_dict: dict type, keys are the elements, values are corresponding concentration in %
        verbose: whether output in a verbose way
        Return: 
        comp_atom_dict: dictionary containing the number of each element, keys are elements, values are the corresponding \
        number of atoms
        comp_atom_list: numbers of each element in the system
        '''
        natom = self._get_natom()
        comp_list = np.array(list(comp_dict.values()), dtype=float)
        asked_conc_list = np.round(comp_list, decimals=3)

        if abs(comp_list.sum() - 100) > 0.001:
            raise ValueError("The sum of concentration should be 100, please check the comp_dict")

        comp_list = comp_list/100
        comp_atom_list = np.rint(comp_list*natom)              

        if comp_atom_list.sum() != natom:
            comp_atom_list[comp_atom_list.argmax()] -= (comp_atom_list.sum() - natom)
        comp_atom_list = np.array(comp_atom_list, dtype=int)

        comp_atom_dict = {}
        for i, j in zip(comp_dict.keys(), comp_atom_list):
            comp_atom_dict[i] = j

        real_conc_list = np.round(100*comp_atom_list/natom, decimals=3)
        self.real_conc_list = real_conc_list

        if verbose:
            element_string = '    |    '.join(comp_dict.keys())
            asked_conc_string = '  |  '.join([f'{i:.3f}' for i in asked_conc_list])
            real_conc_string = '  |  '.join([f'{i:.3f}' for i in real_conc_list])
            occupied_atom_string = '    |    '.join([f'{i:2d}' for i in comp_atom_list])
            print()
            print(f'Asked concentration for   {element_string} (%): ')
            print(f'                        {asked_conc_string}')
            print(f'Real  concentration for   {element_string} (%): ')
            print(f'                        {real_conc_string}')
            print()
            print(f'Total {natom} atoms')
            print(f'Occupation is    {element_string} : ')
            print(f'                 {occupied_atom_string}')

        self.comp_atom_dict = comp_atom_dict
        self.comp_atom_list = comp_atom_list

        return comp_atom_dict, comp_atom_list
    
    def _generate_random_occupation(self,
                                   comp_atom_list: np.ndarray):
        '''
        Generate a totally random occupied array.
        Parameter:
        comp_atom_list: how many atoms for each element.
        Return: 
        occup_array: specify the occupation of each element
        '''
        # corresponds to the comp_dict, 1 represents the 1st element, 2 represen the 2nd...

        natom = self._get_natom()
        occup_array = np.zeros(natom)
        indice = np.array(range(natom))

        for atomNum, Num in enumerate(comp_atom_list):
            replaceIndice = np.random.choice(indice, size=Num, replace=False)
            indice = np.setdiff1d(indice, replaceIndice)
            occup_array[replaceIndice] = atomNum+1

        return occup_array

    def _shuffle_part_atoms(self,
                           numToShuffle: int,
                           occup_array: np.ndarray):
        '''
        Swap possitions of part of the atoms.
        input is an occupation array, return another array with partially shuffled
        Parameters:
        numToShuffle: number of atoms to shuffle
        occup_array: the occupation array that needs to be shuffle
        Return:
        occup_array: the shuffled occupation array
        '''
        natom = len(occup_array)
        indice = np.array(list(range(natom)))

        shuffledIndice = np.random.choice(indice, size=numToShuffle, replace=False)
        indiceToShuffle = np.array(shuffledIndice)

        np.random.shuffle(shuffledIndice)
        occup_array[indiceToShuffle] = occup_array[shuffledIndice]

        return occup_array

    def _compute_ref_corr_mat(self):
        '''
        The reference pair correlation.
        Ideally, one element with concentration X should have pair correlation of X**2 
        '''
        ref_corr_list = list(self.real_conc_list/100 ** 2)
        ref_corr_mat = np.array(ref_corr_list * self.maxNeigh).reshape(len(self.comp_atom_list), self.maxNeigh)
        self.ref_corr_mat = ref_corr_mat
        return ref_corr_mat

    def _compute_pair_corr(self,
                          occup_array):
        '''
        Compute the pair correlation for each element in each nearst neighbor shell.
        Parameters:
        occup_array: occupation array, the occupation array for all atoms
        Return:
        corr_matrix: the correlation matrix for each element in each nearst neighbor shell.
        '''
        # input: 1) neighbor matrix; 2) occupation array; 3) nearst neighbor level

        neigh_mat = self.neigh_mat
        maxNeigh = self.maxNeigh

        nTypeEle = len(np.unique(occup_array))
        corr_matrix = np.zeros((nTypeEle, maxNeigh))

        for nstNeighLevel in range(maxNeigh):
            # loop for different nearst neighbor pairs
            # find the pairs for nearst neighbor level n
            pairI, pairJ = ( neigh_mat == nstNeighLevel+1).nonzero()
            nPairs = len(pairI)
            for ele in range(nTypeEle):
                # elePairCorr is an 1D array, each element indicate whether this level nearst neighbor pair are all this type of element
                # the Pair Correlation result for each element is how many N type element pairs exist in all M pairs in the 1st nearst neighbor pairs
                elePairCorr_list = (occup_array[pairI] == ele+1).astype(int) * (occup_array[pairJ] == ele+1).astype(int)
                corr_matrix[ele, nstNeighLevel] = elePairCorr_list.sum() / nPairs
        
        return corr_matrix


    def iterate_sqs(self,
                    maxIteration: int = 1000,
                    verbose: bool = True):

        if not hasattr(self, "neigh_mat"):
            self.generate_neighbor_matrix()
        
        # the initial occupation array
        self.occup_array = self._generate_random_occupation(self.comp_atom_list)

        if verbose:
            print("Starting iteration for the bese SQS")

        ref_corr_mat = self._compute_ref_corr_mat()

        def get_ave_corr(corr_mat):
            return np.mean(corr_mat, axis=0)

        def get_delta_corr(corr_mat):
            return np.mean((corr_mat-ref_corr_mat)**2, axis=0)

        iterationNum = -1

        if verbose:
            conv_str = '     |     '.join([f'AveCorr. {str(m+1)}' for m in range(self.maxNeigh)])
            delta_conv_str = '     |     '.join([f'AvedCorr. {str(m+1)}' for m in range(self.maxNeigh)])
            print()
            print(f'        step   |     {conv_str}     |     {delta_conv_str}')

        while iterationNum < maxIteration:

            iterationNum += 1
            
            # generate one random structure
            tmp_occup_array = self._generate_random_occupation(self.comp_atom_list)

            # compute the pair correlation
            tmp_corr_mat = self._compute_pair_corr(tmp_occup_array)

            # compute the difference of the curren corr_mat and refrence corr_mat
            delta_corr = get_delta_corr(tmp_corr_mat)

            ave_corr_list = get_ave_corr(tmp_corr_mat)
            ave_dcorr_list = get_ave_corr(delta_corr)

            if verbose:
                mcorr_str = '  |  '.join([ f'{corr: 12.6e}' for corr in ave_corr_list])
                delta_corr_str = '   |  '.join([ f'{dcorr: 12.6e}' for dcorr in ave_dcorr_list])
                print(f'    {iterationNum:8d}   |  {mcorr_str}  |  {delta_corr_str}',end='')

            # if converged, iteration finish
            if np.all(delta_corr_str <= self.conv_thr):

                # sqsLatt is the final structure
                self.sqsLatt = deepcopy(self.oriLatt)

                # generate the substitution elements list
                new_ele_list = np.array(self.sqsLatt.elements)
                sub_ele_list = []
                for i in range(self.natom):
                    sub_ele = list(self.comp_atom_dict.keys())[int(occup_array[i]-1)]
                    sub_ele_list.append(sub_ele)
                new_ele_list[self.sublat_indice] = sub_ele_list
                
                self.sqsLatt.elements = new_ele_list

            pass