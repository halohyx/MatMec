import numpy as np
from copy import deepcopy
from matmec.core import Latt

# BUG BUG BUG Regulate all method names and variables names! Current system is very confusing!
class wcRandSol:
    '''
    The class adopting Warren-Cowley parameter to generate a random solution.
    Parameters:
        metric_method: the method to calculate the metric value, could be cross_prob, atomic_diff and std_prob
                        Currently only support cross_prob and std_prob
        latt: the Latt class object, you can assign this variable to assign the initial structure information
        max_neigh: the max neighbor shell that will be considered
        conv_thr: the convergence threshold. A single value or a list of values for each neighbor shell.
    '''
    def __init__(self,
                 metric_method: str = "cross_prob",
                 latt = None,
                 max_neigh: int = 2,
                 conv_thr: float =1e-5) -> None:
        
        # the convergence criterions
        # BUG conv_thr should be different for different metric
        if np.isscalar(conv_thr):
            self.conv_thr = np.ones(max_neigh)*conv_thr
        else:
            # you can also give a list of conv_thr for each neighbor shell
            self.conv_thr = np.array(conv_thr, dtype=float)
            assert len(self.conv_thr) == max_neigh, "The length of given conv_thr should be same as max_neigh"
        
        # the metric method
        self.metric_method = metric_method

        # how many neighbor shells would be included 
        self.max_neigh = max_neigh

        # add the structure object to current
        if latt is not None:
            self.natom = latt.natom
            self.ori_latt = latt
            self.sublat_indice = np.array(range(self.natom))
        # difference in latt and ori_latt is that, latt could be the sublattice,
        # while the ori_latt would be the latt class initially given
        self.latt = latt
        # final_latt would be the best structure found

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
        # store the original Latt as ori_latt
        self.ori_latt = tmplatt

        self.natom = self.latt.natom
        self.sublat_indice = np.array(range(self.natom))

    def select_sublattice(self,
                          sublattice):
        '''
        Selecting the sublattice that will be used as sites for elements mixing.
        Parameter:
            sublattice: element, all the sites that this element occupy will be regarded as the sublattice you choose
        Return:
            None
        '''
        if len(np.unique(self.latt.elements)) == 1: 
            print("**Bro, there is only one element, no need to select sublattice")
            return 0 
        # 两个点，第一个是把原始的结构找个地方存起来: 通过前面的ori_latt存起来了
        # 第二个是把sublattice选走的位置要记住，后面需要给定elements 找个通过存了一个sublat_indice存起来了
        
        if sublattice not in self.latt.elements:
            raise ValueError("Selected sublattice not in supplemented lattice.")

        # assign the indice of the sublattice
        sublat_indice = (self.latt.elements == sublattice).nonzero()[0]
        self.sublat_indice = sublat_indice

        # update the number of atoms
        self.natom = len(sublat_indice)

        # update the latt obeject as the selected sublattice
        new_atomlist = self.latt.atomlist[sublattice]
        selected_latt = Latt(new_atomlist, cell=self.latt.cell)
        self.latt = selected_latt

    def _get_neighbor_matrix(self):
        '''
        Get the neighbor matrix of the corresponding latt class
        '''
        self.neigh_level_mat, self.neigh_index_mat = deepcopy(self.latt.get_neighbor_matrix(max_neigh = 3,
                                                                                            memory_save = True,
                                                                                            cutoff_radius = 8.,
                                                                                            dis_mat_method = "translational",
                                                                                            verbose = True)[:2])

    def _get_natom(self):
        '''
        Get the neighbor matrix from latt object
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

        comp_list = np.array(list(comp_dict.values()), dtype=float)
        asked_conc_list = np.round(comp_list, decimals=3)

        if abs(comp_list.sum() - 100) > 0.001:
            raise ValueError("The sum of concentration should be 100, please check the comp_dict")

        comp_list = comp_list/100
        comp_atom_list = np.rint(comp_list*self.natom)              

        if comp_atom_list.sum() != self.natom:
            comp_atom_list[comp_atom_list.argmax()] -= (comp_atom_list.sum() - self.natom)
        comp_atom_list = np.array(comp_atom_list, dtype=int)

        comp_atom_dict = {}
        for i, j in zip(comp_dict.keys(), comp_atom_list):
            comp_atom_dict[i] = j

        goal_conc_list = np.round(100*comp_atom_list/self.natom, decimals=3)
        self.goal_conc_list = goal_conc_list

        if verbose:
            element_string = '    |    '.join(comp_dict.keys())
            asked_conc_string = '  |  '.join([f'{i:.3f}' for i in asked_conc_list])
            goal_conc_string = '  |  '.join([f'{i:.3f}' for i in goal_conc_list])
            occupied_atom_string = '    |    '.join([f'{i:2d}' for i in comp_atom_list])
            print()
            print(f'Asked concentration for   {element_string} (%): ')
            print(f'                        {asked_conc_string}')
            print(f'Goal  concentration for   {element_string} (%): ')
            print(f'                        {goal_conc_string}')
            print()
            print(f'Total {self.natom} atoms')
            print(f'Occupation is    {element_string} : ')
            print(f'                 {occupied_atom_string}')

        self.comp_atom_dict = comp_atom_dict
        self.comp_atom_list = comp_atom_list

        return comp_atom_dict, comp_atom_list
    
    def generate_random_occupation(self):
        '''
        Using the composition list, to generate a totally random occupied array.
        Parameter:
            None
        Return: 
            occup_array: specify the sites with different element
        '''
        # corresponds to the comp_dict, 1 represents the 1st element, 2 represen the 2nd...
        natom = self._get_natom()
        occup_array = np.zeros(natom, dtype=int)
        indice = np.array(range(natom))

        if not hasattr(self, "comp_atom_list"):
            raise ValueError("Please input the concentration first")
        else:
            comp_atom_list = deepcopy(self.comp_atom_list)

        for atomNum, Num in enumerate(comp_atom_list):
            replaceIndice = np.random.choice(indice, size=Num, replace=False)
            indice = np.setdiff1d(indice, replaceIndice)
            occup_array[replaceIndice] = atomNum+1

        return occup_array

    def shuffle_part_atoms(self,
                           num_shuffle: int,
                           occup_array: np.ndarray):
        '''
        Swap possitions of part of the atoms.
        input is an occupation array, return another array with partially shuffled
        Parameters:
            num_shuffle: number of atoms to shuffle
            occup_array: the occupation array that needs to be shuffle
        Return:
            occup_array: the shuffled occupation array
        '''
        natom = len(occup_array)
        indice = np.array(list(range(natom)))

        shuffled_indice = np.random.choice(indice, size=num_shuffle, replace=False)
        indice_before_shuffle = np.array(shuffled_indice)

        # this is done in place
        np.random.shuffle(shuffled_indice)

        # use the shuffled indice to replace the original indice
        occup_array[indice_before_shuffle] = occup_array[shuffled_indice]
        
        # return the shuffled occupation array
        return occup_array

    def compute_goal_prob_mat(self,
                            metric_method: str = None):
        '''
        The goal probability matrix for different metrics, representing an ideal mixing.
        Parameters:
            metric_method: cross_prob, std_prob
            For cross_prob:
                Ideally, the probability i, j pair is the product of the concentration of i and j. 
            For std_prob:
                Ideally, the probability i, j pair is the concentration of j.
        '''
        nele = len(self.comp_atom_list)

        # if metric_method is not given, use the default one stored in this class
        if metric_method is None:
            metric_method = self.metric_method
        
        if metric_method == "cross_prob":
            # the goal percentage of each element
            goal_conc_list = self.goal_conc_list/100

            # initialize the goal cross probability matrix
            goal_prob_mat = np.zeros((self.max_neigh, nele, nele))

            # the ideal distribution
            for ele1 in range(nele):
                for ele2 in range(nele):
                    goal_prob_mat[:, ele1, ele2] = goal_conc_list[ele1] * goal_conc_list[ele2]
            
            self.goal_prob_mat = goal_prob_mat

        elif metric_method == "std_prob":
            # the goal probability matrix
            goal_prob_mat = np.tile(self.comp_atom_list/self.natom, self.max_neigh*nele).reshape(self.max_neigh, nele, nele)

        return goal_prob_mat

    def compute_cross_prob_mat(self,
                                occup_array: np.ndarray):
        '''
        Compute the pair correlation for each element in each nearst neighbor shell.
        This is the cross probability method, to calculate the cross probability from finding all A-B pairs among all pairs 
        Say we have total 600 pairs, then first 600 atoms, we have Pa to find the type A atoms,
        and we need to find the type B atoms in the other 600 atoms (to make a pair), thus the result will come as Pa*Pb
        Parameters:
            occup_array: occupation array, the occupation array for all atoms
        Return:
            cross_prob_mat: the cross probability matrix for each element-pair in each nearst neighbor shell.
        '''

        # number of neighboring shells considered
        max_neigh = self.max_neigh

        nele = len(np.unique(occup_array))
        cross_prob_mat = np.zeros((max_neigh, nele, nele))

        # cross probability method: to calculate the crossed probability from finding all A-B pairs among all pairs 
        for nst_neigh_level in range(max_neigh):
            # loop for different nearst neighbor pairs
            pairI, pairJ_tmpindex = ( self.neigh_level_mat == nst_neigh_level+1).nonzero()

            # the 2nd index in neigh_level_mat doesn't mean the index of the atom
            # we should find the real index of the 2nd atom in the pair by locating neigh_index_mat
            pairJ = self.neigh_index_mat[pairI, pairJ_tmpindex]

            # 1st neighbor for example, say each atom has 6 atoms in 1st neighbor shell, then nPairs = 6*natom
            # we are indeed estimating how many A-A, B-B, C-C, ... pairs among all the nPairs
            nPairs = len(pairI)

            for ele1 in range(nele):
                for ele2 in range(nele):
                    # elePairCorr is an 1D array, each element indicate whether this level nearst neighbor pair are all this type of element
                    # the Pair Correlation result for each element is how many N type element pairs exist in all M pairs in the 1st nearst neighbor pairs
                    # array([1, 0, 1])*array([0, 1, 1]) = array([0, 0, 1]), thus we can use to determine how many atoms within the same ele in current shell
                    ele_pair_cross_prob_list = (occup_array[pairI] == ele1+1).astype(int) * (occup_array[pairJ] == ele2+1).astype(int)
                    # and say we have total 600 pairs, then first 600 atoms, we have Pa to find the type A atoms,
                    # and we need to find the type B atoms in the other 600 atoms (to make a pair), thus the 
                    # result will come as Pa*Pb
                
                    cross_prob_mat[nst_neigh_level, ele1, ele2] = ele_pair_cross_prob_list.sum() / nPairs
        
        return cross_prob_mat

    def generate_typenum_neigh_mat(self,
                                occup_array):
        '''
        Compute the atomic neighbor matrix in the format of (assume we have 3 types of elements)):
        [
            neighbor shell 1
            [3, 2, 1] # in this neighbor shell, how many atoms of each type (assume total 6 atoms in this neighbor)
            [4, 2, 0]
            [2, 3, 1]
            ...
        ],
        [
            neighbor shell 2 ...
        ]
        Parameters:
            occup_array: occupation array, the occupation array for all atoms
        Return:
            atomic_neigh_mat: the neighboring atoms counts of each atom at each neighboring shell.
        '''

        # a function to give the percentage of nele element in a list
        # using functions in numpy, be careful! This function will run billion time!
        def get_percentage(array,
                        nele):
            types, counts = np.unique(array, return_counts=True)
            if len(types) == nele:
                return counts
            else:
                output = np.zeros(nele, dtype=int)
                # Fill the concentrations based on the input types for those with 0% concentration
                for t, c in zip(types, counts):
                    output[t-1] = c
                return output

        # number of types of elements
        nele = len(self.comp_atom_list)

        # shape would be (max_neigh, natoms, ncomp) for the composition around each atom
        atomic_neigh_mat = np.zeros((self.max_neigh, self.natom, nele))

        # loop for different nearst neighbor pairs
        for nst_neigh_level in range(self.max_neigh):

            # BUG notice that the pairI, pairJ, seperators for each neighboring shell are always the same during the iteration
            # In the future, we can do some optimization on this part to save time.
            pairI, pairJ_index = ( self.neigh_level_mat == nst_neigh_level+1 ).nonzero()

            # the 2nd index in neigh_level_mat doesn't mean the index of the atom
            # we should find the real index of the 2nd atom in the pair by locating neigh_index_mat
            pairJ = self.neigh_index_mat[pairI, pairJ_index]

            # generate the seperators
            # iatoms will be like [0, 0, 0, 1, 1, 1, 1], seperators will be like [0, 3, 7] to seperate the iatoms and jatoms
            seperators = [0]
            flag = 0
            for i in range(len(pairI)):
                if pairI[i] != flag:
                    seperators.append(i)
                    flag += 1

            # the last seperator should be the length of pairI
            seperators.append(len(pairI))

            for atom_num in range(self.natom):
                tmp_neigh_atom_list_index = pairJ[seperators[atom_num]:seperators[atom_num+1]]

                tmp_neigh_atom_list = occup_array[tmp_neigh_atom_list_index]

                atomic_neigh_mat[nst_neigh_level, atom_num] = get_percentage(array=tmp_neigh_atom_list, 
                                                    nele=nele)
            
        return atomic_neigh_mat
    
    def compute_std_prob_mat(self,
                                occup_array):
        '''
        This method ultilize the atomic neighbor matrix to compute the standard probability matrix.
        % This is the standard definition of probability in the Warren-Cowley equations.
        Parameters:
            occup_array: occupation array, the occupation array for all atoms
        Return:
            prob_mat: the ele1-ele2 paired probability matrix for each neighbor shell. shape is (max_neigh, nele, nele)
        '''
        # firstly get the atomic neighbor matrix
        atomic_neigh_mat = self.generate_typenum_neigh_mat(occup_array)

        # define an empty probability matrix
        std_prob_mat = np.zeros((self.max_neigh, len(self.comp_atom_list), len(self.comp_atom_list)))

        # loop for different element pair probability
        for ele1 in range(len(self.comp_atom_list)):

            # screen out all the neighbor list for ele1 element
            ele1_std_prob_mat = atomic_neigh_mat[:, occup_array == ele1+1, :]

            # sum up the total number of neighbors for ele1 element
            total_neighbors_nums = ele1_std_prob_mat.sum(axis=1).sum(axis=1)

            # compute the probability for ele1-ele2 pair
            for ele2 in range(len(self.comp_atom_list)):
                # np.array(range(self.max_neigh)) as index so we can assign the values for different neighbors at the same time
                # this calculates the number of ele2 atoms in the ele1 neighbor list, and divide by the total number of current neighbor shell
                std_prob_mat[np.array(range(self.max_neigh)), ele1, ele2] = ele1_std_prob_mat[:, :, ele2].sum(axis=1)/total_neighbors_nums
            
        return std_prob_mat


    def compute_metric(self,
                       occup_array: np.ndarray,
                       metric_method: str = None,
                       ):
        '''
        Three types of metric_method, cross_prob, atomic_diff and std_prob.
        cross_prob:
            We would calculate how many i, j pairs among all pairs in different neighbor shells,
            then we compute the probability of seeing i-j pairs by dividing the total number of 
            i-j pairs by the total number of pairs. Ideal would be Pi*Pj.
        atomic_diff:
            firstly we calculate the element distribution of every atom in each neighbor shell and make it a matrix.
            Then we calculate the difference of the concentration in each neighbor shell near each atom and the 
            ideal concentration. Ideal would be comp_atom_list/natom. Then a mean difference of each element 
            would be output to represent the metric value.
        std_prob:
            Firstly we calculate the element distribution of every atom in each neighbor shell and make it a matrix.
            Then the matrix is used to calculate the concentration of atom j in the neighbor shell of atom i. And this
            yields the probability matrix. Ideal would be comp_atom_list/natom.
        Return:
            metric_value: the metric value for each neighbor shell regarding different metric method.
        '''

        # number of types of elements
        nele = len(self.comp_atom_list)

        # the metric method is not specified, use the one stored in this class
        if  metric_method is None:
            metric_method = self.metric_method

        if metric_method == "cross_prob":

            # the goal probability matrix
            goal_prob_mat = self.compute_goal_prob_mat() 

            # the current probability matrix
            current_ave_prob_mat = self.compute_cross_prob_mat(occup_array)

            # the diff_values between the current probability matrix and the goal probability matrix
            diff_values = np.abs(current_ave_prob_mat - goal_prob_mat).mean(axis=1)

            # metric value would be the weighted average of the diff_values
            metric_value = np.sum(diff_values * self.comp_atom_list/self.natom, axis=1)


        elif metric_method == "atomic_diff":
            # the atom-wise probability matrix 
            atomic_neigh_mat = self.generate_typenum_neigh_mat(occup_array)

            # the atom-wise probability matrix
            atomic_neigh_mat = np.array(atomic_neigh_mat, dtype=float) / np.sum(atomic_neigh_mat[0][0])

            # the atom-wise goal probability matrix
            goal_atomic_neigh_mat = np.tile(self.comp_atom_list/self.natom, self.max_neigh*self.natom).reshape(self.max_neigh, self.natom, nele)

            # the diff_values between the current atomic probability matrix and the goal atomic probability matrix
            diff_values = np.abs(atomic_neigh_mat - goal_atomic_neigh_mat).mean(axis=1)

            # metric value would be the weighted average of the diff_values
            metric_value = np.sum(diff_values * self.comp_atom_list/self.natom, axis=1)

        elif metric_method == "std_prob":
            # the current probability matrix
            current_prob_mat = self.compute_std_prob_mat(occup_array)

            # the goal probability matrix
            goal_prob_mat = np.tile(self.comp_atom_list/self.natom, self.max_neigh*nele).reshape(self.max_neigh, nele, nele)

            # the diff_values between the current probability matrix and the goal probability matrix
            diff_values = np.abs(current_prob_mat - goal_prob_mat).mean(axis=1)

            # metric value would be the weighted average of the diff_values
            metric_value = np.sum(diff_values * self.comp_atom_list/self.natom, axis=1)

        # metric value is for each neighbor shell
        return metric_value


    def run(self,
            max_iter: int = 1000,
            output_file: str = "matmec.out",
            verbose: bool = True):
        '''
        Start the iteration for generating wcRandSol geometry, final wcRandSol geometry will be stored at self.final_latt.
        A convergence cretiria between 1E-5 and 1E-6 is recommanded.
        Parameters:
            max_iter: the max iteration program will perform.
        '''
        # BUG BUG BUG Bring in the atoms swap method next time.

        # define the output file
        if isinstance(output_file, str):
            if output_file in ["screen", "Screen", "SCREEN", "sc"]:
                f = None
            else:
                f = open(output_file, 'w')
        
        # get the neighbor matrix
        if not hasattr(self, "neigh_level_mat"):
            self._get_neighbor_matrix()

        # get the goal probability matrix
        goal_prob_mat = self.compute_goal_prob_mat()

        # get the element list
        ele_label_list = list(self.comp_atom_dict.keys())
        nele = len(ele_label_list)

        # initialize the empty best_dict
        best_dict = {}

        def define_best_dict(occup_array, metric_value, step):
            best_dict['occup_array'] = occup_array
            best_dict['metric_value'] = metric_value
            best_dict['step'] = step
            return best_dict

        def generate_sqs_latt(latt, occup_array, sublat_indice, elementTypes):
            # generate the substitution elements list
            new_ele_list = np.array(latt.elements)
            sub_ele_list = []
            for i in range(self.natom):
                sub_ele = elementTypes[int(occup_array[i]-1)]
                sub_ele_list.append(sub_ele)    
            new_ele_list[sublat_indice] = sub_ele_list

            latt.elements = new_ele_list
            return new_ele_list

        # initialize the best structure dictionary
        best_dict = define_best_dict(None, [1E10]*self.max_neigh, -1)

        if verbose:
            # print the target probability matrix for each neighbor shell
            for nst_neigh_level in range(self.max_neigh):
                print(f'\nGoal probability for neighbor shell {nst_neigh_level}:', file=f)
                print('\t' + '\t'.join(ele_label_list), file=f)
                for ele in range(nele):
                    print(ele_label_list[ele] + '\t' + '\t'.join([f'{i:0.4f}' for i in goal_prob_mat[nst_neigh_level, ele]]), file=f)

            conv_thr_string = ', '.join([f'{i:6.3e}' for i in self.conv_thr])
            print(f'\nObjective function convergence criteria (Metric: {self.metric_method}): {conv_thr_string}', file=f)

        iter_num = -1

        if verbose:
            print('\n------------------------START ITERATION------------------------', file=f)
            obj_title = '\t|\t'.join([f'Neigh_shell.{str(m+1)}' for m in range(self.max_neigh)])
            print(f'\tstep\t|\t{obj_title}', file=f)

        converge = False

        # main loop
        while iter_num < max_iter:
            '''
            Two ways to end the iteration
            1): the max iteration reached, then the structure with smallest delta corr will be regarded as final structure
            2): the iteration converged, the structure with delta corr below the threshold found
            '''
            iter_num += 1

            # generate one random structure
            tmp_occup_array = self.generate_random_occupation()

            # compute the obejective function value
            tmp_metric_value = self.compute_metric(tmp_occup_array)

            # print the metric value
            if verbose:
                tmp_metric_value_string = '\t|\t'.join([f'   {m: 0.4f}' for m in tmp_metric_value])
                print(f'\t{iter_num}\t|\t{tmp_metric_value_string}', file=f)
            
            # if converged, iteration finish
            if np.all(tmp_metric_value <= self.conv_thr):
                print("Succeed", file=f)
                converge = True
                best_dict = define_best_dict(tmp_occup_array, tmp_metric_value, iter_num)
                break
            else:
                # if not converged, compare with the best structure
                if sum(tmp_metric_value) < sum(best_dict['metric_value']):
                    best_dict = define_best_dict(tmp_occup_array, tmp_metric_value, iter_num)

        # print the convergence information
        print(f'\n------------------------END ITERATION------------------------', file=f)
        if converge:
            print(f'\nConverged after {iter_num} iterations.', file=f)
        else:
            print(f'\nMax iteration reached ({max_iter}).', file=f)
            print(f'***ATTENTION, iteration not converged.***', file=f)

        print(f'\nBest structure found at step {best_dict["step"]}, average metric value: {sum(best_dict["metric_value"])/self.max_neigh:0.4f}', file=f)

        # calculate the final probability matrix
        if self.metric_method == 'std_prob':
            final_prob_mat = self.compute_std_prob_mat(best_dict['occup_array'])
        elif self.metric_method == 'cross_prob':
            final_prob_mat = self.compute_cross_prob_mat(best_dict['occup_array'])

        # print the final probability matrix
        for nst_neigh_level in range(self.max_neigh):
            print(f'\nProbability for neighbor shell {nst_neigh_level} (Metric: {self.metric_method}):', file=f)
            print('\t' + '\t'.join(ele_label_list), file=f)
            for ele in range(nele):
                print(ele_label_list[ele] + '\t' + '\t'.join([f'{i:0.4f}' for i in final_prob_mat[nst_neigh_level, ele]]), file=f)

        # generate and save the final structure as final_latt
        self.best_dict = best_dict
        from copy import deepcopy
        self.final_latt = deepcopy(self.ori_latt)

        new_ele_list = generate_sqs_latt(latt = self.final_latt, 
                                        occup_array = best_dict["occup_array"], 
                                        sublat_indice = self.sublat_indice, 
                                        elementTypes = list(self.comp_atom_dict.keys()))

        self.final_latt.elements = new_ele_list 

        # close the output file
        if isinstance(output_file, str) and output_file not in ["screen", "Screen", "SCREEN", "sc"]:
            f.close()
