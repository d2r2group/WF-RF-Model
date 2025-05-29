"""Work Function Random Forest Model
This module provides a class for predicting work functions of surfaces
using a pre-trained Random Forest model.
"""

import pandas as pd
import numpy as np
import math
import statistics
import joblib
import json
import warnings
from os.path import join
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.core.surface import SlabGenerator, Slab
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)  # To ignore sklearn warnings

with open(join(str(Path(__file__).absolute().parent), 'atomic_features', 'firstionizationenergy.txt'), 'r') as f:
    content = f.readlines()
fie: list = [float(x.strip()) for x in content]

with open(join(str(Path(__file__).absolute().parent), 'atomic_features', 'mendeleev.txt'), 'r') as f:
    content = f.readlines()
mendeleev: list = [float(x.strip()) for x in content]


class WFRFModel:
    """Class for predicting work functions using a pre-trained Random Forest model."""

    def __init__(self, model_filename: str = 'RF_1748260280.629787.joblib'):
        """Initialize the WFRFModel class and load the pre-trained model and scaler.
        :param model_filename: (str) Name of the joblib file containing the pre-trained Random Forest model
        """
        try:
            self.model = joblib.load(join(str(Path(__file__).absolute().parent), model_filename))
        except FileNotFoundError:
            raise FileNotFoundError('ML model joblib file not found. Please, download the most recent one '
                                    'from here: https://zenodo.org/doi/10.5281/zenodo.10449567')

        # Load feature scaling from training
        with open(join(str(Path(__file__).absolute().parent), 'scaler.json'), 'r') as jf:
            scaler_json = json.load(jf)
        scaler_load = json.loads(scaler_json)
        self.sc = StandardScaler()
        self.sc.scale_ = scaler_load['scale']
        self.sc.mean_ = scaler_load['mean']
        self.sc.var_ = scaler_load['var']
        self.sc.n_samples_seen_ = scaler_load['n_samples_seen']
        self.sc.n_features_in_ = scaler_load['n_features_in']
        self.features_labels = ['f_angles_min', 'f_chi', 'f_1_r', 'f_fie', 'f_fie2', 'f_fie3', 'f_mend',
                                'f_z1_2', 'f_z1_3', 'f_packing_area', 'f_chi_min', 'f_chi2_max',
                                'f_1_r_min', 'f_fie2_min', 'f_mend2_min']

    @staticmethod
    def group_indices_by_layers(positions: np.ndarray, frac_tol: float) -> list[list[int]]:
        """Group indices of atoms by their layers based on their fractional coordinates in the z-direction.
        :param positions: (np.ndarray) Array of fractional coordinates of atoms in the structure
        :param frac_tol: (float) Tolerance in fractional coordinates to determine which atoms belong to the same layer
        :return: (list[list[int]]) List of lists, where each inner list contains indices of atoms in the same layer
        """
        counter: int = 0
        index_list: list = []
        while counter < positions.shape[0]:
            # Find index for atom(s) with lowest c-position
            surface: float = max(positions[:, 2])
            highest_indices: list = []
            for ind, p in enumerate(positions):
                if p[2] > surface - frac_tol:
                    highest_indices.append(ind)
            # Once the index/indices of highest atom(s) is/are found,
            # set that position to zero for the next while loop iteration
            # and increase counter by the number of highest found indices.
            if len(highest_indices) > 0:
                index_list.append(highest_indices)
                for ind in highest_indices:
                    positions[ind] = [0, 0, 0]
                counter += len(highest_indices)
            else:
                print('Warning: No highest index found. Counter = ' + str(counter), flush=True)
                return []
        return index_list

    @staticmethod
    def featurize(struc: Structure, tol: float = 0.4) -> np.ndarray:
        """Featurize a slab structure to extract features for work function prediction.
        :param struc: (Structure) Slab structure as a pymatgen Structure object
        :param tol: (float) Tolerance in Angstroms to determine which atoms belong to the same layer (default 0.4 A)
        :return: (np.ndarray) Array of features extracted from the slab structure
        """
        # Tolerance tol in Angstrom

        # Alternative solution: [list(s.species.get_el_amt_dict().keys())[0] for s in struc.sites]
        for el in [s.species.elements[0].symbol for s in struc.sites]:
            if el in ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'At', 'Rn', 'Fr', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']:
                print('Warning: Structure contains element not supported for featurization.', flush=True)
                return np.array(
                    [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                     None, None, None, None])

        ftol: float = tol / struc.lattice.c
        if len(struc.sites) > 3:
            pos = struc.frac_coords
            indices_list = WFRFModel.group_indices_by_layers(pos, ftol)
            if len(indices_list) == 0:
                return np.array(
                    [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                     None, None, None, None])

            # Check there are at least 3 layers, given tolerance to group layers
            elif len(indices_list) < 3:
                print('Warning: Slab less than 3 atomic layers in z-direction, '
                      'with a tolerance = ' + str(tol) + ' A.', flush=True)
                return np.array(
                    [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                     None, None, None, None])

            pos = struc.frac_coords

            # Check if structure is of form slab with minimum vacuum of 5 A in z-direction
            min_vac = 5.0  # Angstrom
            if max(pos[:][2]) - min(pos[:][2]) * struc.lattice.c + min_vac > struc.lattice.c:
                print('Warning: Input structure either has no vacuum between slabs '
                      'or is not oriented in z-direction', flush=True)
                return np.array(
                    [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                     None, None, None, None])
        else:
            print('Warning: Slab less than 4 atomic layers in z-direction before applying tolerance.', flush=True)
            return np.array([None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                             None, None, None, None])

        # Add features
        chem: list = [s.species.elements[0].symbol for s in struc.sites]
        cell: list = list(struc.lattice.lengths) + list(struc.lattice.angles)

        # Refer to top or bottom surface index:
        sindex: int = 0
        sindex2: int = 1
        sindex3: int = 2

        # Feature Layer 1
        f_chi: list = []
        f_1_r: list = []
        f_fie: list = []
        f_mend: list = []
        f_angles: list = []

        for ind in range(len(indices_list[sindex])):
            f_chi.append(Element(chem[indices_list[sindex][ind]]).X)
            arc = Element(chem[indices_list[sindex][ind]]).atomic_radius_calculated
            ar = Element(chem[indices_list[sindex][ind]]).atomic_radius
            if arc is not None:
                f_1_r.append(1 / arc)
            elif ar is not None:
                f_1_r.append(1 / ar)
            else:
                f_1_r.append(np.nan)  # If no atomic radius is available, append NaN
            f_fie.append(fie[Element(chem[indices_list[sindex][ind]]).Z])
            f_mend.append(mendeleev[Element(chem[indices_list[sindex][ind]]).Z])

            # # Angle feature
            frac_coords1 = struc[indices_list[sindex][ind]].frac_coords
            shortest_distance: float = math.inf
            # Get site index of the closest site, and in which image, and its distance to reference top atom
            for si, site in enumerate(struc):
                if si not in indices_list[sindex]:
                    # print(type(struc[si]), flush=True)
                    frac_coords2 = struc[si].frac_coords
                    distance, image = struc.lattice.get_distance_and_image(frac_coords1, frac_coords2)
                    if distance < shortest_distance:
                        shortest_distance = distance
                        closest_site_index = si
                        closest_site_image = image
            # Get vector to the closest atom
            vector_closest_frac = (struc[closest_site_index].frac_coords +
                                   closest_site_image) - struc[indices_list[sindex][ind]].frac_coords  # pointing down
            vector_closest = struc.lattice.get_cartesian_coords(vector_closest_frac)
            vector_closest /= np.linalg.norm(vector_closest)
            a, b, c = struc.lattice.matrix
            surface_normal = np.cross(a, b)
            surface_normal /= -np.linalg.norm(surface_normal)  # pointing down
            angle = np.rad2deg(np.arccos(np.clip(np.dot(vector_closest, surface_normal), -1.0, 1.0)))
            f_angles.append(angle)

        f_angles_min = min(f_angles)

        f_packing_area = len(indices_list[sindex]) / (cell[0] * cell[1] * math.sin(cell[5]))

        # Features layer 2
        f_z1_2 = abs(pos[indices_list[sindex][0]][2] - pos[indices_list[sindex2][0]][2]) * cell[2]
        f_chi2 = []
        f_fie2 = []
        f_mend2 = []

        for ind2 in range(len(indices_list[sindex2])):
            f_chi2.append(Element(chem[indices_list[sindex2][ind2]]).X)
            f_fie2.append(fie[Element(chem[indices_list[sindex2][ind2]]).Z])
            f_mend2.append(mendeleev[Element(chem[indices_list[sindex2][ind2]]).Z])

        # Features layer 3
        f_z1_3 = abs(pos[indices_list[sindex][0]][2] - pos[indices_list[sindex3][0]][2]) * cell[2]
        f_fie3 = []

        for ind3 in range(len(indices_list[sindex3])):
            f_fie3.append(fie[Element(chem[indices_list[sindex3][ind3]]).Z])

        f_chi_min = min(f_chi)
        f_chi2_max = max(f_chi2)
        f_1_r_min = min(f_1_r)
        f_fie2_min = min(f_fie2)
        f_mend2_min = min(f_mend2)

        return np.array([f_angles_min, statistics.mean(f_chi), statistics.mean(f_1_r),
                         statistics.mean(f_fie), statistics.mean(f_fie2), statistics.mean(f_fie3),
                         statistics.mean(f_mend),
                         f_z1_2, f_z1_3, f_packing_area, f_chi_min, f_chi2_max, f_1_r_min, f_fie2_min, f_mend2_min])

    @staticmethod
    def mirror_slab(slab: Structure) -> Structure:
        """Mirror the slab structure in the z-direction.
        :param slab: (Structure) Slab structure as a pymatgen Structure object
        :return: (Structure) Mirrored slab structure
        """
        coords_reversed_z = []
        species = [sf.species for sf in slab]
        for site in slab:
            c = [1 - s if i == 2 else s for i, s in enumerate(site.frac_coords)]
            coords_reversed_z.append(c)
        return Structure(slab.lattice.matrix, species, coords_reversed_z, coords_are_cartesian=False)

    @staticmethod
    def generate_slabs_from_bulk(bulk: Structure, miller: tuple[int, int, int], tol: float = 0.4) -> list[Slab]:
        """Generate slabs from a bulk structure and a Miller index.
        :param bulk: (Structure) Bulk structure as a pymatgen Structure object
        :param miller: (tuple[int, int, int]) Miller index of slab to generate
        :param tol: (float) Tolerance in Angstroms to determine which atoms belong to the same layer (default 0.4 A)
        :return: (list) List of slab structures generated from the bulk structure
        """
        n = 1
        while True:
            slab = SlabGenerator(bulk, miller, n, 10, in_unit_planes=True).get_slabs()[0]
            ftol = tol / slab.lattice.c
            if len(WFRFModel.group_indices_by_layers(slab.frac_coords, ftol)) > 3:
                break
            else:
                n += 1
        return SlabGenerator(bulk, miller, n, 10, in_unit_planes=True).get_slabs()

    @staticmethod
    def get_elements_topmost_layer(slab: Structure, tol: float = 0.4) -> str:
        """Get the chemical elements of the topmost layer of a slab structure.
        :param slab: (Structure) Slab structure as a pymatgen Structure object
        :param tol: (float) Tolerance in Angstroms to determine which atoms belong to the same layer (default 0.4 A)
        :return: (str) String of unique chemical elements in the topmost layer, separated by hyphens
        """
        topelementsstring = []
        ftol = tol / slab.lattice.c
        slab_ind = WFRFModel.group_indices_by_layers(slab.frac_coords, ftol)
        for ind in slab_ind[0]:
            topelementsstring.append(slab.species[ind].symbol)
        return '-'.join(list(set(topelementsstring)))

    def predict_work_functions_from_slab(self, slab: Structure, tol: float = 0.4, 
                                        significant_digits: int = 4,) -> tuple[float, float]:
        """Predict the work functions (of top and bottom surface) from a single slab structure.
        :param slab: (Structure) Slab structure as a pymatgen Structure object
        :param tol: (float) Tolerance in Angstroms to determine which atoms belong to the same layer (default 0.4 A)
        :param significant_digits: (int) Number of significant digits to round the predicted work function (default 4)
        :return: (tuple[float, float]) Predicted work function of the top and bottom surfaces of the slab
        """
        feat_df = pd.DataFrame(columns=self.features_labels)
        features_top = self.featurize(slab, tol=tol)
        feat_df.loc[0, 'f_angles_min':'f_mend2_min'] = features_top
        x_top = self.sc.transform([feat_df.loc[0, :].values.tolist()])
        features_bottom = self.featurize(self.mirror_slab(slab), tol=tol)
        feat_df.loc[1, 'f_angles_min':'f_mend2_min'] = features_bottom
        x_bottom = self.sc.transform([feat_df.loc[1, :].values.tolist()])
        return round(float(self.model.predict(x_top)[0]), significant_digits), \
            round(float(self.model.predict(x_bottom)[0]), significant_digits)

    def predict_work_functions_from_bulk_and_miller(self,
                                                   bulk: Structure,
                                                   miller: tuple[int, int, int],
                                                   tol: float = 0.4,
                                                   significant_digits: int = 4) -> dict[str, float]:
        """Predict the work functions from a bulk structure and a Miller index.
        :param bulk: (Structure) Bulk structure as a pymatgen Structure object
        :param miller: (tuple[int, int, int]) Miller index of slab to generate
        :param tol: (float) Tolerance in Angstroms to determine which atoms belong to the same layer (default 0.4 A)
        :return: (dict[str, float]) Dictionary with keys as
            '<termination number (i.e., 0, 1, 2,...)>, <bottom/top>, <terminating chemical species (e.g., Cs-Hg)>'
            and the values are the respective predicted WFs
        """
        feat_df: pd.DataFrame = pd.DataFrame(columns=self.features_labels, dtype=float)
        slabs: list[Slab] = self.generate_slabs_from_bulk(bulk, miller, tol=tol)
        surface_elements: dict[int, str] = {}
        si: int = 0
        for ind, s in enumerate(slabs):
            surface_elements[si] = self.get_elements_topmost_layer(s, tol=tol)
            features_top = self.featurize(s, tol=tol)
            feat_df.loc[si, 'f_angles_min':'f_mend2_min'] = features_top
            si += 1
            s_mirror = self.mirror_slab(s)
            surface_elements[si] = self.get_elements_topmost_layer(s_mirror, tol=tol)
            features_bottom = self.featurize(s_mirror, tol=tol)
            feat_df.loc[si, 'f_angles_min':'f_mend2_min'] = features_bottom
            si += 1
        # feat_final = feat_df.round(8).drop_duplicates()
        x = self.sc.transform(feat_df)
        WFs = self.model.predict(x)
        results_dict: dict[str, float] = {}
        ri = 0
        for i in feat_df.index:
            location: str = "top" if i % 2 == 0 else "bottom"
            results_dict[f"{ri},{location},{surface_elements[i]}"] = round(float(WFs[ri]), significant_digits)
            if i % 2 != 0:
                ri += 1
        return results_dict


if __name__ == '__main__':
    pass
