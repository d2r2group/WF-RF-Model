
import pandas as pd
import numpy as np
import numpy.typing as npt
import math
import statistics
import joblib
import json
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import StandardScaler


with open('atomic_features/firstionizationenergy.txt') as f:
    content = f.readlines()
fie = [float(x.strip()) for x in content]

with open('atomic_features/mendeleev.txt') as f:
    content = f.readlines()
mendeleev = [float(x.strip()) for x in content]


class WFRFModel:

    def __init__(self):
        try:
            self.model = joblib.load('RF_1691469908.2138267.joblib')
        except FileNotFoundError:
            raise FileNotFoundError('ML model joblib file not found. Please, download it from here:')

        # Load feature scaling from training
        with open('scaler.json', 'r') as jf:
            scaler_json = json.load(jf)
        scaler_load = json.loads(scaler_json)
        self.sc = StandardScaler()
        self.sc.scale_ = scaler_load['scale']
        self.sc.mean_ = scaler_load['mean']
        self.sc.var_ = scaler_load['var']
        self.sc.n_samples_seen_ = scaler_load['n_samples_seen']
        self.sc.n_features_in_ = scaler_load['n_features_in']

    @staticmethod
    def featurize(struc: Structure, tol: float = 0.4) -> npt.ArrayLike:
        # Tolerance tol in Angstrom

        # Alternative solution: [list(s.species.get_el_amt_dict().keys())[0] for s in struc.sites]
        for el in [s.species.elements[0].symbol for s in struc.sites]:
            if el in ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'At', 'Rn', 'Fr', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']:
                print('Warning: Structure contains element not supported for featurization.', flush=True)
                return np.array(
                    [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                     None, None, None, None])

        # struc *= (2,2,1)
        ftol = tol / struc.lattice.c
        if len(struc.sites) > 3:
            pos = struc.frac_coords
            counter = 0
            indices_list = []
            while counter < len(pos):
                # Find index for atom(s) with lowest c-position
                surface = max(pos[:, 2])
                highest_indices = []
                for ind, p in enumerate(pos):
                    if p[2] > surface - ftol:
                        highest_indices.append(ind)
                # Once the index/indices of highest atom(s) is/are found,
                # set that position to zero for the next while loop iteration
                # and increase counter by the number of highest found indices.
                if len(highest_indices) > 0:
                    indices_list.append(highest_indices)
                    for ind in highest_indices:
                        pos[ind] = [0, 0, 0]
                    counter += len(highest_indices)
                else:
                    print('Warning: No highest index found. Counter = ' + str(counter), flush=True)
                    return np.array(
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                         None, None, None, None])

            # Check there are at least 3 layers, given tolerance to group layers
            # print('Indices list = {}'.format(indices_list))
            if len(indices_list) < 3:
                print('Warning: Slab less than 3 atomic layers in z-direction, '
                      'with a tolerance = ' + str(tol) + ' A.', flush=True)
                return np.array(
                    [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                     None, None, None, None])

            pos = struc.frac_coords

            # Check if structure is of form slab with minimum vacuum of 5 A in z-direction
            min_vac = 5.0  # Angstrom
            # print('max - min = {}'.format((max(p[2] for p in pos) - min(p[2] for p in pos)) * struc.lattice.c))
            # print('c length = {}'.format(struc.lattice.c))
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
        chem = [s.species.elements[0].symbol for s in struc.sites]
        cell = list(struc.lattice.lengths) + list(struc.lattice.angles)
        # pos = struc.get_positions()  # already defined above

        # Refer to top or bottom surface index:
        sindex = 0
        sindex2 = 1
        sindex3 = 2
        # sindex4 = 3

        # Feature Layer 1
        f_chi = []
        f_1_r = []
        f_fie = []
        f_mend = []
        f_angles = []

        for ind in range(len(indices_list[sindex])):
            f_chi.append(Element(chem[indices_list[sindex][ind]]).X)
            if Element(chem[indices_list[sindex][ind]]).atomic_radius_calculated:
                f_1_r.append(1 / Element(chem[indices_list[sindex][ind]]).atomic_radius_calculated)
            else:
                f_1_r.append(1 / Element(chem[indices_list[sindex][ind]]).atomic_radius)
            f_fie.append(fie[Element(chem[indices_list[sindex][ind]]).Z])
            f_mend.append(mendeleev[Element(chem[indices_list[sindex][ind]]).Z])

            # # Angle feature
            frac_coords1 = struc[indices_list[sindex][ind]].frac_coords
            shortest_distance: float = math.inf
            # Get site index of the closest site, and in which image, and its distance to reference top atom
            for si, site in enumerate(struc):
                if si not in indices_list[sindex]:
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
        # f_area = cell[0] * cell[1] * math.sin(cell[5])

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

    def predict_work_function(self, slab: Structure) -> float:
        features_labels = ['f_angles_min', 'f_chi', 'f_1_r', 'f_fie', 'f_fie2', 'f_fie3', 'f_mend', 'f_z1_2', 'f_z1_3',
                           'f_packing_area', 'f_chi_min', 'f_chi2_max', 'f_1_r_min', 'f_fie2_min', 'f_mend2_min']
        feat_df = pd.DataFrame(columns=features_labels)
        features = self.featurize(slab, tol=0.4)
        feat_df.loc[0, 'f_angles_min':'f_mend2_min'] = features
        X = self.sc.transform([feat_df.loc[0, features_labels].tolist()])
        return self.model.predict(X)[0]


if __name__ == '__main__':
    # Tests:
    slab_dict1 = ("{'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0, 'lattice': "
                  "{'matrix': [[1.497946409343668, -2.5945192879986, 0.0], [1.497946409343668, 2.5945192879986, 0.0], "
                  "[0.0, 0.0, 29.90092893148512]], 'a': 2.995892818687336, 'b': 2.995892818687336, 'c': "
                  "29.90092893148512, 'alpha': 90.0, 'beta': 90.0, 'gamma': 119.99999999999999, 'volume': "
                  "232.41698140866004}, 'sites': [{'species': [{'element': 'Cd', 'occu': 1}], 'abc': "
                  "[0.3333333333333333, 0.6666666666666666, 0.25082832768123936], 'xyz': [1.497946409343668, "
                  "0.8648397626662, 7.5], 'label': 'Cd', 'properties': {}}, {'species': [{'element': 'Cd', "
                  "'occu': 1}], 'abc': [0.3333333333333333, 0.6666666666666666, 0.45016566553624787], 'xyz': "
                  "[1.497946409343668, 0.8648397626662, 13.460371572594047], 'label': 'Cd', 'properties': {}}, "
                  "{'species': [{'element': 'Cd', 'occu': 1}], 'abc': [0.3333333333333333, 0.6666666666666666, "
                  "0.6495030033912563], 'xyz': [1.497946409343668, 0.8648397626662, 19.420743145188094], 'label': "
                  "'Cd', 'properties': {}}, {'species': [{'element': 'Cd', 'occu': 1}], 'abc': [0.6666666666666669, "
                  "0.3333333333333334, 0.3504969966087436], 'xyz': [1.4979464093436685, -0.8648397626662002, "
                  "10.480185786297023], 'label': 'Cd', 'properties': {}}, {'species': [{'element': 'Cd', 'occu': 1}], "
                  "'abc': [0.6666666666666669, 0.3333333333333334, 0.5498343344637522], 'xyz': [1.4979464093436685, "
                  "-0.8648397626662002, 16.440557358891073], 'label': 'Cd', 'properties': {}}, {'species': "
                  "[{'element': 'Cd', 'occu': 1}], 'abc': [0.6666666666666669, 0.3333333333333334, "
                  "0.7491716723187606], 'xyz': [1.4979464093436685, -0.8648397626662002, 22.40092893148512], "
                  "'label': 'Cd', 'properties': {}}]}")
    slab_dict2 = ("{'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0, 'lattice': "
                  "{'matrix': [[6.409284534725741e-16, 3.985569468855207, 2.440457446406473e-16], "
                  "[0.0, 0.0, 3.985569468855207], [28.94909458404634, 0.0, 1.7726208010283163e-15]], "
                  "'a': 3.985569468855207, 'b': 3.985569468855207, 'c': 28.94909458404634, 'alpha': 90.0, "
                  "'beta': 90.0, 'gamma': 90.0, 'volume': 459.84953522276135}, 'sites': [{'species': [{'element': "
                  "'Tl', 'occu': 1}], 'abc': [0.0, 1.7432710452650972e-37, 0.2590754601400626], 'xyz': [7.5, 0.0, "
                  "4.592425496802574e-16], 'label': 'Tl', 'properties': {}}, {'species': "
                  "[{'element': 'Tl', 'occu': 1}], 'abc': [0.0, 4.1357490726540435e-33, 0.39675055934857567], "
                  "'xyz': [11.485569468855207, 0.0, 7.032882943209048e-16], 'label': 'Tl', 'properties': {}}, "
                  "{'species': [{'element': 'Tl', 'occu': 1}], 'abc': [0.0, 0.0, 0.5344256585570888], 'xyz': "
                  "[15.471138937710418, 0.0, 9.473340389615522e-16], 'label': 'Tl', 'properties': {}}, "
                  "{'species': [{'element': 'Tl', 'occu': 1}], 'abc': [0.0, 0.0, 0.6721007577656017], 'xyz': "
                  "[19.45670840656562, 0.0, 1.1913797836021993e-15], 'label': 'Tl', 'properties': {}}, "
                  "{'species': [{'element': 'Tl', 'occu': 1}], 'abc': [0.5, 0.5, 0.3278992422343983], 'xyz': "
                  "[9.49238617748072, 1.9927847344276035, 1.9927847344276044], 'label': 'Tl', 'properties': {}}, "
                  "{'species': [{'element': 'Tl', 'occu': 1}], 'abc': [0.5, 0.5, 0.4655743414429113], 'xyz': "
                  "[13.477955646335925, 1.9927847344276035, 1.9927847344276046], 'label': 'Tl', 'properties': {}}, "
                  "{'species': [{'element': 'Tl', 'occu': 1}], 'abc': [0.5, 0.5, 0.6032494406514244], 'xyz': "
                  "[17.463525115191132, 1.9927847344276035, 1.9927847344276048], 'label': 'Tl', 'properties': {}}, "
                  "{'species': [{'element': 'Tl', 'occu': 1}], 'abc': [0.5, 0.5, 0.7409245398599373], 'xyz': "
                  "[21.44909458404634, 1.9927847344276035, 1.992784734427605], 'label': 'Tl', 'properties': {}}]}")
    slab1 = Structure.from_dict(eval(slab_dict1))
    slab2 = Structure.from_dict(eval(slab_dict2))

    WFModel = WFRFModel()
    print(WFModel.predict_work_function(slab1))  # prediction 3.85, ground truth: 3.69
    print(WFModel.predict_work_function(slab2))  # prediction 3.49, ground truth: 3.40
