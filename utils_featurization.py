# Featurizes slabs via function
# Featurizes mixed layers depending on min, max, mean tag

import numpy as np
import math
import pandas as pd
import statistics
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure


with open('atomic_features/firstionizationenergy.txt') as f:
    content = f.readlines()
fie = [float(x.strip()) for x in content]

with open('atomic_features/mendeleev.txt') as f:
    content = f.readlines()
mendeleev = [float(x.strip()) for x in content]


def featurization(struc: Structure, tol: float = 0.4):
    # Tolerance tol in Angstrom
    # print(struct, flush=True)

    error = None
    # print(struc, flush=True)
    # Alternative solution: [list(s.species.get_el_amt_dict().keys())[0] for s in struc.sites]
    for el in [s.species.elements[0].symbol for s in struc.sites]:
        if el in ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'At', 'Rn', 'Fr', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']:
            error = '2.Structure contains element not supported for featurization.'

    if error is None:
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
                    error = '5.Error. No highest index found. Counter = ' + str(counter)
                    break

            # Check there are at least 3 layers, given tolerance to group layers
            # print('Indices list = {}'.format(indices_list))
            if len(indices_list) < 3 and not error:
                error = '4.Slab less than 3 atomic layers in z-direction, with a tolerance = ' + str(tol) + ' A.'

            pos = struc.frac_coords

            # Check if structure is of form slab with minimum vacuum of 5 A in z-direction
            min_vac = 5.0  # Angstrom
            # print('max - min = {}'.format((max(p[2] for p in pos) - min(p[2] for p in pos)) * struc.lattice.c))
            # print('c length = {}'.format(struc.lattice.c))
            if max(pos[:][2]) - min(pos[:][2]) * struc.lattice.c + min_vac > struc.lattice.c:
                error = '6.Input structure either has no vacuum between slabs or is not oriented in z-direction'
        else:
            error = '3.Slab less than 4 atomic layers in z-direction before applying tolerance.'

    if error is None:
        # Add features
        # ------------
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
        f_angles_max = max(f_angles)

        f_packing_area = len(indices_list[sindex]) / (cell[0] * cell[1] * math.sin(cell[5]))
        # f_area = cell[0] * cell[1] * math.sin(cell[5])

        # Features layer 2
        f_z1_2 = abs(pos[indices_list[sindex][0]][2] - pos[indices_list[sindex2][0]][2]) * cell[2]
        f_chi2 = []
        f_1_r2 = []
        f_fie2 = []
        f_mend2 = []

        for ind2 in range(len(indices_list[sindex2])):
            f_chi2.append(Element(chem[indices_list[sindex2][ind2]]).X)
            if Element(chem[indices_list[sindex2][ind2]]).atomic_radius_calculated:
                f_1_r2.append(1 / Element(chem[indices_list[sindex2][ind2]]).atomic_radius_calculated)
            else:
                f_1_r2.append(1 / Element(chem[indices_list[sindex2][ind2]]).atomic_radius)
            f_fie2.append(fie[Element(chem[indices_list[sindex2][ind2]]).Z])
            f_mend2.append(mendeleev[Element(chem[indices_list[sindex2][ind2]]).Z])
        f_packing_area2 = len(indices_list[sindex2]) / (cell[0] * cell[1] * math.sin(cell[5]))

        # Features layer 3
        f_z1_3 = abs(pos[indices_list[sindex][0]][2] - pos[indices_list[sindex3][0]][2]) * cell[2]
        f_chi3 = []
        f_1_r3 = []
        f_fie3 = []
        f_mend3 = []

        for ind3 in range(len(indices_list[sindex3])):
            f_chi3.append(Element(chem[indices_list[sindex3][ind3]]).X)
            if Element(chem[indices_list[sindex3][ind3]]).atomic_radius_calculated:
                f_1_r3.append(1 / Element(chem[indices_list[sindex3][ind3]]).atomic_radius_calculated)
            else:
                f_1_r3.append(1 / Element(chem[indices_list[sindex3][ind3]]).atomic_radius)
            f_fie3.append(fie[Element(chem[indices_list[sindex3][ind3]]).Z])
            f_mend3.append(mendeleev[Element(chem[indices_list[sindex3][ind3]]).Z])
        f_packing_area3 = len(indices_list[sindex3]) / (cell[0] * cell[1] * math.sin(cell[5]))

        return np.array([f_angles_min, f_angles_max, f_chi, f_chi2, f_chi3, f_1_r, f_1_r2, f_1_r3,
                         f_fie, f_fie2, f_fie3, f_mend, f_mend2, f_mend3, f_z1_2, f_z1_3,
                         f_packing_area, f_packing_area2, f_packing_area3], dtype=object)
    else:
        return np.array([None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                         None, None, None, None])


def raw_to_final_features(raw: pd.DataFrame, labels=None):
    if labels is None:
        labels = ['f_chi', 'f_chi2', 'f_chi3', 'f_1_r', 'f_1_r2', 'f_1_r3', 'f_fie', 'f_fie2', 'f_fie3',
                  'f_mend', 'f_mend2', 'f_mend3']
    # deleteindex = []
    for label in labels:
        if any([l in label for l in ['chi', '1_r', 'fie', 'mend']]):
            raw[label] = raw[label].astype(object)
            raw[label + '_max'] = raw[label].apply(lambda lst: max(lst))
            raw[label + '_min'] = raw[label].apply(lambda lst: min(lst))
            raw[label] = raw[label].apply(lambda lst: statistics.mean(lst))

    return raw
