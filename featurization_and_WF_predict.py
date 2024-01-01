
import os
import pandas as pd
import joblib
import json
from pymatgen.core import Structure

from sklearn.preprocessing import StandardScaler

import sys

sys.path.append(os.getcwd())
from utils_featurization import featurization, raw_to_final_features

file = 'slab2.json'

with open(file, 'r') as f:
    slab_dict = f.readline()

slab = Structure.from_dict(eval(slab_dict))
# print(slab)
# slab WF_top: 3.689
# slab2 WF_top: 3.399

feat_df = pd.DataFrame(columns=['f_angles_min', 'f_angles_max', 'f_chi', 'f_chi2', 'f_chi3', 'f_1_r', 'f_1_r2',
                                'f_1_r3', 'f_fie', 'f_fie2', 'f_fie3',
                                'f_mend', 'f_mend2', 'f_mend3', 'f_z1_2', 'f_z1_3', 'f_packing_area',
                                'f_packing_area2', 'f_packing_area3'])


features = featurization(slab, tol=0.4)
feat_df.loc[0, 'f_angles_min':'f_packing_area3'] = features

final = raw_to_final_features(feat_df)

model = joblib.load('RF_1691469908.2138267.joblib')
# Top 15 features from RFE optimized features for RF
features = ['f_angles_min', 'f_chi', 'f_1_r', 'f_fie', 'f_fie2', 'f_fie3', 'f_mend', 'f_z1_2', 'f_z1_3',
            'f_packing_area', 'f_chi_min', 'f_chi2_max', 'f_1_r_min', 'f_fie2_min', 'f_mend2_min']

# Load feature scaling from training
with open('scaler.json', 'r') as f:
    scaler_json = json.load(f)
scaler_load = json.loads(scaler_json)
sc = StandardScaler()
sc.scale_ = scaler_load['scale']
sc.mean_ = scaler_load['mean']
sc.var_ = scaler_load['var']
sc.n_samples_seen_ = scaler_load['n_samples_seen']
sc.n_features_in_ = scaler_load['n_features_in']


X = sc.transform([final.loc[0, features].tolist()])
WF_prediction = model.predict(X)
print(WF_prediction)
