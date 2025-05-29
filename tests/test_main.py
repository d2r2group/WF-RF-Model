"""Test cases for the WFRFModel class in the wfrfmodel package."""

from pymatgen.core import Structure
from wfrfmodel import WFRFModel

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
bulk_dict = ("{'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0.0, 'lattice': {'matrix': [["
                "3.17031556, 0.0, 1.9412584014205245e-16], [5.09825625865827e-16, 3.17031556, 1.9412584014205245e-16],"
                "[0.0, 0.0, 3.17031556]], 'a': 3.17031556, 'b': 3.17031556, 'c': 3.17031556, 'alpha': 90.0, "
                "'beta': 90.0,"
                "'gamma': 90.0, 'volume': 31.864527039671284}, 'sites': [{'species': [{'element': 'W', 'occu': 1.0}], "
                "'abc': [0.0, 0.0, 0.0], 'xyz': [0.0, 0.0, 0.0], 'label': 'W', 'properties': {}}, {'species': [{"
                "'element': 'W', 'occu': 1.0}], 'abc': [0.5, 0.5, 0.5], 'xyz': [1.5851577800000003, 1.58515778, "
                "1.5851577800000003], 'label': 'W', 'properties': {}}]}")
bulk_dict2 = ("{'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0.0, 'lattice': {'matrix': "
                "[[7.054590496875471, 0.0, -1.710473308443313], [-0.7996513496446345, 7.193239775683547, "
                "-2.3248672680183318], [0.0, 0.0, 7.69028104]], 'a': 7.258992079999999, 'b': 7.601785880000001, "
                "'c': 7.69028104, 'alpha': 107.8082093, 'beta': 103.62906256000001, 'gamma': 91.72863655, "
                "'volume': 390.2460872838975}, 'sites': [{'species': [{'element': 'Cs', 'occu': 1.0}], "
                "'abc': [0.75721209, 0.34052268, 0.55138758], 'xyz': [5.069521793586606, 2.4494612862983605, "
                "2.153464350640027], 'label': 'Cs', 'properties': {}}, {'species': [{'element': 'Cs', "
                "'occu': 1.0}], 'abc': [0.24278791, 0.65947732, 0.44861242], 'xyz': [1.1854173536442305, "
                "4.743778489385187, 1.5014761128983285], 'label': 'Cs', 'properties': {}}, {'species': [{'element': "
                "'Cs', 'occu': 1.0}], 'abc': [0.31212837, 0.21193126, 0.86097737], 'xyz': [2.032466714716343, "
                "1.5244723691427315, 5.594558649243264], 'label': 'Cs', 'properties': {}}, {'species': [{'element': "
                "'Cs', 'occu': 1.0}], 'abc': [0.68787163, 0.78806874, 0.13902263], 'xyz': [4.222472432514493, "
                "5.668767406540816, -1.9396181857049082], 'label': 'Cs', 'properties': {}}, {'species': [{"
                "'element': 'Hg', 'occu': 1.0}], 'abc': [0.81113755, 0.89539163, 0.71289813], "
                "'xyz': [5.006242126498843, 6.440766687730125, 2.0132911511947715], 'label': 'Hg', 'properties': {"
                "}}, {'species': [{'element': 'Hg', 'occu': 1.0}], 'abc': [0.18886245, 0.10460837, 0.28710187], "
                "'xyz': [1.2486970207319934, 0.7524730879534215, 1.6416493123435842], 'label': 'Hg', 'properties': "
                "{}}, {'species': [{'element': 'Hg', 'occu': 1.0}], 'abc': [0.13512203, 0.75269973, 0.94631554], "
                "'xyz': [0.35133323378487036, 5.414349636982267, 5.296382864498449], 'label': 'Hg', 'properties': {"
                "}}, {'species': [{'element': 'Hg', 'occu': 1.0}], 'abc': [0.86487797, 0.24730027, 0.05368446], "
                "'xyz': [5.903605913445966, 1.7788901387012805, -1.6414424009600936], 'label': 'Hg', 'properties': "
                "{}}]}")

def test_predict_work_functions_from_slab():
    """Test the predict_work_functions_from_slab method of WFRFModel."""
    slab1 = Structure.from_dict(eval(slab_dict1)) # prediction 3.85, ground truth: 3.69
    slab2 = Structure.from_dict(eval(slab_dict2)) # prediction 3.49, ground truth: 3.40
    model = WFRFModel()
    # n_params = sum([tree.tree_.node_count for tree in WFModel.model.estimators_]) * 5
    # print(n_params)
    top_WF_slab1, bottom_WF_slab1 = model.predict_work_functions_from_slab(slab1)
    top_WF_slab2, bottom_WF_slab2 = model.predict_work_functions_from_slab(slab2)
    assert abs(top_WF_slab1 - 3.85) < 0.05, "Top WF for slab1 is incorrect"
    assert abs(bottom_WF_slab1 - 3.85) < 0.05, "Bottom WF for slab1 is incorrect"
    assert abs(top_WF_slab2 - 3.5) < 0.05, "Top WF for slab2 is incorrect"
    assert abs(bottom_WF_slab2 - 3.5) < 0.05, "Bottom WF for slab2 is incorrect"

def test_predict_work_functions_from_bulk_and_miller():
    """Test the predict_work_functions_from_bulk_and_miller method of WFRFModel."""
    bulk_W = Structure.from_dict(eval(bulk_dict))
    bulk_CsHg = Structure.from_dict(eval(bulk_dict2))
    model = WFRFModel()
    WF_slabs_of_W = model.predict_work_functions_from_bulk_and_miller(bulk_W, (1, 1, 0))  # returns dict with one slab with two WFs
    WF_slabs_pf_CsHg = model.predict_work_functions_from_bulk_and_miller(bulk_CsHg, (1, 0, 0))  # returns dict with 10 WFs
    assert len(WF_slabs_of_W) == 2, "WF prediction for W bulk should return one slab with two WFs"
    assert len(WF_slabs_pf_CsHg) == 10, "WF prediction for CsHg bulk should return 10 slabs with one WF each"