from setuptools import setup, find_packages

setup(
    name='WF-RF-Model',
    version='2.0.1',
    url='https://github.com/d2r2group/WF-RF-Model',
    description='Random Forest Model to predict the work function of surfaces',
    author='Peter Schindler',
    author_email='p.schindler@northeastern.edu',
    python_requires='>=3.8',
    install_requires=['numpy', 'pandas', 'scikit-learn==1.1.1', 'pymatgen==2022.5.26', 'joblib==1.0.1'],
    packages=find_packages(),
    package_data={'WF-RF-Model': ['*.json', '*.txt']},
    include_package_data=True
)
