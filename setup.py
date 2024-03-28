from setuptools import setup, find_packages

setup(
    name='seamlessvalidation',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # package dependencies below
        'scikit-learn',
        'pandas',
        'numpy',
        'matplotlib',
        #'tensorflow', 'keras', 'torch', 'spacy', 'nltk'
    ],
    author='ZhuZheng(Iverson) ZHOU',
    author_email='zzho044@aucklanduni.ac.nz',
    description='A package for easy validation and post deployment monitoring of common linear and non linear ML models and clustering model',
    keywords='machine learning validation monitoring',
)
