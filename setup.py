from setuptools import setup
from sigopt_sklearn.version import VERSION

# Keep this in sync with `requirements.txt` and the conda install process in `.travis.yml`!
install_requires = [
  'joblib>=0.9.4',
  'numpy>=1.9',
  'scikit-learn>=0.17.1',
  'sigopt>=2.6.0',
]

setup(
  name='sigopt_sklearn',
  version=VERSION,
  description='SigOpt + scikit-learn Integrations',
  author='SigOpt',
  author_email='support@sigopt.com',
  url='https://sigopt.com/',
  packages=['sigopt_sklearn'],
  install_requires=install_requires,
  extras_require={
    'ensemble': ['xgboost>=0.4a30'],
  },
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
  ]
)
