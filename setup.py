from setuptools import setup
from sigopt_sklearn.version import VERSION

# keep this in sync with requirements.txt
install_requires=[
   'sigopt',
   'numpy',
   'sklearn',
   'joblib']

setup(
  name='sigopt_sklearn',
  version=VERSION,
  description='SigOpt + scikit-learn Integrations',
  author='SigOpt',
  author_email='support@sigopt.com',
  url='https://sigopt.com/',
  packages=['sigopt_sklearn'],
  install_requires=install_requires,
  classifiers=[
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
  ]
)
