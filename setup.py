from setuptools import setup, find_packages

setup(name='ipmi',
      version='0.1.0',
      author='17113498',
      packages=find_packages(exclude=['*tests']),
      install_requires=[
          'nibabel',
          'nipype',
          'Pillow',
          'matplotlib',
          'seaborn',
          'pandas',
          ],
     )
