from setuptools import setup, find_packages
from os.path import join, dirname
from pip.req import parse_requirements

with open(join(dirname(__file__), 'dinosaur/version.txt'), 'rb') as f:
    version = f.read().decode('ascii').strip()

requirements = [str(ir.req) for ir in parse_requirements("requirements.txt", session=False)]

setup(
    name='dinosaur',
    version=version,
    description='A research project on China mutual fund industry allocation cycling',
    author='Zhao, Yunlu',
    author_email='t_One_8@yeah.net',
    requirements=requirements,
    packages=find_packages(),
    # entry_points={},
    classifiers=[
            'Programming Language :: Python',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: Unix',
            # 'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ]
     )

