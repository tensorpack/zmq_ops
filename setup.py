import setuptools
from setuptools import setup
import glob

from zmq_ops.common import get_ext_suffix

so_files = glob.glob('zmq_ops/*' + get_ext_suffix())
assert len(so_files), "Need to compile the libraries first!"

setup(
    name='zmq_ops',
    packages=['zmq_ops'],
    version='0.1.0',
    description='TensorFlow ZMQ Ops',
    author='Yuxin Wu',

    zip_safe=False,
    include_package_data=True,
    package_data={'zmq_ops': ['*' + get_ext_suffix()]},
    install_requires=[ ]
)
