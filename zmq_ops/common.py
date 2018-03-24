#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py

import sysconfig
import tensorflow as tf
import os


# https://github.com/uber/horovod/blob/10835d25eccf4b198a23a0795edddf0896f6563d/horovod/tensorflow/mpi_ops.py#L30-L40
def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def get_src_dir():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'src')


def compile():
    cxxflags = ' '.join(tf.sysconfig.get_compile_flags())
    ldflags = ' '.join(tf.sysconfig.get_link_flags())
    ext_suffix = get_ext_suffix()
    py_include = '-isystem ' + sysconfig.get_path('include')
    py_ldflags = sysconfig.get_config_var('LDFLAGS') + ' -lpython' \
        + (sysconfig.get_config_var('LDVERSION') or sysconfig.get_config_var('VERSION'))
    compile_cmd = 'TF_CXXFLAGS="{}" TF_LDFLAGS="{}" EXT_SUFFIX="{}" ' \
        'PYTHON_CXXFLAGS="{}" PYTHON_LDFLAGS="{}" make -C "{}"'.format(
        cxxflags, ldflags, ext_suffix,
        py_include, py_ldflags, get_src_dir())
    print("Compile ops by command " + compile_cmd + ' ...')
    ret = os.system(compile_cmd)
    return ret


def maybe_compile():
    dirname = os.path.dirname(__file__)
    for sobase in ['zmq_pull_op', 'libzmqop']:
        soname = sobase + get_ext_suffix()
        sofile = os.path.join(dirname, soname)
        if not os.path.isfile(sofile):
            ret = compile()
            if ret != 0:
                raise RuntimeError("ops compilation failed!")
            return


if __name__ == '__main__':
    compile()
