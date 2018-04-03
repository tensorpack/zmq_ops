#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: zmq_ops.py

import sys
import tensorflow as tf
import struct
import numpy as np
import os

from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework import types_pb2 as DT
# have to import like this: https://github.com/tensorflow/tensorflow/commit/955f038afbeb81302cea43058078e68574000bce

from .common import maybe_compile, get_ext_suffix

__all__ = ['dump_arrays', 'ZMQPullSocket']


def _load_op():
    maybe_compile()
    basename = 'zmq_pull_op' + get_ext_suffix()
    so_file = os.path.join(os.path.dirname(__file__), basename)
    return tf.load_op_library(so_file)


_zmq_ops = _load_op()
from . import libzmqop


class ZMQPullSocket(object):
    def __init__(self, end_point, types, hwm=None, bind=True, name=None):
        """
        Args:
            end_point (str): zmq endpoint
            types ([dtype]): list of tensorflow datatype
            hwm (int): zmq hwm (buffer size)
            bind (bool): to bind or connect
        """
        self._types = types
        assert isinstance(bind, bool), bind

        if name is None:
            self._name = (tf.get_default_graph()
                          .unique_name(self.__class__.__name__))
        else:
            self._name = name

        self._zmq_handle = _zmq_ops.zmq_connection(
            end_point, hwm, bind=bind, shared_name=self._name)

    @property
    def name(self):
        return self._name

    def pull(self):
        return _zmq_ops.zmq_pull(
            self._zmq_handle, self._types)


def dump_arrays(arrs):
    """
    Dump a list of nparray into a format that the ZMQPull op would accept.

    Returns:
        a binary string

    Notes:
        The format is:

        [#tensors(int32)]
        [tensor1][tensor2]...

        Where each tensor is:

        [dtype(int32)][ndims(int32)][shape[0](int32)]...[shape[n](int32)]
        [len(buffer)(int64)][buffer]
    """
    assert isinstance(arrs, (list, tuple))
    for idx, arr in enumerate(arrs):

        if isinstance(arr, float):
            arrs[idx] = np.asarray(arr).astype('float32')
        elif isinstance(arr, int):
            arrs[idx] = np.asarray(arr).astype('int32')

        assert arrs[idx].flags['C_CONTIGUOUS']
        assert isinstance(arrs[idx], np.ndarray), type(arrs[idx])

    return libzmqop.dump_arrays(arrs)


# copied from tensorflow/python/framework/dtypes.py
_DTYPE_DICT = {
    np.float16: DT.DT_HALF,
    np.float32: DT.DT_FLOAT,
    np.float64: DT.DT_DOUBLE,

    np.uint8: DT.DT_UINT8,
    np.uint16: DT.DT_UINT16,
    np.uint32: DT.DT_UINT32,
    np.uint64: DT.DT_UINT64,

    np.int64: DT.DT_INT64,
    np.int32: DT.DT_INT32,
    np.int16: DT.DT_INT16,
    np.int8: DT.DT_INT8,

    np.complex64: DT.DT_COMPLEX64,
    np.complex128: DT.DT_COMPLEX128,

    np.bool: DT.DT_BOOL,
}
_DTYPE_DICT = {np.dtype(k): v for k, v in _DTYPE_DICT.items()}


def dump_arrays_py(arrs):
    """ The old slow python implementation. """
    s = struct.pack('=i', len(arrs))
    for arr in arrs:
        try:
            dtype = _DTYPE_DICT[arr.dtype]
        except KeyError:
            raise KeyError("Dtype {} is unsupported by current ZMQ Op!".format(arr.dtype))


        s += struct.pack('=i', int(dtype))
        dims = arr.shape
        s += struct.pack('=i', len(dims))
        for k in dims:
            s += struct.pack('=i', k)

        tensor_content = arr.tobytes()
        s += struct.pack('=q', len(tensor_content))
        s += tensor_content
    return s
