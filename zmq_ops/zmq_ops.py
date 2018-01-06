#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: zmq_ops.py

import tensorflow as tf
import struct
import numpy as np
import os

from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework import types_pb2 as DT
# have to import like this: https://github.com/tensorflow/tensorflow/commit/955f038afbeb81302cea43058078e68574000bce

from .common import compile, get_ext_suffix, get_src_dir

__all__ = ['dumps_zmq_op', 'ZMQPullSocket']


_zmq_mod = None


def _load_op():
    basename = 'zmq_pull_op' + get_ext_suffix()
    so_file = os.path.join(get_src_dir(), basename)
    if not os.path.isfile(so_file):
        ret = compile()
        if ret != 0:
            raise RuntimeError("ops compilation failed!")

    global _zmq_mod
    _zmq_mod = tf.load_op_library(so_file)


_load_op()


class ZMQPullSocket(object):
    def __init__(self, end_point, types, hwm=None, bind=True, name=None):
        self._types = types
        assert isinstance(bind, bool), bind

        if name is None:
            self._name = (tf.get_default_graph()
                          .unique_name(self.__class__.__name__))
        else:
            self._name = name

        self._zmq_handle = _zmq_mod.zmq_connection(
            end_point, hwm, bind=bind, shared_name=self._name)

    @property
    def name(self):
        return self._name

    def pull(self):
        return _zmq_mod.zmq_pull(
            self._zmq_handle, self._types)


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


def to_tensor_proto(arr):
    """
    Convert a numpy array to TensorProto

    Args:
        arr: numpy.ndarray. only supports common numerical types
    """
    if isinstance(arr, float):
        arr = np.asarray(arr).astype('float32')
    elif isinstance(arr, int):
        arr = np.asarray(arr).astype('int32')
    assert isinstance(arr, np.ndarray), type(arr)
    try:
        dtype = _DTYPE_DICT[arr.dtype]
    except KeyError:
        raise KeyError("Dtype {} is unsupported by current ZMQ Op!".format(arr.dtype))

    ret = TensorProto()
    shape = ret.tensor_shape
    for s in arr.shape:
        d = shape.dim.add()
        d.size = s

    ret.dtype = dtype

    buf = arr.tobytes()
    ret.tensor_content = buf
    return ret


def dump_tensor_protos(protos):
    """
    Serialize a list of :class:`TensorProto`, for communication between custom TensorFlow ops.

    Args:
        protos (list): list of :class:`TensorProto` instance

    Notes:
        The format is:

        [#tensors(int32)]
        [tensor1][tensor2]...

        Where each tensor is:

        [dtype(int32)][ndims(int32)][shape[0](int32)]...[shape[n](int32)]
        [len(buffer)(int64)][buffer]
    """

    s = struct.pack('=i', len(protos))
    for p in protos:
        tensor_content = p.tensor_content

        s += struct.pack('=i', int(p.dtype))
        dims = p.tensor_shape.dim
        s += struct.pack('=i', len(dims))
        for k in dims:
            s += struct.pack('=i', k.size)
        s += struct.pack('=q', len(tensor_content))
        s += tensor_content
    return s


def dumps_zmq_op(dp):
    """
    Dump a datapoint (list of nparray) into a format that the ZMQPull op would accept.

    Args:
        dp: list of nparray

    Returns:
        a binary string
    """
    assert isinstance(dp, (list, tuple))
    protos = [to_tensor_proto(arr) for arr in dp]
    return dump_tensor_protos(protos)
