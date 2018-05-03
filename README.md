
## TensorFlow ZMQ Op

A convenient way to receive data from other processes. This small library can:

+ Send a list of numpy arrays from python; serialization is written in C++ for efficiency.
  + One copy in merging all the buffers; One copy in pybind11 overhead (TODO); One copy in ZMQ send.
+ Receive a list of tensors from tensorflow;
  + One copy in ZMQ recv; One copy to split the buffer into tensors.
  + The op is stateful and safe to be evaluated multiple times in one `sess.run` call.
+ Serialization is in a custom protocol for efficiency;

## Why:

Sometimes for complicated large-scale tasks you would really want data processing to be separate from TensorFlow.
However in TensorFlow there is no good way to receive data from other processes.

## Build:

Require gcc>=5.3, tensorflow>=1.4, zeromq>=4.

Require the `zmq.hpp` header from [cppzmq](https://github.com/zeromq/cppzmq) at
your compiler's include path, or under the `src` directory.

Add `/path/to/git/clone/zmq_ops` to `PYTHONPATH` to be able to import it.
Or use `pip install .` to install it.

Ops will be compiled the first time it gets imported.
Note that it usually requires recompilation after a TensorFlow reinstallation.

## Use:

See `benchmark.py` for usage.

On my machine this script can achieve about 1.3GB/s throughput. Equivalent to about 2.3k float32 (or 9.2k uint8) imagenet images per second.
