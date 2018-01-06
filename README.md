
## TensorFlow ZMQ Op

+ Send a list of numpy arrays from python; serialization is written in C++ for efficiency.
+ Recv a list of tensors from tensorflow;
+ Serialization is in a custom protocol for efficiency;

## Build:

Install libzmq, as well as the `zmq.hpp` header from cppzmq.

Then, `make`

## Use:

See `benchmark.py` for usage.
