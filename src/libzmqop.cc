//File: libzmqop.cc

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include <tensorflow/core/framework/types.pb.h>

namespace py = pybind11;

namespace {

// return the number of serialized bytes
size_t arr_size(const py::array& arr) {
  return 4 + 4 + arr.ndim() * 4 + 8 + arr.nbytes();
}

inline void write_int32(int v, char** p) {
  *reinterpret_cast<int*>(*p) = v;
  *p += 4;
}

inline void write_int64(int64_t v, char** p) {
  *reinterpret_cast<int64_t*>(*p) = v;
  *p += 8;
}

int get_tf_dtype(const py::array& arr) {
  typedef tensorflow::DataType T;
  switch (arr.dtype().kind()) {
    case 'f':
      switch (arr.itemsize()) {
        case 2: return T::DT_HALF;
        case 4: return T::DT_FLOAT;
        case 8: return T::DT_DOUBLE;
      }
      break;
    case 'u':
      switch (arr.itemsize()) {
        case 1: return T::DT_UINT8;
        case 2: return T::DT_UINT16;
        case 4: return T::DT_UINT32;
        case 8: return T::DT_UINT64;
      }
      break;
    case 'i':
      switch (arr.itemsize()) {
        case 1: return T::DT_INT8;
        case 2: return T::DT_INT16;
        case 4: return T::DT_INT32;
        case 8: return T::DT_INT64;
      }
      break;
    case 'c':
      switch (arr.itemsize()) {
        case 8: return T::DT_COMPLEX64;
        case 16: return T::DT_COMPLEX128;
      }
      break;
    case 'b':
      return T::DT_BOOL;
  }
  fprintf(stderr, "Unsupported array type!\n");
  fflush(stderr);
  abort();
}

}

py::bytes dump_arrays(const std::vector<py::array>& arrs) {
  size_t total_size = 4;
  for (auto& arr : arrs) total_size += arr_size(arr);

  std::string ret;
  ret.resize(total_size);  // pre-allocate space
  char* ptr = &ret[0];
  write_int32((int)arrs.size(), &ptr);

  for (auto& arr: arrs) {
    // printf("dtype=%c, nbytes=%d\n", arr.dtype().kind(), arr.nbytes());
    // printf("tfdtype=%d\n", get_tf_dtype(arr));
    write_int32(get_tf_dtype(arr), &ptr);
    write_int32(arr.ndim(), &ptr);
    for (int i = 0; i < arr.ndim(); ++i) write_int32(arr.shape(i), &ptr);

    size_t nb = arr.nbytes();
    write_int64(nb, &ptr);
    memcpy(ptr, arr.data(), nb);
    ptr += nb;
  }
  return py::bytes{ret};  // TODO extra copy here: https://github.com/pybind/pybind11/issues/1236
}

PYBIND11_MODULE(libzmqop, m) {
  m.def("dump_arrays", &dump_arrays);
}
