# $File: Makefile
# $Date: Thu Feb 01 21:06:48 2018 -0800

OBJ_DIR = obj
SO_DIR = $(realpath ../zmq_ops)
PYTHON ?= python
PYTHON_CONFIG ?= python-config

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CXX ?= g++
endif
ifeq ($(UNAME_S),Darwin)
	CXX ?= clang++
endif

OPTFLAGS ?= -O3 -march=native
#OPTFLAGS ?= -g3 -fsanitize=address,undefined -O2 -lasan
#OPTFLAGS ?= -g3 -fsanitize=leak -O2 -lubsan

# libraries: TF preceeds others, so g++ looks for protobuf among TF headers
ifneq ($(MAKECMDGOALS), clean)
TF_CXXFLAGS ?= $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS ?= $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
PYTHON_CXXFLAGS ?= $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS ?= $(shell $(PYTHON_CONFIG) --ldflags)
endif
CXXFLAGS += -I. $(TF_CXXFLAGS) $(PYTHON_CXXFLAGS)

# extra packages from pkg-config
LIBS = libzmq
CXXFLAGS += $(shell pkg-config --cflags $(LIBS))
LDFLAGS += $(shell pkg-config $(LIBS) --libs)

CXXFLAGS += -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
CXXFLAGS += $(DEFINES) -std=c++11 $(OPTFLAGS) -fPIC

LDFLAGS += $(OPTFLAGS)
LDFLAGS += -shared -fPIC
ifeq ($(UNAME_S),Darwin)
	LDFLAGS += -Wl,-undefined -Wl,dynamic_lookup
endif

SHELL = bash
# sources to include
SRCDIR =
ccSOURCES = $(shell find $(SRCDIR) -name "*.cc" | sed 's/^\.\///g')
OBJS = $(addprefix $(OBJ_DIR)/,$(ccSOURCES:.cc=.o))
DEPFILES = $(OBJS:.o=.d)

EXT_SUFFIX ?= $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
TF_SO = $(SO_DIR)/zmq_pull_op$(EXT_SUFFIX)
PYBIND_SO = $(SO_DIR)/libzmqop$(EXT_SUFFIX)
ALL_SO = $(TF_SO) $(PYBIND_SO)

.PHONY: all clean

all: $(ALL_SO)

ifneq ($(MAKECMDGOALS), clean)
sinclude $(DEPFILES)
endif

$(TF_SO): $(SO_DIR)/%$(EXT_SUFFIX): $(OBJ_DIR)/%.o
	@echo "Linking $@ ..."
	@$(CXX) $^ -o $@ $(LDFLAGS) $(TF_LDFLAGS)
	@echo "done."

$(PYBIND_SO): $(SO_DIR)/%$(EXT_SUFFIX): $(OBJ_DIR)/%.o
	@echo "Linking $@ ..."
	@$(CXX) $^ -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS)
	@echo "done."

$(OBJ_DIR)/%.o: %.cc
	@echo "[cc] $< ..."
	@$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJ_DIR)/%.d: %.cc Makefile
	@mkdir -pv $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(CXXFLAGS) -MM -MT "$(OBJ_DIR)/$(<:.cc=.o) $(OBJ_DIR)/$(<:.cc=.d)" "$<"  > "$@" || rm "$@"

clean:
	@rm -rvf $(OBJ_DIR) $(ALL_SO)

