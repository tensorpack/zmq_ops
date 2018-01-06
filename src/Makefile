# $File: Makefile
# $Date: Sat Jan 06 06:50:56 2018 -0800

OBJ_DIR = obj
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
CXXFLAGS += $(TF_CXXFLAGS) $(PYTHON_CXXFLAGS)
#LDFLAGS += $(TF_LDFLAGS) $(PYTHON_LDFLAGS)

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
TF_SO = zmq_pull_op$(EXT_SUFFIX)
PYBIND_SO = libzmqop$(EXT_SUFFIX)

.PHONY: all clean

all: $(TF_SO) $(PYBIND_SO)

ifneq ($(MAKECMDGOALS), clean)
sinclude $(DEPFILES)
endif

$(TF_SO): %$(EXT_SUFFIX): $(OBJ_DIR)/%.o
	@echo "Linking $@ ..."
	@$(CXX) $^ -o $@ $(LDFLAGS) $(TF_LDFLAGS)
	@echo "done."

$(PYBIND_SO): %$(EXT_SUFFIX): $(OBJ_DIR)/%.o
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
	@rm -rvf $(OBJ_DIR) $(SO)

