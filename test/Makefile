# SAM regression tests. Run from the SAM root with `make test` (after
# `make config` and `make`), or here with `make`.
#
# Each test_*.cpp is built against lib/libsam.a into bin/test/ and run by
# run_tests.sh as one MPI process, with SAM_DATA_DIR set to a fresh temporary
# directory (removed afterwards), so the tests never read or write your data.
# A summary table follows the output. Override the launcher if needed, e.g. on
# a cluster:
#   make test MPIRUN="srun -n 1"

SAM_ROOT ?= $(abspath $(CURDIR)/..)
CONFIG_DIR ?= $(SAM_ROOT)/config

include $(CONFIG_DIR)/config.mk
include $(CONFIG_DIR)/make.d/pretty.mk

BIN := $(SAM_ROOT)/bin/test
SOURCES := $(sort $(wildcard test_*.cpp))
PROGRAMS := $(patsubst %.cpp,$(BIN)/%,$(SOURCES))

# the mpirun next to the configured mpicxx, else the one on the path
MPIRUN ?= $(shell p="$(dir $(CXX))mpirun"; [ -x "$$p" ] && echo "$$p" || echo mpirun) -np 1

# keep assert() active in tests, also in release builds
TEST_FLAGS = $(CXXFLAGS) -UNDEBUG

.PHONY: all test clean
all: test

test: $(PROGRAMS)
	@bash run_tests.sh "$(MPIRUN)" $(PROGRAMS)

$(BIN)/test_%: test_%.cpp $(wildcard *.h) $(SAM_LIB) | $(BIN)
	$(call pretty_print,CXX,$@)
	$(Q)$(CXX) $(TEST_FLAGS) -o $@ $< $(SAM_LIB) $(LDFLAGS)

$(BIN):
	$(Q)mkdir -p $@

clean:
	$(Q)rm -rf $(BIN)
