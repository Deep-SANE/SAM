# SAM/config/config.mk
# Central build definitions for SAM

CONFIG_DIR ?= $(SAM_ROOT)/config
GEN_MK := $(CONFIG_DIR)/config.generated.mk

ifeq ($(wildcard $(GEN_MK)),)
  $(error SAM is not configured. Run 'make config' first)
endif

include $(GEN_MK)

# Ensure the MPI compiler wrapper (mpicxx) invokes the inherited GCC, not the
# system default (e.g. Apple clang), by exporting these into the build env.
export OMPI_CXX MPICH_CXX

# Include paths
INCLUDES := -I$(SAM_ROOT)/include \
            -I$(SAM_ROOT)/utils \
            -I$(UNTOU3_DIR)/include \
            -I$(U4LIB_DIR)/include \
            -I$(U3LIB_DIR)/include \
            -I$(U3LIB_DIR)/utils

ifneq ($(strip $(BOOST_ROOT)),)
  INCLUDES += -I$(BOOST_ROOT)/include
endif

ifneq ($(strip $(EIGEN_DIR)),)
  INCLUDES += -I$(EIGEN_DIR)
endif

ifneq ($(strip $(FMT_DIR)),)
  INCLUDES += -I$(FMT_DIR)/include
  CXXFLAGS += -DHAVE_FMT=1 -DFMT_HEADER_ONLY=1
endif

ifneq ($(strip $(SPECTRA_DIR)),)
  INCLUDES += -I$(SPECTRA_DIR)
endif

ifeq ($(USE_MPI),1)
  CXXFLAGS += -DUSE_MPI
endif

ifeq ($(USE_OPENMP),1)
  CXXFLAGS += $(OMP_CXXFLAGS)
  LDFLAGS += $(OMP_LDFLAGS)
endif

ifeq ($(USE_WIGNERNJ),1)
  CXXFLAGS += -DUSE_WIGNERNJ
  ifneq ($(strip $(WIGNERNJ_DIR)),)
    INCLUDES += -I$(WIGNERNJ_DIR)/include
    LDFLAGS += -L$(WIGNERNJ_DIR)/lib -Wl,-rpath,$(WIGNERNJ_DIR)/lib
  endif
  LDFLAGS += -lwignernj
endif

ifeq ($(NULLSPACE_SOLVER),krans)
  CXXFLAGS += -DUSE_KRANS -DNULLSPACE_SOLVER_KRANS -DNULLSPACE_SPARSE
  ifneq ($(strip $(KRANS_DIR)),)
    INCLUDES += -I$(KRANS_DIR)/include
    LDFLAGS += -L$(KRANS_DIR)/build -lkrans_eigen -lkrans_core
  endif
endif

# zlib for gzipped two-body matrix elements reader
LDFLAGS += -lz

# SAM data path support
CXXFLAGS += -DSAM_DATA_DIR
ifneq ($(strip $(SAM_DATA_DIR)),)
  CXXFLAGS += -DSAM_DEFAULT_DATA_DIR=\"$(SAM_DATA_DIR)\"
endif

SAM_LIB := $(SAM_ROOT)/lib/libsam.a

CXXFLAGS += $(INCLUDES)
