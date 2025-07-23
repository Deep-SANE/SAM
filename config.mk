# compiler specification:
ifeq ($(USE_SLEPC),1)
CXX=$(MPICXX)
endif

# SLEPc related options
ifeq ($(USE_SLEPC),1)
CXXFLAGS += -DU3LIB_USE_SLEPC
CXXFLAGS += -DU3LIB_SPARSE_MAT
CXXFLAGS += -DU4LIB_USE_SLEPC
CXXFLAGS += -DU4LIB_SPARSE_MAT
CXXFLAGS += -DSAM_USE_SLEPC
CXXFLAGS += -DSAM_SPARSE_MAT
endif

# header file inclusion paths:
ifdef SAM_DIR
CXXFLAGS += -I$(SAM_DIR)/include
endif

ifdef U3LIB_DIR
CXXFLAGS += -I$(U3LIB_DIR)/include
endif

ifdef U4LIB_DIR
CXXFLAGS += -I$(U4LIB_DIR)/include
endif

ifdef QCX_UTILS_DIR
CXXFLAGS += -I$(QCX_UTILS_DIR)/include
endif

ifdef UNTOU3_DIR
CXXFLAGS += -I$(UNTOU3_DIR)/include
endif

ifdef BOOST_ROOT
CXXFLAGS += -I$(BOOST_ROOT)/include
endif

ifdef HAVE_FMT
CXXFLAGS += -DHAVE_FMT
ifdef FMT_ROOT
CXXFLAGS += -I$(FMT_ROOT)/include
endif
endif

ifdef EIGEN_DIR
CXXFLAGS += -I$(EIGEN_DIR)
endif 

ifdef WIGXJPF_DIR
CXXFLAGS += -I$(WIGXJPF_DIR)/inc
endif

ifeq ($(USE_SLEPC),1)
ifdef PETSC_DIR
CXXFLAGS += -I$(PETSC_DIR)/include
ifdef PETSC_ARCH
CXXFLAGS += -I$(PETSC_DIR)/$(PETSC_ARCH)/include
endif
endif
ifdef SLEPC_DIR
CXXFLAGS += -I$(SLEPC_DIR)/include
ifdef SLEPC_ARCH
CXXFLAGS += -I$(SLEPC_DIR)/$(SLEPC_ARCH)/include
endif
endif
endif

# linking options:
ifdef SAM_DIR
LDFLAGS += -L$(SAM_DIR)/lib
endif
LDFLAGS += -lsam

ifdef WIGXJPF_DIR
LDFLAGS += -L$(WIGXJPF_DIR)/lib 
endif
LDFLAGS += -lwigxjpf

ifeq ($(USE_SLEPC),1)
ifdef PETSC_DIR
LDFLAGS += -L$(PETSC_DIR)/lib
ifdef PETSC_ARCH
LDFLAGS += -L$(PETSC_DIR)/$(PETSC_ARCH)/lib
endif
endif
ifdef SLEPC_DIR
LDFLAGS += -L$(SLEPC_DIR)/lib
ifdef SLEPC_ARCH
LDFLAGS += -L$(SLEPC_DIR)/$(SLEPC_ARCH)/lib
endif
endif
LDFLAGS += -lpetsc -lslepc 
endif
