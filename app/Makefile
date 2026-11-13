# SAM programs:
#   tutorial/*.cpp  example programs shipped with SAM -> bin/tutorial/
#   work/*.cpp      your own programs (not tracked by git) -> bin/work/
# Copy a tutorial into work/ and edit the copy; tutorial/ stays as shipped.
# Only the .cpp files directly in work/ are built, so subfolders can hold
# notes or old versions.

SAM_ROOT ?= $(abspath $(CURDIR)/..)
CONFIG_DIR ?= $(SAM_ROOT)/config

include $(CONFIG_DIR)/config.mk
include $(CONFIG_DIR)/make.d/pretty.mk

BIN_DIR = $(SAM_ROOT)/bin
OBJ_DIR = obj
SAM_LIB = $(SAM_ROOT)/lib/libsam.a

TUTORIAL_SRCS = $(wildcard tutorial/*.cpp)
WORK_SRCS = $(wildcard work/*.cpp)

# tutorial/X.cpp -> bin/tutorial/X, work/X.cpp -> bin/work/X
PRGS = $(patsubst %.cpp,$(BIN_DIR)/%,$(TUTORIAL_SRCS) $(WORK_SRCS))
OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(TUTORIAL_SRCS) $(WORK_SRCS))
DEPS = $(OBJS:.o=.d)

.PHONY: all
all: $(PRGS)

$(BIN_DIR)/%: $(OBJ_DIR)/%.o $(SAM_LIB)
	$(Q)mkdir -p $(dir $@)
	$(call pretty_print,LINK,$@)
	$(Q)$(CXX) $(CXXFLAGS) -o $@ $< $(SAM_LIB) $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cpp
	$(Q)mkdir -p $(dir $@)
	$(call pretty_print,CXX,$@)
	$(Q)$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

.PHONY: clean
clean:
	$(Q)rm -rf $(OBJ_DIR) $(BIN_DIR)/tutorial $(BIN_DIR)/work
