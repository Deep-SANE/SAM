SAM_ROOT ?= $(abspath $(CURDIR)/..)
CONFIG_DIR ?= $(SAM_ROOT)/config

include $(CONFIG_DIR)/config.mk
include $(CONFIG_DIR)/make.d/pretty.mk

BIN_DIR = $(SAM_ROOT)/bin
OBJ_DIR = obj

SRCS = $(wildcard *.cpp)
OBJS = $(addprefix $(OBJ_DIR)/,$(SRCS:.cpp=.o))
DEPS = $(OBJS:.o=.d)
PRGS = $(addprefix $(BIN_DIR)/,$(basename $(SRCS)))
SAM_LIB = $(SAM_ROOT)/lib/libsam.a

.PHONY: all
all: $(PRGS)

$(BIN_DIR)/%: $(OBJ_DIR)/%.o $(SAM_LIB) | $(BIN_DIR)
	$(call pretty_print,LINK,$@)
	$(Q)$(CXX) $(CXXFLAGS) -o $@ $< $(SAM_LIB) $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(call pretty_print,CXX,$@)
	$(Q)$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	$(Q)mkdir -p $@

-include $(DEPS)

.PHONY: clean
clean:
	$(Q)rm -rf $(OBJ_DIR) $(PRGS)
