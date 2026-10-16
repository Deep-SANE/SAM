SAM_ROOT ?= $(abspath $(CURDIR)/..)
CONFIG_DIR ?= $(SAM_ROOT)/config

include $(CONFIG_DIR)/config.mk
include $(CONFIG_DIR)/make.d/pretty.mk

OBJ_DIR = obj

SRCS = $(wildcard *.cpp)
OBJS = $(addprefix $(OBJ_DIR)/,$(SRCS:.cpp=.o))
DEPS = $(OBJS:.o=.d)

LIB_DIR = $(SAM_ROOT)/lib
LIB = $(LIB_DIR)/libsam.a

.PHONY: all
all: $(LIB)

$(LIB): $(OBJS) | $(LIB_DIR)
	$(call pretty_print,AR,$@)
	$(Q)$(AR) rcs $@ $^

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(call pretty_print,CXX,$@)
	$(Q)$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(LIB_DIR) $(OBJ_DIR):
	$(Q)mkdir -p $@

-include $(DEPS)

.PHONY: clean
clean:
	$(Q)rm -rf $(OBJ_DIR) $(LIB)
