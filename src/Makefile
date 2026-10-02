include $(QCX_CONFIG_DIR)/make.d/pretty.mk

OBJ_DIR = obj

SRCS = $(wildcard *.cpp)
OBJS = $(addprefix $(OBJ_DIR)/,$(SRCS:.cpp=.o))
DEPS = $(OBJS:.o=.d)

LIB_DIR = $(SAM_DIR)/lib
LIB = $(LIB_DIR)/libsam.a

.PHONY: all
all: $(LIB)

$(LIB): $(OBJS) | $(LIB_DIR)
	$(call qcx_print,AR,$@)
	$(Q)$(AR) rcs $@ $^

# Compile into obj/ (keeps src/ source-only -- no .o/.d beside the sources),
# with -MMD -MP header-dependency tracking so editing a header recompiles
# exactly the objects that need it.
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(call qcx_print,CXX,$@)
	$(Q)$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(LIB_DIR) $(OBJ_DIR):
	$(Q)mkdir -p $@

-include $(DEPS)

.PHONY: clean
clean:
	$(Q)rm -rf $(OBJ_DIR) $(LIB)
