include $(QCX_CONFIG_DIR)/make.d/pretty.mk

BIN_DIR = $(SAM_DIR)/bin
OBJ_DIR = obj

SRCS = $(wildcard *.cpp)
OBJS = $(addprefix $(OBJ_DIR)/,$(SRCS:.cpp=.o))
DEPS = $(OBJS:.o=.d)
PRGS = $(addprefix $(BIN_DIR)/,$(basename $(SRCS)))
SAM_LIB = $(SAM_DIR)/lib/libsam.a

.PHONY: all
all: $(PRGS)

# Link each driver against libsam. The $(SAM_LIB) prerequisite means
# recompiling the library relinks every driver -- and when only the library
# implementation changed (no header), it is a relink, not a recompile.
$(BIN_DIR)/%: $(OBJ_DIR)/%.o $(SAM_LIB) | $(BIN_DIR)
	$(call qcx_print,LINK,$@)
	$(Q)$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Compile into obj/ (keeps app/ source-only -- no .o/.d beside the drivers)
# with -MMD -MP header-dependency tracking so a library-header edit rebuilds
# exactly the affected drivers.
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(call qcx_print,CXX,$@)
	$(Q)$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	$(Q)mkdir -p $@

-include $(DEPS)

.PHONY: clean
clean:
	$(Q)rm -rf $(OBJ_DIR) $(PRGS)
