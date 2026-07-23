include $(QCX_CONFIG_DIR)/make.d/pretty.mk

OBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
DEPS = $(OBJS:.o=.d)

LIB_DIR = $(SAM_DIR)/lib

LIB = $(LIB_DIR)/libsam.a

.PHONY: all
all: $(LIB)

$(LIB): $(OBJS)
	$(Q)mkdir -p $(LIB_DIR)
	$(call qcx_print,AR,$@)
	$(Q)$(AR) rcs $@ $^

# -MMD -MP emits a per-object .d file listing the (non-system) headers it
# includes, so editing a header recompiles exactly the objects that need it.
%.o: %.cpp
	$(call qcx_print,CXX,$@)
	$(Q)$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

.PHONY: clean
clean:
	rm -f *.o *.d $(LIB)
