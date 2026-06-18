include $(QCX_CONFIG_DIR)/make.d/pretty.mk

OBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))

LIB_DIR = $(SAM_DIR)/lib

LIB = $(LIB_DIR)/libsam.a

.PHONY: all
all: $(LIB)

$(LIB): $(OBJS)
	$(Q)mkdir -p $(LIB_DIR)
	$(call qcx_print,AR,$@)
	$(Q)$(AR) rcs $@ $^

%.o: %.cpp
	$(call qcx_print,CXX,$@)
	$(Q)$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o $(LIB)
