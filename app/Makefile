include $(QCX_CONFIG_DIR)/make.d/pretty.mk

BIN_DIR = $(SAM_DIR)/bin

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
PRGS = $(addprefix $(BIN_DIR)/,$(basename $(SRCS)))
SAM_LIB = $(SAM_DIR)/lib/libsam.a

.PHONY: all
all: $(BIN_DIR) $(PRGS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: %.o $(SAM_LIB) | $(BIN)
	$(call qcx_print,LINK,$@)
	$(Q)$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

%.o: %.cpp
	$(call qcx_print,CXX,$@)
	$(Q)$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(PRGS) *.o

