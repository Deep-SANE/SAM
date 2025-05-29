#include $(SAM_DIR)/config.mk

BIN = bin
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
PRGS = $(addprefix $(BIN)/,$(basename $(SRCS)))
SAM_LIB = $(SAM_DIR)/src/libsam.a

.PHONY: all
all: $(PRGS)

$(BIN)/%: %.o $(SAM_LIB) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(PRGS) *.o