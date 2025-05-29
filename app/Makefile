BIN_DIR = $(SAM_DIR)/bin

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
PRGS = $(addprefix $(BIN_DIR)/,$(basename $(SRCS)))

LDFLAGS += -L$(SAM_DIR)/lib -lsam

.PHONY: all
all: $(BIN_DIR) $(PRGS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: %.o 
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(PRGS) *.o
