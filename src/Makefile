BIN = ../bin

SRCS = $(wildcard *.cpp)
PRGS = $(addprefix $(BIN)/,$(basename $(SRCS)))

.PHONY: all
all: $(PRGS)

$(BIN)/%: %.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(PRGS)
