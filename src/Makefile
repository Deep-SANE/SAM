OBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))

LIB_DIR = $(SAM_DIR)/lib

LIB = $(LIB_DIR)/libsam.a

.PHONY: all
all: $(LIB)

$(LIB): $(OBJS)
	mkdir -p $(LIB_DIR)
	$(AR) rcs $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o $(LIB)
