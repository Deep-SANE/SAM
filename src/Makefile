OBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
LIB = libsam.a

.PHONY: all
all: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I../include -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o *.a