ifdef BOOST_ROOT
CXXFLAGS += -I$(BOOST_ROOT)/include
endif

export

.PHONY: all
all: 
	$(MAKE) -C src

.PHONY: clean
clean:
	$(MAKE) -C src clean
