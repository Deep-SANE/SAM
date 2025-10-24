include $(QCX_CONFIG_DIR)/make.d/*.mk

export

.PHONY: all
all: 
	$(MAKE) -C src
	$(MAKE) -C app

.PHONY: clean
clean:
	$(MAKE) -C src clean
	$(MAKE) -C app clean
