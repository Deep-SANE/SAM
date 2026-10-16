# SAM — self-contained build. No environment variables required:
#   make config     resolve dependencies from u3lib, u4lib, untou3, Spectra, MPI
#   make            build lib/libsam.a and bin/* applications
#   make lib        build only lib/libsam.a
#   make apps       build only bin/* applications
#   make clean      clean built objects, lib, and binaries
#   make distclean  reset to pristine state

SAM_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CONFIG_DIR := $(SAM_ROOT)/config
export SAM_ROOT CONFIG_DIR

NOCONFIG_GOALS := config reconfigure clean distclean help

ifeq ($(filter $(NOCONFIG_GOALS),$(MAKECMDGOALS)),)
include $(CONFIG_DIR)/config.mk
endif
export

.PHONY: all
all: lib apps

.PHONY: lib
lib:
	$(MAKE) -C src

.PHONY: apps
apps: lib
	$(MAKE) -C app

.PHONY: config reconfigure
config reconfigure:
	@bash $(CONFIG_DIR)/configure.sh $(CONFIG_ARGS)

.PHONY: clean
clean:
	rm -rf $(SAM_ROOT)/bin $(SAM_ROOT)/lib
	$(MAKE) -C src clean 2>/dev/null || true
	$(MAKE) -C app clean 2>/dev/null || true

.PHONY: distclean
distclean: clean
	rm -f $(CONFIG_DIR)/config.generated.mk
