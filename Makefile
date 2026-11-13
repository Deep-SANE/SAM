# SAM — self-contained build. No environment variables required:
#   make config     resolve dependencies from u3lib, u4lib, untou3, Spectra, MPI
#   make            build lib/libsam.a and bin/* applications
#   make lib        build only lib/libsam.a
#   make apps       build only the programs in app/tutorial and app/work
#                   (-> bin/tutorial, bin/work)
#   make test       build and run the regression tests in test/
#   make clean      clean built objects, lib, and binaries
#   make distclean  reset to pristine state
#   make data       download the released operator expansions into the data
#                   directory (OPS="Kinetic N2LO_opt_bare" for some only)
#   make data-verify  check downloaded files against the release checksums
#   make data-list  show what the data release contains

SAM_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CONFIG_DIR := $(SAM_ROOT)/config
export SAM_ROOT CONFIG_DIR

NOCONFIG_GOALS := config reconfigure clean distclean help data-list

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

.PHONY: test
test: lib
	$(MAKE) -C test

.PHONY: config reconfigure
config reconfigure:
	@bash $(CONFIG_DIR)/configure.sh $(CONFIG_ARGS)

.PHONY: clean
clean:
	rm -rf $(SAM_ROOT)/bin $(SAM_ROOT)/lib
	$(MAKE) -C src clean 2>/dev/null || true
	$(MAKE) -C app clean 2>/dev/null || true
	$(MAKE) -C test clean 2>/dev/null || true

.PHONY: distclean
distclean: clean
	rm -f $(CONFIG_DIR)/config.generated.mk

# Released data (see config/data_release/README.md). Files go into the data directory
# chosen at `make config`; `make data SAM_DATA_DIR=/path` overrides it.
.PHONY: data data-verify data-list
data:
	@SAM_DATA_DIR="$(SAM_DATA_DIR)" FORCE="$(FORCE)" \
	  bash $(SAM_ROOT)/config/data_release/fetch.sh get $(OPS)

data-verify:
	@SAM_DATA_DIR="$(SAM_DATA_DIR)" bash $(SAM_ROOT)/config/data_release/fetch.sh verify

data-list:
	@bash $(SAM_ROOT)/config/data_release/fetch.sh list
