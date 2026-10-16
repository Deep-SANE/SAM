# ---------------------------------------------------------------------------
# Pretty / quiet build output.
#
#   Default      : one short coloured line per action, e.g.  "  CXX   foo.o"
#   make V=1     : restore the full command lines (the old, verbose behaviour)
# ---------------------------------------------------------------------------

PRETTY_LIB_ROOT := $(abspath $(CURDIR)/..)
PRETTY_LIB_NAME := $(notdir $(PRETTY_LIB_ROOT))

pretty_tgt = $(PRETTY_LIB_NAME)/$(lastword $(subst /$(PRETTY_LIB_NAME)/, ,$(abspath $(1))))

ifeq ($(V),1)
  Q :=
  pretty_print = @:
else
  Q := @
  pretty_print = @printf '  \033[1;34m%-5s\033[0m %s\n' '$(1)' '$(call pretty_tgt,$(2))'
endif
