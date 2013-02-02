#
# Rules to add CMSIS3 DSP Library to a target
#

CMSIS3_DSPLIB_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
SRC += $(wildcard $(CMSIS3_DSPLIB_DIR)Source/*/*.c)
EXTRAINCDIRS += $(CMSIS3_DSPLIB_DIR)Include

