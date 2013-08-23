#
# Rules to (help) build the F4xx device support.
#

#
# Directory containing this makefile
#
PIOS_DEVLIB			:=	$(dir $(lastword $(MAKEFILE_LIST)))
include $(PIOS_DEVLIB)/library.mk

#
# Hardcoded linker script names for now
#
LINKER_SCRIPTS_APP	 =	$(PIOS_DEVLIB)/link_STM32F4xx_OP_memory.ld \
						$(PIOS_DEVLIB)/link_STM32F4xx_sections.ld

#
# PIOS device library source and includes
#
SRC					+=	$(wildcard $(PIOS_DEVLIB)*.c)
