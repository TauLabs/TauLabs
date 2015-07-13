#
# Rules to (help) build the F4xx device support.
#

#
# Directory containing this makefile
#
PIOS_DEVLIB			:=	$(dir $(lastword $(MAKEFILE_LIST)))

#
# Hardcoded linker script names for now
#
#LINKER_SCRIPTS_APP	 =	$(PIOS_DEVLIB)/link_STM32F4xx_OP_memory.ld \
						$(PIOS_DEVLIB)/link_STM32F4xx_sections.ld

#LINKER_SCRIPTS_BL	 =	$(PIOS_DEVLIB)/link_STM32F4xx_BL_memory.ld \
						$(PIOS_DEVLIB)/link_STM32F4xx_sections.ld

#
# Compiler options implied by posix
#
ARCHFLAGS			+= -DARCH_POSIX
ARCHFLAGS			+= -D_GNU_SOURCE
ARCHFLAGS			+= -m32
