#
# Rules to (help) build the Posix-targeted code.
#

include $(MAKE_INC_DIR)/system-id.mk

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

# The posix code is not threaded but it is possibly re-entrant into the
# system libraries because of task switching.  As a result, request the
# re-entrant libraries and -D_REENTRANT
ifdef MACOSX
# If we are building with clang, it doesn't like the pthread argument being
# passed at link time.  Annoying.
CONLYFLAGS			+= -pthread
else
ARCHFLAGS			+= -pthread
endif

# Build 32 bit code.
ifdef AMD64
ARCHFLAGS                      += -m32
endif

