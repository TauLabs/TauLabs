#
# Rules to (help) build the F30x device support.
#

#
# Directory containing this makefile
#
PIOS_DEVLIB			:=	$(dir $(lastword $(MAKEFILE_LIST)))

include $(PIOS_DEVLIB)/library.mk

#
# Hardcoded linker script names for now
#
LINKER_SCRIPTS_APP	 =	$(PIOS_DEVLIB)/link_STM32F30x_FW_memory.ld \
						$(PIOS_DEVLIB)/link_STM32F30x_sections.ld

#
# PIOS device library source and includes
#
SRC					+=	$(wildcard $(PIOS_DEVLIB)*.c)

#
# ST Peripheral library
#
SRC					+=	$(wildcard $(PERIPHLIB)/src/*.c)

#
# ST USB FS library
#
SRC					+=	$(addprefix $(USBFSLIB)/src/,$(USBFSLIB_SRC))

#
# FreeRTOS
#
# If the application has included the generic FreeRTOS support, then add in
# the device-specific pieces of the code.
#
SRC					+=	$(wildcard $(FREERTOS_PORTDIR)/portable/GCC/ARM_CM4F/*.c)

