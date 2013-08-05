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
LINKER_SCRIPTS_BL	 =	$(PIOS_DEVLIB)/link_STM32F4xx_BL_memory.ld \
						$(PIOS_DEVLIB)/link_STM32F4xx_sections.ld

#
# ST Peripheral library
#
SRC					+=	$(wildcard $(PERIPHLIB)/src/*.c)

#
# ST USB OTG library
#
SRC				+=	$(addprefix $(USBOTGLIB)/src/,$(USBOTGLIB_SRC))

#
# ST USB Device library
#
SRC				+=	$(wildcard $(USBDEVLIB)/Core/src/*.c)

#
# FreeRTOS
#
# If the application has included the generic FreeRTOS support, then add in
# the device-specific pieces of the code.
#
SRC					+=	$(wildcard $(FREERTOS_PORTDIR)/portable/GCC/ARM_CM4F/*.c)
