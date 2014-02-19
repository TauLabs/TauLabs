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
LINKER_SCRIPTS_APP	 =	$(PIOS_DEVLIB)/sections_chibios.ld

#
# Compiler options implied by the F4xx
#
CDEFS 				+= -DUSE_STDPERIPH_DRIVER
ARCHFLAGS			+= -mcpu=cortex-m4 -march=armv7e-m -mfpu=fpv4-sp-d16 -mfloat-abi=hard

#
# PIOS device library source and includes
#
SRC					+=	$(filter-out $(PIOS_DEVLIB)vectors_stm32f4xx.c $(PIOS_DEVLIB)startup.c, $(wildcard $(PIOS_DEVLIB)*.c))
EXTRAINCDIRS		+=	$(PIOS_DEVLIB)/inc

#
# ST Peripheral library
#
PERIPHLIB			 =	$(PIOS_DEVLIB)/Libraries/STM32F4xx_StdPeriph_Driver
EXTRAINCDIRS		+=	$(PERIPHLIB)/inc
SRC					+=	$(wildcard $(PERIPHLIB)/src/*.c)

#
# ST USB OTG library
#
USBOTGLIB			=	$(PIOS_DEVLIB)/Libraries/STM32_USB_OTG_Driver
USBOTGLIB_SRC			=	usb_core.c usb_dcd.c usb_dcd_int.c
EXTRAINCDIRS			+=	$(USBOTGLIB)/inc
SRC				+=	$(addprefix $(USBOTGLIB)/src/,$(USBOTGLIB_SRC))

#
# ST USB Device library
#
USBDEVLIB			=	$(PIOS_DEVLIB)/Libraries/STM32_USB_Device_Library
EXTRAINCDIRS			+=	$(USBDEVLIB)/Core/inc
SRC				+=	$(wildcard $(USBDEVLIB)/Core/src/*.c)

