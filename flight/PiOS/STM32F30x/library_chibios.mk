#
# Rules to (help) build the F30x device support.
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
# Compiler options implied by the F30x
#
CDEFS				+= -DSTM32F30X
CDEFS				+= -DHSE_VALUE=$(OSCILLATOR_FREQ)
CDEFS 				+= -DUSE_STDPERIPH_DRIVER
ARCHFLAGS			+= -mcpu=cortex-m4 -march=armv7e-m -mfpu=fpv4-sp-d16 -mfloat-abi=hard

#
# PIOS device library source and includes
#
SRC					+=	$(filter-out $(PIOS_DEVLIB)vectors_stm32f30x.c $(PIOS_DEVLIB)startup.c, $(wildcard $(PIOS_DEVLIB)*.c))
EXTRAINCDIRS		+=	$(PIOS_DEVLIB)/inc

#
# ST Peripheral library
#
PERIPHLIB			 =	$(PIOS_DEVLIB)/Libraries/STM32F30x_StdPeriph_Driver
EXTRAINCDIRS		+=	$(PERIPHLIB)/inc
SRC					+=	$(wildcard $(PERIPHLIB)/src/*.c)

#
# ST USB FS library
#
USBFSLIB			=	$(PIOS_DEVLIB)/Libraries/STM32_USB-FS-Device_Driver
USBFSLIB_SRC		=	usb_core.c usb_init.c usb_int.c usb_mem.c usb_regs.c usb_sil.c
EXTRAINCDIRS		+=	$(USBFSLIB)/inc
SRC					+=	$(addprefix $(USBFSLIB)/src/,$(USBFSLIB_SRC))

