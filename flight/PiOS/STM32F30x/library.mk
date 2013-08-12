#
# Rules to (help) build the F30x device support.
#

#
# Hardcoded linker script names for now
#
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
EXTRAINCDIRS		+=	$(PIOS_DEVLIB)/inc

#
# CMSIS for the F3
#
include $(PIOSCOMMONLIB)/CMSIS3/library.mk
CMSIS3_DEVICEDIR	:=	$(PIOS_DEVLIB)/Libraries/CMSIS3/Device/ST/STM32F30x
SRC					+=      $(BOARD_INFO_DIR)/cmsis_system.c
EXTRAINCDIRS		+=	$(CMSIS3_DEVICEDIR)/Include

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

#
# FreeRTOS
#
# If the application has included the generic FreeRTOS support, then add in
# the device-specific pieces of the code.
#
ifneq ($(FREERTOS_DIR),)
FREERTOS_PORTDIR	:=	$(PIOS_DEVLIB)/Libraries/FreeRTOS/Source
EXTRAINCDIRS		+=	$(FREERTOS_PORTDIR)/portable/GCC/ARM_CM4F
SRC					+=	$(wildcard $(FREERTOS_PORTDIR)/portable/GCC/ARM_CM4F/*.c)
endif


