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
# PIOS device library source and includes
#
SRC					+=	$(PIOS_DEVLIB)/pios_bl_helper.c
SRC					+=	$(PIOS_DEVLIB)/pios_delay.c
SRC					+=	$(PIOS_DEVLIB)/pios_exti.c
SRC					+=	$(PIOS_DEVLIB)/pios_flash_internal.c
SRC					+=	$(PIOS_DEVLIB)/pios_gpio.c
SRC					+=	$(PIOS_DEVLIB)/pios_iap.c
SRC					+=	$(PIOS_DEVLIB)/pios_irq.c
SRC					+=	$(PIOS_DEVLIB)/pios_led.c
SRC					+=	$(PIOS_DEVLIB)/pios_rtc.c
SRC					+=	$(PIOS_DEVLIB)/pios_spi.c
SRC					+=	$(PIOS_DEVLIB)/pios_sys.c
SRC					+=	$(PIOS_DEVLIB)/pios_tim.c
SRC					+=	$(PIOS_DEVLIB)/pios_usb.c
SRC					+=	$(PIOS_DEVLIB)/pios_usb_cdc.c
SRC					+=	$(PIOS_DEVLIB)/pios_usb_hid.c
SRC					+=	$(PIOS_DEVLIB)/pios_usbhook.c
SRC					+=	$(PIOS_DEVLIB)/pios_wdg.c
SRC					+=	$(PIOS_DEVLIB)/startup.c
SRC					+=	$(PIOS_DEVLIB)/vectors_stm32f4xx.c
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
