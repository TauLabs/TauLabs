#
# Rules to add ChibiOS/RT to a PiOS target
#
# Note that the PIOS target-specific makefile will detect that CHIBIOS_DIR
# has been defined and add in the target-specific pieces separately.
#

# ChibiOS
CHIBIOS := $(PIOSCOMMONLIB)/ChibiOS

include $(PIOSCOMMONLIB)/ChibiOS/os/hal/platforms/STM32F4xx/platform.mk
include $(PIOSCOMMONLIB)/ChibiOS/os/hal/hal.mk
include $(PIOSCOMMONLIB)/ChibiOS/os/ports/GCC/ARMCMx/STM32F4xx/port.mk
include $(PIOSCOMMONLIB)/ChibiOS/os/kernel/kernel.mk

SRC += $(PLATFORMSRC)
SRC += $(HALSRC)
SRC += $(PORTSRC)
SRC += $(KERNSRC)

EXTRAINCDIRS += $(PLATFORMINC)
EXTRAINCDIRS += $(HALINC)
EXTRAINCDIRS += $(PORTINC)
EXTRAINCDIRS += $(KERNINC)

