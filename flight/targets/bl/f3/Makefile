###############################################################################
# @file       Makefile
# @author     The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2009.
# @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
# @addtogroup 
# @{
# @addtogroup 
# @{
# @brief Makefile to build bootloader for STM32F3 boards
###############################################################################
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#

ifndef MAKE_INC_DIR
$(error This Makefile cannot be run directly)
endif
include $(MAKE_INC_DIR)/firmware-defs.mk
include $(BOARD_INFO_DIR)/board-info.mk

# Paths
PIOSINC = $(PIOS)/inc
PIOSCOMMON = $(PIOS)/Common
PIOSCOMMONLIB = $(PIOSCOMMON)/Libraries
FLIGHTLIBINC = $(FLIGHTLIB)/inc

# List C source files here. (C dependencies are automatically generated.)
# use file-extension c for "c-only"-files

## Bootloader (architecture specific)
SRC += $(BLCOMMONDIR)/main.c
SRC += $(BLCOMMONDIR)/led_pwm.c
SRC += $(BLCOMMONDIR)/bl_xfer.c

## Bootloader (board specific implementations)
SRC += $(BLBOARDDIR)/pios_board.c
SRC += $(BLBOARDDIR)/pios_usb_board_data.c

## PIOS Hardware (STM32F30x)
PIOS_DEVLIB			:=	$(PIOS)/STM32F30x
include $(PIOS_DEVLIB)/library.mk
SRC	+=	$(PIOS_DEVLIB)/pios_bl_helper.c
SRC	+=	$(PIOS_DEVLIB)/pios_exti.c
SRC	+=	$(PIOS_DEVLIB)/pios_flash_internal.c
SRC	+=	$(PIOS_DEVLIB)/pios_gpio.c
SRC	+=	$(PIOS_DEVLIB)/pios_iap.c
SRC	+=	$(PIOS_DEVLIB)/pios_irq.c
SRC	+=	$(PIOS_DEVLIB)/pios_led.c
SRC	+=	$(PIOS_DEVLIB)/pios_rtc.c
SRC	+=	$(PIOS_DEVLIB)/pios_spi.c
SRC	+=	$(PIOS_DEVLIB)/pios_sys.c
SRC	+=	$(PIOS_DEVLIB)/pios_tim.c
SRC	+=	$(PIOS_DEVLIB)/pios_usb.c
SRC	+=	$(PIOS_DEVLIB)/pios_usb_cdc.c
SRC	+=	$(PIOS_DEVLIB)/pios_usb_hid.c
SRC	+=	$(PIOS_DEVLIB)/pios_usb_hid_istr.c
SRC	+=	$(PIOS_DEVLIB)/pios_usb_hid_pwr.c
SRC	+=	$(PIOS_DEVLIB)/pios_usbhook.c
SRC	+=	$(PIOS_DEVLIB)/pios_wdg.c
SRC	+=	$(PIOS_DEVLIB)/startup.c
SRC	+=	$(PIOS_DEVLIB)/vectors_stm32f30x.c
#
# Hardcoded linker script names for now
#
LINKER_SCRIPTS_BL	 =	$(PIOS_DEVLIB)/link_STM32F30x_BL_memory.ld \
						$(PIOS_DEVLIB)/link_STM32F30x_sections.ld

#
# ST USB FS library
#
SRC					+=	$(addprefix $(USBFSLIB)/src/,$(USBFSLIB_SRC))

## PIOS Hardware (Common)
SRC += $(PIOSCOMMON)/pios_delay.c
SRC += $(PIOSCOMMON)/pios_board_info.c
SRC += $(PIOSCOMMON)/pios_com_msg.c
SRC += $(PIOSCOMMON)/printf-stdarg.c
SRC += $(PIOSCOMMON)/pios_usb_desc_hid_only.c
SRC += $(PIOSCOMMON)/pios_usb_util.c
SRC += $(PIOSCOMMON)/pios_flash.c
SRC += $(PIOSCOMMON)/pios_heap.c
SRC += $(PIOSCOMMON)/pios_semaphore.c


# List any extra directories to look for include files here.
#    Each directory must be seperated by a space.
EXTRAINCDIRS  += $(PIOS)
EXTRAINCDIRS  += $(PIOSINC)
EXTRAINCDIRS  += $(FLIGHTLIBINC)
EXTRAINCDIRS  += $(STMSPDINCDIR)
EXTRAINCDIRS  += $(BOARD_INFO_DIR)
EXTRAINCDIRS  += $(BLBOARDDIR)
EXTRAINCDIRS  += $(BLARCHDIR)
EXTRAINCDIRS  += $(BLCOMMONDIR)

# Place project-specific -D (define) and/or 
# -U options for C here.
CDEFS += -DSYSCLK_FREQ=$(SYSCLK_FREQ)
CDEFS += -DUSE_$(BOARD)

# Provide (only) the bootloader with board-specific defines
BLONLY_CDEFS += -DBOARD_TYPE=$(BOARD_TYPE)
BLONLY_CDEFS += -DBOARD_REVISION=$(BOARD_REVISION)
BLONLY_CDEFS += -DHW_TYPE=$(HW_TYPE)
BLONLY_CDEFS += -DBOOTLOADER_VERSION=$(BOOTLOADER_VERSION)
BLONLY_CDEFS += -DFW_BANK_BASE=$(FW_BANK_BASE)
BLONLY_CDEFS += -DFW_BANK_SIZE=$(FW_BANK_SIZE)
BLONLY_CDEFS += -DFW_DESC_BASE=$(FW_DESC_BASE)
BLONLY_CDEFS += -DFW_DESC_SIZE=$(FW_DESC_SIZE)
BLONLY_CDEFS += -DEE_BANK_BASE=$(EE_BANK_BASE)
BLONLY_CDEFS += -DEE_BANK_SIZE=$(EE_BANK_SIZE)

# Place project-specific -D and/or -U options for 
# Assembler with preprocessor here.
ADEFS = -D__ASSEMBLY__

#-----

# Compiler flags.

#  -g*:          generate debugging information
#  -O*:          optimization level
#  -f...:        tuning, see GCC manual and avr-libc documentation
#  -Wall...:     warning level
#  -Wa,...:      tell GCC to pass this to the assembler.
#    -adhlns...: create assembler listing
#
# Flags for C and C++ (arm-elf-gcc/arm-elf-g++)

# This is not the best place for these.  Really should abstract out
# to the board file or something
CFLAGS += -DSTM32F30X
CFLAGS += -DMEM_SIZE=262144

CFLAGS += -gdwarf-2
CFLAGS += -Os
CFLAGS += -fdata-sections -ffunction-sections

CFLAGS += $(ARCHFLAGS)
CFLAGS += $(CDEFS)
CFLAGS += $(BLONLY_CDEFS)
CFLAGS += $(patsubst %,-I%,$(EXTRAINCDIRS)) -I.

CFLAGS += -mapcs-frame 
CFLAGS += -fomit-frame-pointer

CFLAGS += -Wall
CFLAGS += -Werror
CFLAGS += -Wa,-adhlns=$(addprefix $(OUTDIR)/, $(notdir $(addsuffix .lst, $(basename $<))))
# Compiler flags to generate dependency files:
CFLAGS += -MD -MP -MF $(OUTDIR)/dep/$(@F).d

# flags only for C
CONLYFLAGS += -std=gnu99

# Assembler flags.
#  -Wa,...:    tell GCC to pass this to the assembler.
#  -ahlns:     create listing
ASFLAGS  = $(ARCHFLAGS) -I. -x assembler-with-cpp
ASFLAGS += $(ADEFS)
ASFLAGS += -Wa,-adhlns=$(addprefix $(OUTDIR)/, $(notdir $(addsuffix .lst, $(basename $<))))
ASFLAGS += $(patsubst %,-I%,$(EXTRAINCDIRS))

# Linker flags.
#  -Wl,...:     tell GCC to pass this to linker.
#    -Map:      create map file
#    --cref:    add cross reference to  map file
LDFLAGS = -nostartfiles -Wl,-Map=$(OUTDIR)/$(TARGET).map,--cref,--gc-sections
LDFLAGS += -Wl,-static
LDFLAGS += -lc -lgcc 
LDFLAGS += -Wl,--warn-common
LDFLAGS += -Wl,--fatal-warnings

# Linker scripts
#LDFLAGS += $(addprefix -T,$(LINKER_SCRIPTS_BL))
LDFLAGS += $(addprefix -T,$(LINKER_SCRIPTS_BL))

# List of all source files.
ALLSRC     = $(ASRC) $(SRC)
# List of all source files without directory and file-extension.
ALLSRCBASE = $(notdir $(basename $(ALLSRC)))

# Define all object files.
ALLOBJ     = $(addprefix $(OUTDIR)/, $(addsuffix .o, $(ALLSRCBASE)))

# Link: create ELF output file from object files.
$(eval $(call LINK_TEMPLATE, $(OUTDIR)/$(TARGET).elf, $(ALLOBJ)))

# Assemble: create object files from assembler source files.
$(foreach src, $(ASRC), $(eval $(call ASSEMBLE_TEMPLATE, $(src)))) 

# Compile: create object files from C source files.
$(foreach src, $(SRC), $(eval $(call COMPILE_C_TEMPLATE, $(src)))) 

# Compile: create assembler files from C source files. ARM/Thumb
$(eval $(call PARTIAL_COMPILE_TEMPLATE, SRC))

# Add jtag targets (program and wipe)
$(eval $(call JTAG_TEMPLATE,$(OUTDIR)/$(TARGET).bin,$(BL_BANK_BASE),$(BL_BANK_SIZE),$(OPENOCD_JTAG_CONFIG),$(OPENOCD_CONFIG)))

.PHONY: hex bin
hex: $(OUTDIR)/$(TARGET).hex
bin: $(OUTDIR)/$(TARGET).bin

# Display sizes of sections.
$(eval $(call SIZE_TEMPLATE, $(OUTDIR)/$(TARGET).elf))

# Generate Doxygen documents
docs:
	doxygen  $(DOXYGENDIR)/doxygen.cfg

# Install: install binary file with prefix/suffix into install directory
install: $(OUTDIR)/$(TARGET).bin
ifneq ($(INSTALL_DIR),)
	@echo $(MSG_INSTALLING) $(call toprel, $<)
	$(V1) mkdir -p $(INSTALL_DIR)
	$(V1) $(INSTALL) $< $(INSTALL_DIR)/$(INSTALL_PFX)$(TARGET)$(INSTALL_SFX).bin
else
	$(error INSTALL_DIR must be specified for $@)
endif

# Create output files directory
# all known MS Windows OS define the ComSpec environment variable
ifdef ComSpec
$(shell md $(subst /,\\,$(OUTDIR)) 2>NUL)
else
$(shell mkdir -p $(OUTDIR) 2>/dev/null)
endif

# Include the dependency files.
ifdef ComSpec
-include $(shell md $(subst /,\\,$(OUTDIR))\dep 2>NUL) $(wildcard $(OUTDIR)/dep/*)
else
-include $(shell mkdir -p $(OUTDIR) 2>/dev/null) $(shell mkdir $(OUTDIR)/dep 2>/dev/null) $(wildcard $(OUTDIR)/dep/*)
endif

# Listing of phony targets.
.PHONY : install
