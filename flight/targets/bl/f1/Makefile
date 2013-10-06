###############################################################################
# @file       Makefile
# @author     The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2009.
# @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
# @addtogroup 
# @{
# @addtogroup 
# @{
# @brief Makefile to build bootloader for STM32F1 boards
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
PIOSSTM32F10X = $(PIOS)/STM32F10x
PIOSCOMMON = $(PIOS)/Common
PIOSCOMMONLIB = $(PIOSCOMMON)/Libraries
FLIGHTLIBINC = $(FLIGHTLIB)/inc
APPLIBDIR = $(PIOSSTM32F10X)/Libraries
STMLIBDIR = $(APPLIBDIR)
STMSPDDIR = $(STMLIBDIR)/STM32F10x_StdPeriph_Driver
STMUSBDIR = $(STMLIBDIR)/STM32_USB-FS-Device_Driver
STMSPDSRCDIR = $(STMSPDDIR)/src
STMSPDINCDIR = $(STMSPDDIR)/inc
STMUSBSRCDIR = $(STMUSBDIR)/src
STMUSBINCDIR = $(STMUSBDIR)/inc
CMSISDIR  = $(STMLIBDIR)/CMSIS/Core/CM3

# List C source files here. (C dependencies are automatically generated.)
# use file-extension c for "c-only"-files

## Bootloader (architecture specific)
SRC += $(BLARCHDIR)/main.c
SRC += $(BLARCHDIR)/op_dfu.c

## Bootloader (board specific implementations)
SRC += $(BLBOARDDIR)/pios_board.c
SRC += $(BLBOARDDIR)/pios_usb_board_data.c

## PIOS Hardware (STM32F10x)
SRC += $(PIOSSTM32F10X)/pios_sys.c
SRC += $(PIOSSTM32F10X)/pios_led.c
SRC += $(PIOSSTM32F10X)/pios_irq.c
SRC += $(PIOSSTM32F10X)/pios_gpio.c
SRC += $(PIOSSTM32F10X)/pios_iap.c
SRC += $(PIOSSTM32F10X)/pios_bl_helper.c

## PIOS Hardware (USB)
SRC += $(PIOSSTM32F10X)/pios_usb.c
SRC += $(PIOSSTM32F10X)/pios_usbhook.c
SRC += $(PIOSSTM32F10X)/pios_usb_hid.c
SRC += $(PIOSSTM32F10X)/pios_usb_hid_istr.c
SRC += $(PIOSSTM32F10X)/pios_usb_hid_pwr.c

## PIOS Hardware (Common)
SRC += $(PIOSCOMMON)/pios_delay.c
SRC += $(PIOSCOMMON)/pios_board_info.c
SRC += $(PIOSCOMMON)/pios_com_msg.c
SRC += $(PIOSCOMMON)/printf-stdarg.c
SRC += $(PIOSCOMMON)/pios_usb_desc_hid_only.c
SRC += $(PIOSCOMMON)/pios_usb_util.c
SRC += $(PIOSCOMMON)/pios_heap.c
SRC += $(PIOSCOMMON)/pios_semaphore.c


## Libraries for flight calculations
SRC += $(FLIGHTLIB)/fifo_buffer.c

## CMSIS for STM32
include $(PIOSCOMMONLIB)/CMSIS3/library.mk
SRC += $(BOARD_INFO_DIR)/cmsis_system.c

## Used parts of the STM-Library
SRC += $(STMSPDSRCDIR)/stm32f10x_bkp.c
SRC += $(STMSPDSRCDIR)/stm32f10x_crc.c
SRC += $(STMSPDSRCDIR)/stm32f10x_flash.c
SRC += $(STMSPDSRCDIR)/stm32f10x_gpio.c
SRC += $(STMSPDSRCDIR)/stm32f10x_pwr.c
SRC += $(STMSPDSRCDIR)/stm32f10x_rcc.c
SRC += $(STMSPDSRCDIR)/stm32f10x_rtc.c
SRC += $(STMSPDSRCDIR)/stm32f10x_spi.c
SRC += $(STMSPDSRCDIR)/stm32f10x_dbgmcu.c
SRC += $(STMSPDSRCDIR)/misc.c

## STM32 USB Library
SRC += $(STMUSBSRCDIR)/usb_core.c
SRC += $(STMUSBSRCDIR)/usb_init.c
SRC += $(STMUSBSRCDIR)/usb_int.c
SRC += $(STMUSBSRCDIR)/usb_mem.c
SRC += $(STMUSBSRCDIR)/usb_regs.c
SRC += $(STMUSBSRCDIR)/usb_sil.c

# List Assembler source files here.
# Make them always end in a capital .S. Files ending in a lowercase .s
# will not be considered source files but generated files (assembler
# output from the compiler), and will be deleted upon "make clean"!
# Even though the DOS/Win* filesystem matches both .s and .S the same,
# it will preserve the spelling of the filenames, and gcc itself does
# care about how the name is spelled on its command-line.
ASRC = $(PIOSSTM32F10X)/startup_stm32f10x_$(MODEL)$(MODEL_SUFFIX).S

# List any extra directories to look for include files here.
#    Each directory must be seperated by a space.
EXTRAINCDIRS  += $(PIOS)
EXTRAINCDIRS  += $(PIOSINC)
EXTRAINCDIRS  += $(FLIGHTLIBINC)
EXTRAINCDIRS  += $(PIOSSTM32F10X)/inc
EXTRAINCDIRS  += $(STMSPDINCDIR)
EXTRAINCDIRS  += $(STMUSBINCDIR)
EXTRAINCDIRS  += $(CMSISDIR)
EXTRAINCDIRS  += $(BOARD_INFO_DIR)
EXTRAINCDIRS  += $(BLBOARDDIR)
EXTRAINCDIRS  += $(BLARCHDIR)
EXTRAINCDIRS  += $(BLCOMMONDIR)

# Path to Linker-Scripts
LINKERSCRIPTPATH = $(PIOSSTM32F10X)

# Place project-specific -D (define) and/or 
# -U options for C here.
CDEFS = -DSTM32F10X_$(MODEL)
CDEFS += -DUSE_STDPERIPH_DRIVER
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

CFLAGS += -gdwarf-2
CFLAGS += -Os
CFLAGS += -ffunction-sections

CFLAGS += -mcpu=$(MCU)
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
ASFLAGS  = -mcpu=$(MCU) -I. -x assembler-with-cpp
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

# Set linker-script name depending on selected submodel name
LDFLAGS += -T$(LINKERSCRIPTPATH)/link_$(BOARD)_memory.ld
LDFLAGS += -T$(LINKERSCRIPTPATH)/link_$(BOARD)_BL_sections.ld

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
