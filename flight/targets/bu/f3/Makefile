 #####
 # Project: OpenPilot
 #    
 #  
 # Makefile for OpenPilot project build PiOS and the AP.
 #
 # The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2009.
 #    
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
 #####

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

## Bootloader Updater (common)
SRC += $(BUCOMMONDIR)/main.c
SRC += $(BUCOMMONDIR)/bu_payload.c
SRC += $(BUCOMMONDIR)/pios_board.c

## Bootloader Updater (architecture specific)

## Bootloader Updater (board specific implementations)

## PIOS Hardware (STM32F30x)
PIOS_DEVLIB			:=	$(PIOS)/STM32F30x
include $(PIOS_DEVLIB)/library.mk

SRC += $(PIOS_DEVLIB)/startup.c
SRC += $(PIOS_DEVLIB)/pios_sys.c
SRC += $(PIOS_DEVLIB)/pios_led.c
SRC += $(PIOS_DEVLIB)/pios_flash_internal.c
SRC += $(PIOS_DEVLIB)/pios_irq.c

## PIOS Hardware (Common)
SRC += $(PIOSCOMMON)/pios_delay.c
SRC += $(PIOSCOMMON)/pios_flash.c
SRC += $(PIOSCOMMON)/pios_heap.c
SRC += $(PIOSCOMMON)/pios_semaphore.c


# List Assembler source files here.
# Make them always end in a capital .S. Files ending in a lowercase .s
# will not be considered source files but generated files (assembler
# output from the compiler), and will be deleted upon "make clean"!
# Even though the DOS/Win* filesystem matches both .s and .S the same,
# it will preserve the spelling of the filenames, and gcc itself does
# care about how the name is spelled on its command-line.
#ASRC = $(PIOSSTM32F10X)/startup_stm32f10x_$(MODEL)$(MODEL_SUFFIX).S

# List any extra directories to look for include files here.
#    Each directory must be seperated by a space.
EXTRAINCDIRS  += $(PIOS)
EXTRAINCDIRS  += $(PIOSINC)
EXTRAINCDIRS  += $(FLIGHTLIBINC)
EXTRAINCDIRS  += $(BOARD_INFO_DIR)
EXTRAINCDIRS  += $(BUBOARDDIR)
EXTRAINCDIRS  += $(BUARCHDIR)
EXTRAINCDIRS  += $(BUCOMMONDIR)

# Place project-specific -D (define) and/or 
# -U options for C here.
CDEFS += -DMEM_SIZE=$(FW_BANK_SIZE)
CDEFS += -DUSE_$(BOARD)
CDEFS += -DBU_PAYLOAD_FILE=$(ROOT_DIR)/build/bl_$(BOARD_NAME)/bl_$(BOARD_NAME).bin

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

CFLAGS += $(ARCHFLAGS) -mthumb
CFLAGS += $(CDEFS)
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
ASFLAGS  = $(ARCHFLAGS) -mthumb -I. -x assembler-with-cpp
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
LINKER_SCRIPTS_APP	 =	$(PIOS_DEVLIB)/link_STM32F30x_FW_memory.ld \
						$(PIOS_DEVLIB)/link_STM32F30x_sections.ld
LDFLAGS += $(addprefix -T,$(LINKER_SCRIPTS_APP))

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

$(eval $(call TLFW_TEMPLATE,$(OUTDIR)/$(TARGET).bin,$(BOARD_TYPE),$(BOARD_REVISION)))

.PHONY: elf hex bin tlfw
elf: $(OUTDIR)/$(TARGET).elf
hex: $(OUTDIR)/$(TARGET).hex
bin: $(OUTDIR)/$(TARGET).bin
tlfw: $(OUTDIR)/$(TARGET).tlfw

# Display sizes of sections.
$(eval $(call SIZE_TEMPLATE, $(OUTDIR)/$(TARGET).elf))

# Generate Doxygen documents
docs:
	doxygen  $(DOXYGENDIR)/doxygen.cfg

# Install: install binary file with prefix/suffix into install directory
install: $(OUTDIR)/$(TARGET).tlfw
ifneq ($(INSTALL_DIR),)
	@echo $(MSG_INSTALLING) $(call toprel, $<)
	$(V1) mkdir -p $(INSTALL_DIR)
	$(V1) $(INSTALL) $< $(INSTALL_DIR)/$(INSTALL_PFX)$(TARGET)$(INSTALL_SFX).tlfw
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
