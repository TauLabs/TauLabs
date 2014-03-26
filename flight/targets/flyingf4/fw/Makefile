###############################################################################
# @file       Makefile
# @author     The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2009.
# @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
# @addtogroup 
# @{
# @addtogroup 
# @{
# @brief Makefile to build firmware for the FlyingF4 board.
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

include $(MAKE_INC_DIR)/firmware-defs.mk
include $(BOARD_INFO_DIR)/board-info.mk

# Set developer code and compile options
# Set to YES for debugging
DEBUG ?= NO
ERASE_FLASH ?= NO

# List of modules to include
MODULES = Sensors
MODULES += Attitude 
MODULES += Actuator
MODULES += ManualControl
MODULES += Stabilization
MODULES += FirmwareIAP
MODULES += Telemetry

OPTMODULES += GPS
OPTMODULES += PathPlanner
OPTMODULES += AltitudeHold 
OPTMODULES += VtolPathFollower 
OPTMODULES += FixedWingPathFollower
OPTMODULES += GroundPathFollower
OPTMODULES += CameraStab
OPTMODULES += Autotune
OPTMODULES += TxPID
OPTMODULES += VibrationAnalysis
OPTMODULES += UAVOMavlinkBridge
OPTMODULES += UAVOFrSKYSensorHubBridge
OPTMODULES += Battery
OPTMODULES += ComUsbBridge
OPTMODULES += Airspeed/revolution
OPTMODULES += UAVOLighttelemetryBridge
OPTMODULES += PicoC
OPTMODULES += UAVOFrSKYSPortBridge
OPTMODULES += Geofence

# Paths
OPUAVTALKINC = $(OPUAVTALK)/inc
OPUAVOBJINC = $(OPUAVOBJ)/inc
PIOSINC = $(PIOS)/inc
FLIGHTLIBINC = $(FLIGHTLIB)/inc
MATHLIB = $(FLIGHTLIB)/math
MATHLIBINC = $(FLIGHTLIB)/math
RSCODE = $(FLIGHTLIB)/rscode
RSCODEINC = $(FLIGHTLIB)/rscode
PIOSSTM32F4XX = $(PIOS)/STM32F4xx
PIOSCOMMON = $(PIOS)/Common
PIOSCOMMONLIB = $(PIOSCOMMON)/Libraries
APPLIBDIR = $(PIOSSTM32F4XX)/Libraries
STMLIBDIR = $(APPLIBDIR)
STMSPDDIR = $(STMLIBDIR)/STM32F4xx_StdPeriph_Driver
STMSPDSRCDIR = $(STMSPDDIR)/src
STMSPDINCDIR = $(STMSPDDIR)/inc
DEBUG_CM3_DIR = $(PIOSCOMMONLIB)/Debug
DEBUG_CM3_DIR_INC = $(DEBUG_CM3_DIR)/inc
OPUAVOBJINC = $(OPUAVOBJ)/inc
FLIGHTPLANLIB = $(OPMODULEDIR)/FlightPlan/lib
FLIGHTPLANS = $(OPMODULEDIR)/FlightPlan/flightplans
MAVLINKINC = $(FLIGHTLIB)/mavlink/v1.0/common

# optional component libraries
include $(PIOSCOMMONLIB)/FreeRTOS/library.mk
#include $(PIOSCOMMONLIB)/dosfs/library.mk
include $(FLIGHTLIB)/CMSIS3/DSP_Lib/library.mk


# List C source files here. (C dependencies are automatically generated.)
# use file-extension c for "c-only"-files

## MODULES
SRC += ${foreach MOD, ${MODULES} ${OPTMODULES}, ${wildcard ${OPMODULEDIR}/${MOD}/*.c}}
## OPENPILOT CORE:
SRC += ${OPMODULEDIR}/System/systemmod.c
SRC += main.c
SRC += pios_board.c
SRC += pios_usb_board_data.c
SRC += $(FLIGHTLIB)/alarms.c
SRC += $(OPUAVTALK)/uavtalk.c
SRC += $(OPUAVOBJ)/uavobjectmanager.c
SRC += $(OPUAVOBJ)/eventdispatcher.c

ifeq ($(DEBUG),YES)
SRC += $(DEBUG_CM3_DIR)/dcc_stdio.c
SRC += $(DEBUG_CM3_DIR)/cm3_fault_handlers.c
endif

ifeq ($(ERASE_FLASH), YES)
CDEFS += -DERASE_FLASH
endif

SRC += $(FLIGHTLIB)/paths.c
SRC += $(FLIGHTLIB)/fifo_buffer.c
SRC += $(FLIGHTLIB)/WorldMagModel.c
SRC += $(FLIGHTLIB)/insgps14state.c
SRC += $(FLIGHTLIB)/taskmonitor.c
SRC += $(FLIGHTLIB)/sanitycheck.c
SRC += $(MATHLIB)/coordinate_conversions.c
SRC += $(MATHLIB)/sin_lookup.c
SRC += $(MATHLIB)/misc_math.c
SRC += $(MATHLIB)/pid.c
SRC += $(MATHLIB)/atmospheric_math.c

## PIOS Hardware (STM32F4xx)
include $(PIOS)/STM32F4xx/library_fw.mk

## PIOS Hardware (Common)
SRC += $(PIOSCOMMON)/pios_delay.c
SRC += $(PIOSCOMMON)/pios_mpu6050.c
SRC += $(PIOSCOMMON)/pios_gcsrcvr.c
SRC += $(PIOSCOMMON)/pios_hmc5883.c
SRC += $(PIOSCOMMON)/pios_ms5611.c
SRC += $(PIOSCOMMON)/pios_etasv3.c
SRC += $(PIOSCOMMON)/pios_mpxv5004.c
SRC += $(PIOSCOMMON)/pios_mpxv7002.c
SRC += $(PIOSCOMMON)/pios_crc.c
SRC += $(PIOSCOMMON)/pios_com.c
SRC += $(PIOSCOMMON)/pios_rcvr.c
SRC += $(PIOSCOMMON)/pios_sbus.c
SRC += $(PIOSCOMMON)/pios_hsum.c
SRC += $(PIOSCOMMON)/pios_sensors.c
SRC += $(PIOSCOMMON)/pios_flash.c
SRC += $(PIOSCOMMON)/pios_flash_jedec.c
SRC += $(PIOSCOMMON)/pios_flashfs_logfs.c
SRC += $(PIOSCOMMON)/printf-stdarg.c
SRC += $(PIOSCOMMON)/pios_usb_desc_hid_cdc.c
SRC += $(PIOSCOMMON)/pios_usb_desc_hid_only.c
SRC += $(PIOSCOMMON)/pios_usb_util.c
SRC += $(PIOSCOMMON)/pios_adc.c
SRC += $(PIOSCOMMON)/pios_heap.c
SRC += $(PIOSCOMMON)/pios_semaphore.c
SRC += $(PIOSCOMMON)/pios_mutex.c
SRC += $(PIOSCOMMON)/pios_thread.c
SRC += $(PIOSCOMMON)/pios_queue.c


include ./UAVObjects.inc
SRC += $(UAVOBJSRC)

# List C++ source files here.
# use file-extension .cpp for C++-files (not .C)
CPPSRC =

# List any extra directories to look for include files here.
#    Each directory must be seperated by a space.
EXTRAINCDIRS += $(SHAREDAPIDIR)
EXTRAINCDIRS += $(PIOS)
EXTRAINCDIRS += $(PIOSINC)
EXTRAINCDIRS += $(OPUAVTALK)
EXTRAINCDIRS += $(OPUAVTALKINC)
EXTRAINCDIRS += $(OPUAVOBJ)
EXTRAINCDIRS += $(OPUAVOBJINC)
EXTRAINCDIRS += $(OPUAVSYNTHDIR)
EXTRAINCDIRS += $(FLIGHTLIBINC)
EXTRAINCDIRS += $(MATHLIBINC)
EXTRAINCDIRS += $(RSCODEINC)
EXTRAINCDIRS += $(PIOSSTM32F4XX)
EXTRAINCDIRS += $(PIOSCOMMON)
EXTRAINCDIRS += $(BOARD_INFO_DIR)
EXTRAINCDIRS += $(STMSPDINCDIR)
EXTRAINCDIRS += $(CMSISDIR)
EXTRAINCDIRS += $(BOOTINC)
EXTRAINCDIRS += $(DEBUG_CM3_DIR_INC)
EXTRAINCDIRS += $(MAVLINKINC)

EXTRAINCDIRS += ${foreach MOD, ${MODULES} ${OPTMODULES}, $(OPMODULEDIR)/${MOD}/inc} ${OPMODULEDIR}/System/inc

# List any extra directories to look for library files here.
# Also add directories where the linker should search for
# includes from linker-script to the list
#     Each directory must be seperated by a space.
EXTRA_LIBDIRS =

# Extra Libraries
#    Each library-name must be seperated by a space.
#    i.e. to link with libxyz.a, libabc.a and libefsl.a:
#    EXTRA_LIBS = xyz abc efsl
# for newlib-lpc (file: libnewlibc-lpc.a):
#    EXTRA_LIBS = newlib-lpc
EXTRA_LIBS =

# Path to Linker-Scripts
LINKERSCRIPTPATH = $(PIOSSTM32F4XX)

# Optimization level, can be [0, 1, 2, 3, s].
# 0 = turn off optimization. s = optimize for size.
# (Note: 3 is not always the best optimization level. See avr-libc FAQ.)

ifeq ($(DEBUG),YES)
CFLAGS += -O0
CFLAGS += -DGENERAL_COV
CFLAGS += -finstrument-functions -ffixed-r10
else
CFLAGS += -Os
endif


   
# common architecture-specific flags from the device-specific library makefile
CFLAGS += $(ARCHFLAGS)
CFLAGS += -DDIAGNOSTICS
CFLAGS += -DDIAG_TASKS

# configure CMSIS DSP Library
CDEFS += -DARM_MATH_CM4
CDEFS += -DARM_MATH_MATRIX_CHECK
CDEFS += -DARM_MATH_ROUNDING
CDEFS += -D__FPU_PRESENT=1
CDEFS += -DUNALIGNED_SUPPORT_DISABLE

# This is not the best place for these.  Really should abstract out
# to the board file or something
CFLAGS += -DSTM32F4XX

# Debugging format.
DEBUGF = dwarf-2

# Place project-specific -D (define) and/or
# -U options for C here.
CDEFS += -DHSE_VALUE=$(OSCILLATOR_FREQ)
CDEFS += -DSYSCLK_FREQ=$(SYSCLK_FREQ)
CDEFS += -DUSE_STDPERIPH_DRIVER
CDEFS += -DUSE_$(BOARD)
ifeq ($(ENABLE_DEBUG_CONSOLE), YES)
CDEFS += -DPIOS_INCLUDE_DEBUG_CONSOLE
endif

ifeq ($(ENABLE_DEBUG_CONSOLE), YES)
CDEFS += -DPIOS_INCLUDE_DEBUG_CONSOLE
endif


# Declare all non-optional modules as built-in to force inclusion, strip off any appended varieties of module
get_mod_name = $(shell echo $(1) | sed "s/\/[^\/]*$///")
BUILTIN_DEFS := ${foreach MOD, ${MODULES}, -DMODULE_$(call get_mod_name, $(MOD))_BUILTIN }
CDEFS += ${BUILTIN_DEFS}

# Place project-specific -D and/or -U options for
# Assembler with preprocessor here.
#ADEFS = -DUSE_IRQ_ASM_WRAPPER
ADEFS = -D__ASSEMBLY__

# Compiler flag to set the C Standard level.
# c89   - "ANSI" C
# gnu89 - c89 plus GCC extensions
# c99   - ISO C99 standard (not yet fully implemented)
# gnu99 - c99 plus GCC extensions
CSTANDARD = -std=gnu99

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

CFLAGS += -g$(DEBUGF)
CFLAGS += -ffast-math
CFLAGS += -mcpu=$(MCU)
CFLAGS += $(CDEFS)
CFLAGS += $(patsubst %,-I%,$(EXTRAINCDIRS)) -I.
CFLAGS += -mapcs-frame
CFLAGS += -fomit-frame-pointer
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -Wdouble-promotion

CFLAGS += -Wall
CFLAGS += -Werror
CFLAGS += -Wa,-adhlns=$(addprefix $(OUTDIR)/, $(notdir $(addsuffix .lst, $(basename $<))))
# Compiler flags to generate dependency files:
CFLAGS += -MD -MP -MF $(OUTDIR)/dep/$(@F).d

# flags only for C
#CONLYFLAGS += -Wnested-externs
CONLYFLAGS += $(CSTANDARD)

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
LDFLAGS += $(patsubst %,-L%,$(EXTRA_LIBDIRS))
LDFLAGS += $(patsubst %,-l%,$(EXTRA_LIBS))
LDFLAGS += -lc -lgcc -lm
LDFLAGS += -Wl,--warn-common
LDFLAGS += -Wl,--fatal-warnings

#Linker scripts                                                                                                                                                                                                              
LDFLAGS += $(addprefix -T,$(LINKER_SCRIPTS_APP))


# Define programs and commands.
REMOVE  = rm -f

# List of all source files.
ALLSRC     =  $(ASRC) $(SRC) $(CPPSRC)
# List of all source files without directory and file-extension.
ALLSRCBASE = $(notdir $(basename $(ALLSRC)))
# Define all object files.
ALLOBJ     = $(addprefix $(OUTDIR)/, $(addsuffix .o, $(ALLSRCBASE)))
# Define all listing files (used for make clean).
LSTFILES   = $(addprefix $(OUTDIR)/, $(addsuffix .lst, $(ALLSRCBASE)))
# Define all depedency-files (used for make clean).
DEPFILES   = $(addprefix $(OUTDIR)/dep/, $(addsuffix .o.d, $(ALLSRCBASE)))

# Default target.
all: gccversion build

build: elf hex bin lss sym

# Link: create ELF output file from object files.
$(eval $(call LINK_TEMPLATE, $(OUTDIR)/$(TARGET).elf, $(ALLOBJ)))

# Assemble: create object files from assembler source files.
$(foreach src, $(ASRC), $(eval $(call ASSEMBLE_TEMPLATE, $(src))))

# Compile: create object files from C source files.
$(foreach src, $(SRC), $(eval $(call COMPILE_C_TEMPLATE, $(src))))

# Compile: create object files from C++ source files.
$(foreach src, $(CPPSRC), $(eval $(call COMPILE_CPP_TEMPLATE, $(src))))

# Compile: create assembler files from C source files. ARM/Thumb
$(eval $(call PARTIAL_COMPILE_TEMPLATE, SRC))

$(OUTDIR)/$(TARGET).bin.o: $(OUTDIR)/$(TARGET).bin

$(eval $(call TLFW_TEMPLATE,$(OUTDIR)/$(TARGET).bin,$(BOARD_TYPE),$(BOARD_REVISION)))

# Add jtag targets (program and wipe)
$(eval $(call JTAG_TEMPLATE,$(OUTDIR)/$(TARGET).bin,$(FW_BANK_BASE),$(FW_BANK_SIZE),$(OPENOCD_JTAG_CONFIG),$(OPENOCD_CONFIG)))

.PHONY: elf lss sym hex bin bino tlfw
elf: $(OUTDIR)/$(TARGET).elf
lss: $(OUTDIR)/$(TARGET).lss
sym: $(OUTDIR)/$(TARGET).sym
hex: $(OUTDIR)/$(TARGET).hex
bin: $(OUTDIR)/$(TARGET).bin
bino: $(OUTDIR)/$(TARGET).bin.o
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

# Target: clean project.
clean: clean_list

clean_list :
	@echo $(MSG_CLEANING)
	$(V1) $(REMOVE) $(OUTDIR)/$(TARGET).map
	$(V1) $(REMOVE) $(OUTDIR)/$(TARGET).elf
	$(V1) $(REMOVE) $(OUTDIR)/$(TARGET).hex
	$(V1) $(REMOVE) $(OUTDIR)/$(TARGET).bin
	$(V1) $(REMOVE) $(OUTDIR)/$(TARGET).sym
	$(V1) $(REMOVE) $(OUTDIR)/$(TARGET).lss
	$(V1) $(REMOVE) $(OUTDIR)/$(TARGET).bin.o
	$(V1) $(REMOVE) $(ALLOBJ)
	$(V1) $(REMOVE) $(LSTFILES)
	$(V1) $(REMOVE) $(DEPFILES)
	$(V1) $(REMOVE) $(SRC:.c=.s)
	$(V1) $(REMOVE) $(CPPSRC:.cpp=.s)

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
-include $(shell mkdir $(OUTDIR) 2>/dev/null) $(shell mkdir $(OUTDIR)/dep 2>/dev/null) $(wildcard $(OUTDIR)/dep/*)
endif

# Listing of phony targets.
.PHONY : all build clean clean_list install docs
