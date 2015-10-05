###############################################################################
# @file       Makefile
# @author     The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2009.
# @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2015
# @addtogroup 
# @{
# @addtogroup 
# @{
# @brief Makefile to build firmware for the Naze family of boards.
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

# @startgroup Compile Options
# @groupbrief Set developer code and compile options
# @optbrief Set to YES to compile for debugging
DEBUG ?= NO

# @optbrief Include objects that are just nice information to show
STACK_DIAGNOSTICS ?= NO
MIXERSTATUS_DIAGNOSTICS ?= NO
RATEDESIRED_DIAGNOSTICS ?= NO
WDG_STATS_DIAGNOSTICS ?= NO
DIAG_TASKS ?= NO

#Or just turn on all the above diagnostics. WARNING: This consumes massive amounts of memory.
ALL_DIGNOSTICS ?=NO

# @optbrief Set to YES to build a FW version that will erase all flash memory
ERASE_FLASH ?= NO

# @optbrief Set to YES to use the Servo output pins for debugging via scope or logic analyser
ENABLE_DEBUG_PINS ?= NO

# @optbrief Set to Yes to enable the AUX UART which is mapped on the S1 (Tx) and S2 (Rx) servo outputs
ENABLE_AUX_UART ?= NO

# @endgroup Compile Options

# @startgroup  Optional modules
# @groupbrief Optional module and driver defaults
USE_CAMERASTAB ?= YES
USE_GPS ?= YES
USE_TXPID ?= YES
USE_ALTITUDE ?= NO
USE_AUTOTUNE ?= YES
USE_UAVOMAVLINKBRIDGE ?= YES
USE_UAVOLIGHTTELEMETRYBRIDGE ?= NO
USE_UAVOFRSKYSENSORHUBBRIDGE ?= YES
USE_BATTERY ?= NO
# @endgroup  Optional modules

# List of optional modules to include
OPTMODULES =
ifeq ($(USE_CAMERASTAB), YES)
OPTMODULES += CameraStab
endif
ifeq ($(USE_GPS), YES)
OPTMODULES += GPS
endif
ifeq ($(USE_TXPID), YES)
OPTMODULES += TxPID
endif
ifeq ($(USE_ALTITUDE), YES)
OPTMODULES += Altitude
endif
ifeq ($(USE_AUTOTUNE), YES)
OPTMODULES += Autotune
endif
ifeq ($(USE_UAVOMAVLINKBRIDGE), YES)
OPTMODULES += UAVOMavlinkBridge
endif
ifeq ($(USE_UAVOFRSKYSENSORHUBBRIDGE), YES)
OPTMODULES += UAVOFrSKYSensorHubBridge
endif
ifeq ($(USE_BATTERY), YES)
OPTMODULES += Battery
endif
# List of mandatory modules to include
MODULES += Attitude/smallf1
MODULES += Stabilization Actuator ManualControl FirmwareIAP
# Telemetry must be last to grab the optional modules (why?)
MODULES += Telemetry

# Paths
OPUAVTALKINC = $(OPUAVTALK)/inc
OPUAVOBJINC = $(OPUAVOBJ)/inc
OPTESTS  = ./Tests
FLIGHTLIBINC = $(FLIGHTLIB)/inc
STATEESTIMATIONLIB = $(FLIGHTLIB)/StateEstimationFilters
STATEESTIMATIONLIBINC = $(STATEESTIMATIONLIB)/inc
MATHLIB = $(FLIGHTLIB)/math
MATHLIBINC = $(FLIGHTLIB)/math
PIOSINC = $(PIOS)/inc
PIOSSTM32F10X = $(PIOS)/STM32F10x
PIOSCOMMON = $(PIOS)/Common
PIOSCOMMONLIB = $(PIOSCOMMON)/Libraries
APPLIBDIR = $(PIOSSTM32F10X)/Libraries
STMLIBDIR = $(APPLIBDIR)
STMSPDDIR = $(STMLIBDIR)/STM32F10x_StdPeriph_Driver
STMUSBDIR = $(STMLIBDIR)/STM32_USB-FS-Device_Driver
STMSPDSRCDIR = $(STMSPDDIR)/src
STMSPDINCDIR = $(STMSPDDIR)/inc
STMUSBSRCDIR = $(STMUSBDIR)/src
STMUSBINCDIR = $(STMUSBDIR)/inc
CMSISDIR  = $(STMLIBDIR)/CMSIS/Core/CM3
MSDDIR  = $(APPLIBDIR)/msd
RTOSDIR = $(PIOSCOMMON)/Libraries/FreeRTOS
RTOSSRCDIR = $(RTOSDIR)/Source
RTOSINCDIR = $(RTOSSRCDIR)/include
RTOSPORTDIR = $(APPLIBDIR)/FreeRTOS/Source
DOXYGENDIR = ../Doc/Doxygen
MAVLINKINC  = $(FLIGHTLIB)/mavlink/v1.0/common
PYMITE = $(FLIGHTLIB)/PyMite
PYMITELIB = $(PYMITE)/lib
PYMITEPLAT = $(PYMITE)/platform/openpilot
PYMITETOOLS = $(PYMITE)/tools
PYMITEVM = $(PYMITE)/vm
PYMITEINC = $(PYMITEVM)
PYMITEINC += $(PYMITEPLAT)
PYMITEINC += $(OUTDIR)
FLIGHTPLANLIB = $(OPMODULEDIR)/FlightPlan/lib
FLIGHTPLANS = $(OPMODULEDIR)/FlightPlan/flightplans

# List C source files here. (C dependencies are automatically generated.)
# use file-extension c for "c-only"-files

ifndef TESTAPP
## MODULES
SRC += ${foreach MOD, ${OPTMODULES}, ${wildcard ${OPMODULEDIR}/${MOD}/*.c}}
SRC += ${foreach MOD, ${MODULES}, ${wildcard ${OPMODULEDIR}/${MOD}/*.c}}
## OPENPILOT CORE:
SRC += ${OPMODULEDIR}/System/systemmod.c
SRC += main.c
SRC += pios_board.c
SRC += $(FLIGHTLIB)/alarms.c
SRC += $(OPUAVTALK)/uavtalk.c
SRC += $(OPUAVOBJ)/uavobjectmanager.c
SRC += $(OPUAVOBJ)/eventdispatcher.c
else
## TESTCODE
SRC += $(OPTESTS)/test_common.c
SRC += $(OPTESTS)/$(TESTAPP).c
endif


## PIOS Hardware (STM32F10x)
SRC += $(PIOSSTM32F10X)/pios_sys.c
SRC += $(PIOSSTM32F10X)/pios_led.c
SRC += $(PIOSSTM32F10X)/pios_usart.c
SRC += $(PIOSSTM32F10X)/pios_irq.c
#SRC += $(PIOSSTM32F10X)/pios_internal_adc.c
SRC += $(PIOSSTM32F10X)/pios_servo.c
#SRC += $(PIOSSTM32F10X)/pios_i2c.c
SRC += $(PIOSSTM32F10X)/pios_i2c_alt.c
SRC += $(PIOSSTM32F10X)/pios_spi.c
SRC += $(PIOSSTM32F10X)/pios_ppm.c
SRC += $(PIOSSTM32F10X)/pios_pwm.c
SRC += $(PIOSSTM32F10X)/pios_debug.c
SRC += $(PIOSSTM32F10X)/pios_gpio.c
SRC += $(PIOSSTM32F10X)/pios_exti.c
SRC += $(PIOSSTM32F10X)/pios_rtc.c
SRC += $(PIOSSTM32F10X)/pios_wdg.c
SRC += $(PIOSSTM32F10X)/pios_iap.c
SRC += $(PIOSSTM32F10X)/pios_tim.c
SRC += $(PIOSSTM32F10X)/pios_bl_helper.c
SRC += $(PIOSSTM32F10X)/pios_flash_internal.c

## PIOS Hardware (Common)
SRC += $(PIOSCOMMON)/pios_delay.c
SRC += $(PIOSCOMMON)/pios_crc.c
SRC += $(PIOSCOMMON)/pios_dsm.c
SRC += $(PIOSCOMMON)/pios_flashfs_logfs.c
SRC += $(PIOSCOMMON)/pios_flash_jedec.c
SRC += $(PIOSCOMMON)/pios_ms5611.c
SRC += $(PIOSCOMMON)/pios_mpu6050.c
SRC += $(PIOSCOMMON)/pios_com.c
SRC += $(PIOSCOMMON)/pios_rcvr.c
SRC += $(PIOSCOMMON)/pios_hsum.c
SRC += $(PIOSCOMMON)/pios_sensors.c
SRC += $(PIOSCOMMON)/pios_gcsrcvr.c
SRC += $(PIOSCOMMON)/printf-stdarg.c
#SRC += $(PIOSCOMMON)/pios_adc.c
SRC += $(PIOSCOMMON)/pios_flash.c
SRC += $(PIOSCOMMON)/pios_heap.c
SRC += $(PIOSCOMMON)/pios_semaphore.c
SRC += $(PIOSCOMMON)/pios_mutex.c
SRC += $(PIOSCOMMON)/pios_thread.c
SRC += $(PIOSCOMMON)/pios_queue.c
SRC += $(PIOSCOMMON)/pios_hal.c

SRC += $(PIOSCOMMON)/pios_board_info.c


## Libraries for flight calculations
SRC += $(FLIGHTLIB)/fifo_buffer.c
SRC += $(FLIGHTLIB)/taskmonitor.c
SRC += $(FLIGHTLIB)/sanitycheck.c
SRC += $(MATHLIB)/coordinate_conversions.c
SRC += $(MATHLIB)/misc_math.c
SRC += $(MATHLIB)/pid.c

## CMSIS for STM32
include $(PIOSCOMMONLIB)/CMSIS3/library.mk
#SRC += $(BOARD_INFO_DIR)/cmsis_system.c
SRC += $(BOARD_INFO_DIR)/system_stm32f10x.c

## Used parts of the STM-Library
SRC += $(STMSPDSRCDIR)/stm32f10x_adc.c
SRC += $(STMSPDSRCDIR)/stm32f10x_bkp.c
SRC += $(STMSPDSRCDIR)/stm32f10x_crc.c
SRC += $(STMSPDSRCDIR)/stm32f10x_dac.c
SRC += $(STMSPDSRCDIR)/stm32f10x_dma.c
SRC += $(STMSPDSRCDIR)/stm32f10x_exti.c
SRC += $(STMSPDSRCDIR)/stm32f10x_flash.c
SRC += $(STMSPDSRCDIR)/stm32f10x_gpio.c
SRC += $(STMSPDSRCDIR)/stm32f10x_i2c.c
SRC += $(STMSPDSRCDIR)/stm32f10x_pwr.c
SRC += $(STMSPDSRCDIR)/stm32f10x_rcc.c
SRC += $(STMSPDSRCDIR)/stm32f10x_rtc.c
SRC += $(STMSPDSRCDIR)/stm32f10x_spi.c
SRC += $(STMSPDSRCDIR)/stm32f10x_tim.c
SRC += $(STMSPDSRCDIR)/stm32f10x_usart.c
SRC += $(STMSPDSRCDIR)/stm32f10x_iwdg.c
SRC += $(STMSPDSRCDIR)/stm32f10x_dbgmcu.c
SRC += $(STMSPDSRCDIR)/misc.c

## RTOS
SRC += $(RTOSSRCDIR)/list.c
SRC += $(RTOSSRCDIR)/queue.c
SRC += $(RTOSSRCDIR)/tasks.c

## RTOS Portable
SRC += $(RTOSPORTDIR)/portable/GCC/ARM_CM3/port.c

## PyMite files
#SRC += $(wildcard ${PYMITEVM}/*.c)
#SRC += $(wildcard ${PYMITEPLAT}/*.c)
#SRC += $(OUTDIR)/pmlib_img.c
#SRC += $(OUTDIR)/pmlib_nat.c
#SRC += $(OUTDIR)/pmlibusr_img.c
#SRC += $(OUTDIR)/pmlibusr_nat.c

## Mass Storage Device
#SRC += $(MSDDIR)/msd.c
#SRC += $(MSDDIR)/msd_bot.c
#SRC += $(MSDDIR)/msd_desc.c
#SRC += $(MSDDIR)/msd_memory.c
#SRC += $(MSDDIR)/msd_scsi.c
#SRC += $(MSDDIR)/msd_scsi_data.c

# List C source files here which must be compiled in ARM-Mode (no -mthumb).
# use file-extension c for "c-only"-files
## just for testing, timer.c could be compiled in thumb-mode too
SRCARM =

# List C++ source files here.
# use file-extension .cpp for C++-files (not .C)
CPPSRC =

# List C++ source files here which must be compiled in ARM-Mode.
# use file-extension .cpp for C++-files (not .C)
#CPPSRCARM = $(TARGET).cpp
CPPSRCARM =

# List Assembler source files here.
# Make them always end in a capital .S. Files ending in a lowercase .s
# will not be considered source files but generated files (assembler
# output from the compiler), and will be deleted upon "make clean"!
# Even though the DOS/Win* filesystem matches both .s and .S the same,
# it will preserve the spelling of the filenames, and gcc itself does
# care about how the name is spelled on its command-line.
ASRC = $(PIOSSTM32F10X)/startup_stm32f10x_$(MODEL)$(MODEL_SUFFIX).S

# List Assembler source files here which must be assembled in ARM-Mode..
ASRCARM =

# List any extra directories to look for include files here.
#    Each directory must be seperated by a space.
EXTRAINCDIRS  += $(SHAREDAPIDIR)
EXTRAINCDIRS  += $(OPUAVTALK)
EXTRAINCDIRS  += $(OPUAVTALKINC)
EXTRAINCDIRS  += $(OPUAVOBJ)
EXTRAINCDIRS  += $(OPUAVOBJINC)
EXTRAINCDIRS  += $(OPUAVSYNTHDIR)
EXTRAINCDIRS  += $(PIOS)
EXTRAINCDIRS  += $(PIOSINC)
EXTRAINCDIRS  += $(FLIGHTLIBINC)
EXTRAINCDIRS  += $(STATEESTIMATIONLIBINC)
EXTRAINCDIRS  += $(MATHLIBINC)
EXTRAINCDIRS  += $(PIOSSTM32F10X)
EXTRAINCDIRS  += $(PIOSSTM32F10X)/inc
EXTRAINCDIRS  += $(PIOSCOMMON)
EXTRAINCDIRS  += $(BOARD_INFO_DIR)
EXTRAINCDIRS  += $(STMSPDINCDIR)
EXTRAINCDIRS  += $(STMUSBINCDIR)
EXTRAINCDIRS  += $(CMSISDIR)
EXTRAINCDIRS  += $(MSDDIR)
EXTRAINCDIRS  += $(RTOSINCDIR)
EXTRAINCDIRS  += $(APPLIBDIR)
EXTRAINCDIRS  += $(RTOSPORTDIR)/portable/GCC/ARM_CM3
EXTRAINCDIRS  += $(AHRSBOOTLOADERINC)
EXTRAINCDIRS  += $(PYMITEINC)
EXTRAINCDIRS  += $(MAVLINKINC)

EXTRAINCDIRS += ${foreach MOD, ${OPTMODULES} ${MODULES}, ${OPMODULEDIR}/${MOD}/inc} ${OPMODULEDIR}/System/inc


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
LINKERSCRIPTPATH = ..

# Optimization level, can be [0, 1, 2, 3, s].
# 0 = turn off optimization. s = optimize for size.
# (Note: 3 is not always the best optimization level. See avr-libc FAQ.)

ifeq ($(DEBUG),YES)
OPT = 1
else
OPT = s
endif

# Output format. (can be ihex or binary or both)
#  binary to create a load-image in raw-binary format i.e. for SAM-BA,
#  ihex to create a load-image in Intel hex format
#LOADFORMAT = ihex
#LOADFORMAT = binary
LOADFORMAT = both

# Debugging format.
DEBUGF = dwarf-2

# Place project-specific -D (define) and/or
# -U options for C here.
CDEFS = -DSTM32F10X_$(MODEL)
CDEFS += -DUSE_STDPERIPH_DRIVER
CDEFS += -DUSE_$(BOARD)
ifeq ($(ENABLE_DEBUG_PINS), YES)
CDEFS += -DPIOS_ENABLE_DEBUG_PINS
endif
ifeq ($(ENABLE_DEBUG_CONSOLE), YES)
CDEFS += -DPIOS_INCLUDE_DEBUG_CONSOLE
endif
ifeq ($(ERASE_FLASH), YES)
CDEFS += -DERASE_FLASH
endif

ifneq ($(USE_GPS), NO)
CDEFS += -DUSE_GPS
endif

ifeq ($(USE_I2C), YES)
CDEFS += -DUSE_I2C
endif

ifeq ($(USE_PCF8591), YES)
CDEFS += -DUSE_PCF8591
endif

# Declare all non-optional modules as built-in to force inclusion, strip off any appended varieties of module
get_mod_name = $(shell echo $(1) | sed "s/\/[^\/]*$///")
BUILTIN_DEFS := ${foreach MOD, ${MODULES}, -DMODULE_$(call get_mod_name, $(MOD))_BUILTIN }
CDEFS += ${BUILTIN_DEFS}

# Provide board-specific defines
CDEFS += -DBOARD_TYPE=$(BOARD_TYPE)
CDEFS += -DBOARD_REVISION=$(BOARD_REVISION)
CDEFS += -DHW_TYPE=$(HW_TYPE)
CDEFS += -DBOOTLOADER_VERSION=$(BOOTLOADER_VERSION)
CDEFS += -DFW_BANK_BASE=$(FW_BANK_BASE)
CDEFS += -DFW_BANK_SIZE=$(FW_BANK_SIZE)
CDEFS += -DFW_DESC_SIZE=$(FW_DESC_SIZE)
CDEFS += -DEE_BANK_BASE=$(EE_BANK_BASE)
CDEFS += -DEE_BANK_SIZE=$(EE_BANK_SIZE)

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

ifeq ($(DEBUG),YES)
CFLAGS += -DDEBUG
endif

#The following Makefile command, ifneq (, $(filter) $(A), $(B) $(C)) is equivalent 
# to the pseudocode `if(A== B || A==C)`
ifneq (,$(filter YES,$(STACK_DIAGNOSTICS) $(ALL_DIGNOSTICS)))
CFLAGS += -DSTACK_DIAGNOSTICS
endif

ifneq (,$(filter YES,$(MIXERSTATUS_DIAGNOSTICS) $(ALL_DIGNOSTICS)))
CFLAGS += -DMIXERSTATUS_DIAGNOSTICS
endif

ifneq (,$(filter YES,$(RATEDESIRED_DIAGNOSTICS) $(ALL_DIGNOSTICS)))
CFLAGS += -DRATEDESIRED_DIAGNOSTICS
endif

ifneq (,$(filter YES,$(WDG_STATS_DIAGNOSTICS) $(ALL_DIGNOSTICS)))
CFLAGS += -DWDG_STATS_DIAGNOSTICS
endif

ifneq (,$(filter YES,$(DIAG_TASKS) $(ALL_DIGNOSTICS)))
CFLAGS += -DDIAG_TASKS
endif

CFLAGS += -g$(DEBUGF)
CFLAGS += -O$(OPT)
CFLAGS += -mcpu=$(MCU)
CFLAGS += $(CDEFS)
CFLAGS += $(patsubst %,-I%,$(EXTRAINCDIRS)) -I.

#CFLAGS += -fno-cprop-registers -fno-defer-pop -fno-guess-branch-probability -fno-section-anchors
#CFLAGS += -fno-if-conversion -fno-if-conversion2 -fno-ipa-pure-const -fno-ipa-reference -fno-merge-constants
#CFLAGS += -fno-split-wide-types -fno-tree-ccp -fno-tree-ch -fno-tree-copy-prop -fno-tree-copyrename
#CFLAGS += -fno-tree-dce -fno-tree-dominator-opts -fno-tree-dse -fno-tree-fre -fno-tree-sink -fno-tree-sra
#CFLAGS += -fno-tree-ter
#CFLAGS += -g$(DEBUGF) -DDEBUG

CFLAGS += -mapcs-frame
CFLAGS += -fomit-frame-pointer

CFLAGS += -Wall
CFLAGS += -Werror
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -Wdouble-promotion
CFLAGS += -Wa,-adhlns=$(addprefix $(OUTDIR)/, $(notdir $(addsuffix .lst, $(basename $<))))
# Compiler flags to generate dependency files:
CFLAGS += -MD -MP -MF $(OUTDIR)/dep/$(@F).d

# flags only for C
#CONLYFLAGS += -Wnested-externs
CONLYFLAGS += $(CSTANDARD)

# Assembler flags.
#  -Wa,...:    tell GCC to pass this to the assembler.
#  -ahlns:     create listing
ASFLAGS  = -mcpu=$(MCU) -I. -x assembler-with-cpp
ASFLAGS += $(ADEFS)
ASFLAGS += -Wa,-adhlns=$(addprefix $(OUTDIR)/, $(notdir $(addsuffix .lst, $(basename $<))))
ASFLAGS += $(patsubst %,-I%,$(EXTRAINCDIRS))

MATH_LIB = -lm

LIBS += $(UAVOBJLIB)

# Linker flags.
#  -Wl,...:     tell GCC to pass this to linker.
#    -Map:      create map file
#    --cref:    add cross reference to  map file
LDFLAGS = -nostartfiles -Wl,-Map=$(OUTDIR)/$(TARGET).map,--cref,--gc-sections
LDFLAGS += $(patsubst %,-L%,$(EXTRA_LIBDIRS))
LDFLAGS += -lc
LDFLAGS += $(patsubst %,-l%,$(EXTRA_LIBS))
LDFLAGS += $(MATH_LIB)
LDFLAGS += -lc -lgcc
LDFLAGS += -Wl,--warn-common
LDFLAGS += -Wl,--fatal-warnings

# Set linker-script name depending on selected submodel name
LDFLAGS += -T$(LINKERSCRIPTPATH)/link_$(BOARD)_memory.ld
LDFLAGS += -T$(LINKERSCRIPTPATH)/link_$(BOARD)_sections.ld

include ./UAVObjects.inc
include $(MAKE_INC_DIR)/firmware-common.mk
