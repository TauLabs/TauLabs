 #####
 # Project: OpenPilot
 #
 #
 # Makefile for OpenPilot project build PiOS and the AP.
 #
 # The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2009.
 # Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
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

include $(MAKE_INC_DIR)/firmware-defs.mk
include $(BOARD_INFO_DIR)/board-info.mk

# Set developer code and compile options
# Set to YES to compile for debugging
DEBUG ?= NO

# Include objects that are just nice information to show
STACK_DIAGNOSTICS ?= NO
MIXERSTATUS_DIAGNOSTICS ?= NO
RATEDESIRED_DIAGNOSTICS ?= NO
WDG_STATS_DIAGNOSTICS ?= NO
DIAG_TASKS ?= NO

#Or just turn on all the above diagnostics. WARNING: This consumes massive amounts of memory.
ALL_DIGNOSTICS ?=NO

# Set to YES to build a FW version that will erase all flash memory
ERASE_FLASH ?= NO
# Set to YES to use the Servo output pins for debugging via scope or logic analyser
ENABLE_DEBUG_PINS ?= NO

# Set to Yes to enable the AUX UART which is mapped on the S1 (Tx) and S2 (Rx) servo outputs
ENABLE_AUX_UART ?= NO

USE_GPS ?= NO

USE_I2C ?= YES

# List of modules to include
MODULES += ComUsbBridge
MODULES += RadioComBridge
MODULES += UAVOTaranis

# Paths
OPUAVTALKINC = $(OPUAVTALK)/inc
OPUAVOBJINC = $(OPUAVOBJ)/inc
OPTESTS  = ./Tests
FLIGHTLIBINC = $(FLIGHTLIB)/inc
RSCODEINC = $(FLIGHTLIB)/rscode
PIOSINC = $(PIOS)/inc
PIOSSTM32F10X = $(PIOS)/STM32F10x
PIOSCOMMON = $(PIOS)/Common
PIOSCOMMONLIB = $(PIOSCOMMON)/Libraries
MATHLIB = $(FLIGHTLIB)/math
MATHLIBINC = $(FLIGHTLIB)/math
APPLIBDIR = $(PIOSSTM32F10X)/Libraries
STMLIBDIR = $(APPLIBDIR)
STMSPDDIR = $(STMLIBDIR)/STM32F10x_StdPeriph_Driver
STMUSBDIR = $(STMLIBDIR)/STM32_USB-FS-Device_Driver
STMSPDSRCDIR = $(STMSPDDIR)/src
STMSPDINCDIR = $(STMSPDDIR)/inc
STMUSBSRCDIR = $(STMUSBDIR)/src
STMUSBINCDIR = $(STMUSBDIR)/inc
CMSISDIR  = $(STMLIBDIR)/CMSIS/Core/CM3
DOSFSDIR  = $(APPLIBDIR)/dosfs
MSDDIR  = $(APPLIBDIR)/msd
RTOSDIR = $(PIOSCOMMON)/Libraries/FreeRTOS
RTOSSRCDIR = $(RTOSDIR)/Source
RTOSINCDIR = $(RTOSSRCDIR)/include
RTOSPORTDIR = $(APPLIBDIR)/FreeRTOS/Source

# List C source files here. (C dependencies are automatically generated.)
# use file-extension c for "c-only"-files

ifndef TESTAPP
## MODULES
SRC += ${foreach MOD, ${MODULES} ${OPTMODULES}, ${wildcard ${OPMODULEDIR}/${MOD}/*.c}}
## OPENPILOT CORE:
SRC += ${OPMODULEDIR}/PipXtreme/pipxtrememod.c
SRC += main.c
SRC += pios_board.c
SRC += $(OPUAVTALK)/uavtalk.c
SRC += $(OPUAVOBJ)/uavobjectmanager.c
SRC += $(OPUAVOBJ)/eventdispatcher.c
else
## TESTCODE
SRC += $(OPTESTS)/test_common.c
SRC += $(OPTESTS)/$(TESTAPP).c
endif

## UAVOBJECTS
ifndef TESTAPP
SRC += $(OPUAVSYNTHDIR)/gcsreceiver.c
SRC += $(OPUAVSYNTHDIR)/hwtaulink.c
SRC += $(OPUAVSYNTHDIR)/rfm22bstatus.c
SRC += $(OPUAVSYNTHDIR)/objectpersistence.c
SRC += $(OPUAVSYNTHDIR)/modulesettings.c
SRC += $(OPUAVSYNTHDIR)/rfm22breceiver.c
SRC += $(OPUAVSYNTHDIR)/radiocombridgestats.c
SRC += $(OPUAVSYNTHDIR)/flightbatterystate.c
endif

## PIOS Hardware (STM32F10x)
SRC += $(PIOSSTM32F10X)/pios_sys.c
SRC += $(PIOSSTM32F10X)/pios_led.c
SRC += $(PIOSSTM32F10X)/pios_usart.c
SRC += $(PIOSSTM32F10X)/pios_irq.c
SRC += $(PIOSSTM32F10X)/pios_flash_internal.c
SRC += $(PIOSSTM32F10X)/pios_spi.c
SRC += $(PIOSSTM32F10X)/pios_ppm.c
SRC += $(PIOSSTM32F10X)/pios_debug.c
SRC += $(PIOSSTM32F10X)/pios_gpio.c
SRC += $(PIOSSTM32F10X)/pios_exti.c
SRC += $(PIOSSTM32F10X)/pios_rtc.c
SRC += $(PIOSSTM32F10X)/pios_wdg.c
SRC += $(PIOSSTM32F10X)/pios_tim.c
SRC += $(PIOSSTM32F10X)/pios_pwm.c
SRC += $(PIOSSTM32F10X)/pios_bl_helper.c

# PIOS USB related files (seperated to make code maintenance more easy)
SRC += $(PIOSSTM32F10X)/pios_usb.c
SRC += $(PIOSSTM32F10X)/pios_usbhook.c
SRC += $(PIOSSTM32F10X)/pios_usb_hid.c
SRC += $(PIOSSTM32F10X)/pios_usb_cdc.c
SRC += $(PIOSSTM32F10X)/pios_usb_hid_istr.c
SRC += $(PIOSSTM32F10X)/pios_usb_hid_pwr.c
SRC += pios_usb_board_data.c
SRC += $(PIOSCOMMON)/pios_usb_desc_hid_cdc.c
SRC += $(PIOSCOMMON)/pios_usb_desc_hid_only.c
SRC += $(PIOSCOMMON)/pios_usb_util.c

## PIOS Hardware (Common)
SRC += $(PIOSCOMMON)/pios_delay.c
SRC += $(PIOSCOMMON)/pios_crc.c
SRC += $(PIOSCOMMON)/pios_com.c
SRC += $(PIOSCOMMON)/pios_i2c_esc.c
SRC += $(PIOSCOMMON)/pios_rcvr.c
SRC += $(PIOSCOMMON)/pios_flash.c
SRC += $(PIOSCOMMON)/pios_flashfs_logfs.c
SRC += $(PIOSCOMMON)/printf-stdarg.c
SRC += $(PIOSCOMMON)/pios_rfm22b.c
SRC += $(PIOSCOMMON)/pios_rfm22b_com.c
SRC += $(PIOSCOMMON)/pios_heap.c
SRC += $(PIOSCOMMON)/pios_semaphore.c
SRC += $(PIOSCOMMON)/pios_mutex.c
SRC += $(PIOSCOMMON)/pios_thread.c
SRC += $(PIOSCOMMON)/pios_queue.c


## Libraries for flight calculations
SRC += $(FLIGHTLIB)/fifo_buffer.c
SRC += $(FLIGHTLIB)/taskmonitor.c
SRC += $(FLIGHTLIB)/aes.c
## The Reed-Solomon FEC library
SRC += $(FLIGHTLIB)/rscode/rs.c
SRC += $(FLIGHTLIB)/rscode/berlekamp.c
SRC += $(FLIGHTLIB)/rscode/galois.c
SRC += $(MATHLIB)/misc_math.c

## CMSIS for STM32
include $(PIOSCOMMONLIB)/CMSIS3/library.mk
SRC += $(BOARD_INFO_DIR)/cmsis_system.c

## Used parts of the STM-Library
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

## STM32 USB Library
SRC += $(STMUSBSRCDIR)/usb_core.c
SRC += $(STMUSBSRCDIR)/usb_init.c
SRC += $(STMUSBSRCDIR)/usb_int.c
SRC += $(STMUSBSRCDIR)/usb_mem.c
SRC += $(STMUSBSRCDIR)/usb_regs.c
SRC += $(STMUSBSRCDIR)/usb_sil.c

## RTOS
SRC += $(RTOSSRCDIR)/list.c
SRC += $(RTOSSRCDIR)/queue.c
SRC += $(RTOSSRCDIR)/tasks.c

## RTOS Portable
SRC += $(RTOSPORTDIR)/portable/GCC/ARM_CM3/port.c

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
EXTRAINCDIRS  += $(MATHLIBINC)
EXTRAINCDIRS  += $(FLIGHTLIBINC)
EXTRAINCDIRS  += $(RSCODEINC)
EXTRAINCDIRS  += $(PIOSSTM32F10X)
EXTRAINCDIRS  += $(PIOSSTM32F10X)/inc
EXTRAINCDIRS  += $(PIOSCOMMON)
EXTRAINCDIRS  += $(BOARD_INFO_DIR)
EXTRAINCDIRS  += $(STMSPDINCDIR)
EXTRAINCDIRS  += $(STMUSBINCDIR)
EXTRAINCDIRS  += $(CMSISDIR)
EXTRAINCDIRS  += $(DOSFSDIR)
EXTRAINCDIRS  += $(MSDDIR)
EXTRAINCDIRS  += $(RTOSINCDIR)
EXTRAINCDIRS  += $(APPLIBDIR)
EXTRAINCDIRS  += $(RTOSPORTDIR)/portable/GCC/ARM_CM3

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
LINKERSCRIPTPATH = $(PIOSSTM32F10X)

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
ifeq ($(ENABLE_AUX_UART), YES)
CDEFS += -DPIOS_ENABLE_AUX_UART
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

CDEFS += -DEE_BANK_BASE=$(EE_BANK_BASE)
CDEFS += -DEE_BANK_SIZE=$(EE_BANK_SIZE)

# Declare all non-optional modules as built-in to force inclusion
CDEFS += $(foreach MOD, $(notdir $(MODULES)), -DMODULE_$(MOD)_BUILTIN)

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
CFLAGS = -DDEBUG
endif

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
CFLAGS += -ffunction-sections -fdata-sections

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
ASFLAGS  = -mcpu=$(MCU) -I. -x assembler-with-cpp
ASFLAGS += $(ADEFS)
ASFLAGS += -Wa,-adhlns=$(addprefix $(OUTDIR)/, $(notdir $(addsuffix .lst, $(basename $<))))
ASFLAGS += $(patsubst %,-I%,$(EXTRAINCDIRS))

MATH_LIB = -lm

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

include $(MAKE_INC_DIR)/firmware-common.mk

