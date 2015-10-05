uavobjects_armsoftfp: FLOATABI = -mfloat-abi=soft
uavobjects_armsoftfp: $(OUTDIR)/libuavobject.a
uavobjects_armhardfp: FLOATABI = -mfloat-abi=hard
uavobjects_armhardfp: $(OUTDIR)/libuavobject.a

include $(MAKE_INC_DIR)/firmware-defs.mk

OPUAVTALKINC = $(OPUAVTALK)/inc
OPUAVOBJINC = $(OPUAVOBJ)/inc
FLIGHTLIBINC = $(FLIGHTLIB)/inc
PIOSINC = $(PIOS)/inc
PIOSCOMMON = $(PIOS)/Common
PIOSCOMMONLIB = $(PIOSCOMMON)/Libraries
STMLIBDIR = $(APPLIBDIR)
STMSPDSRCDIR = $(STMSPDDIR)/src
STMSPDINCDIR = $(STMSPDDIR)/inc
STMUSBSRCDIR = $(STMUSBDIR)/src
STMUSBINCDIR = $(STMUSBDIR)/inc
MSDDIR  = $(APPLIBDIR)/msd
DOXYGENDIR = ../Doc/Doxygen

# List C source files here. (C dependencies are automatically generated.)
# use file-extension c for "c-only"-files

EXTRAINCDIRS  += $(OPUAVTALK)
EXTRAINCDIRS  += $(OPUAVTALKINC)
EXTRAINCDIRS  += $(OPUAVOBJ)
EXTRAINCDIRS  += $(OPUAVOBJINC)
EXTRAINCDIRS  += $(OPUAVSYNTHDIR)
EXTRAINCDIRS  += $(PIOS)
EXTRAINCDIRS  += $(PIOSINC)
EXTRAINCDIRS  += $(PIOSCOMMON)
EXTRAINCDIRS  += $(APPLIBDIR)

CFLAGS += -g3
CFLAGS += -std=gnu99
CFLAGS += -Os
CFLAGS += -mcpu=cortex-m3
CFLAGS += $(FLOATABI)

CFLAGS += $(CDEFS)
CFLAGS += $(patsubst %,-I%,$(EXTRAINCDIRS)) -I.

CFLAGS += -fomit-frame-pointer

CFLAGS += -Wall
CFLAGS += -Werror
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -Wdouble-promotion
CFLAGS += -Wa,-adhlns=$(addprefix $(OUTDIR)/, $(notdir $(addsuffix .lst, $(basename $<))))
# Compiler flags to generate dependency files:
CFLAGS += -MD -MP -MF $(OUTDIR)/dep/$(@F).d

# Linker flags.
#  -Wl,...:     tell GCC to pass this to linker.
#    -Map:      create map file
#    --cref:    add cross reference to  map file
LDFLAGS = -nostartfiles -Wl,-Map=$(OUTDIR)/$(TARGET).map,--cref,--gc-sections
LDFLAGS += $(patsubst %,-L%,$(EXTRA_LIBDIRS))
LDFLAGS += -lc
LDFLAGS += $(patsubst %,-l%,$(EXTRA_LIBS))
LDFLAGS += $(MATH_LIB)
LDFLAGS += -Wl,--warn-common
LDFLAGS += -Wl,--fatal-warnings

SRC += $(filter-out $(OPUAVSYNTHDIR)/uavobjectsinit.c,$(wildcard $(OPUAVSYNTHDIR)/*.c))
NO_AUTO_UAVO = YES

include $(MAKE_INC_DIR)/firmware-common.mk

$(OUTDIR)/libuavobject.a: $(ALLOBJ)
	$(AR) rvs $@ $?

