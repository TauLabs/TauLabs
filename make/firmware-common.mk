ifneq ($(NO_AUTO_UAVO),YES)

# Common UAVOs to all targets:

UAVOBJSRCFILENAMES += accels
UAVOBJSRCFILENAMES += firmwareiapobj
UAVOBJSRCFILENAMES += flightstatus
UAVOBJSRCFILENAMES += flighttelemetrystats
UAVOBJSRCFILENAMES += gcsreceiver
UAVOBJSRCFILENAMES += gcstelemetrystats
UAVOBJSRCFILENAMES += modulesettings
UAVOBJSRCFILENAMES += objectpersistence
UAVOBJSRCFILENAMES += receiveractivity
UAVOBJSRCFILENAMES += sessionmanaging
UAVOBJSRCFILENAMES += systemalarms
UAVOBJSRCFILENAMES += systemsettings
UAVOBJSRCFILENAMES += systemstats
UAVOBJSRCFILENAMES += taskinfo
UAVOBJSRCFILENAMES += watchdogstatus

ifneq ($(UAVO_MINIMAL),YES)
# On all non-discovery board targets:

UAVOBJSRCFILENAMES += accessorydesired
UAVOBJSRCFILENAMES += actuatorcommand
UAVOBJSRCFILENAMES += actuatordesired
UAVOBJSRCFILENAMES += actuatorsettings
UAVOBJSRCFILENAMES += airspeedactual
UAVOBJSRCFILENAMES += attitudeactual
UAVOBJSRCFILENAMES += attitudesettings
UAVOBJSRCFILENAMES += baroaltitude
UAVOBJSRCFILENAMES += cameradesired
UAVOBJSRCFILENAMES += camerastabsettings
UAVOBJSRCFILENAMES += fixedwingairspeeds
UAVOBJSRCFILENAMES += fixedwingpathfollowerstatus
UAVOBJSRCFILENAMES += flightbatterysettings
UAVOBJSRCFILENAMES += flightbatterystate
UAVOBJSRCFILENAMES += gpsposition
UAVOBJSRCFILENAMES += gpsvelocity
UAVOBJSRCFILENAMES += gyros
UAVOBJSRCFILENAMES += homelocation
UAVOBJSRCFILENAMES += manualcontrolcommand
UAVOBJSRCFILENAMES += manualcontrolsettings
UAVOBJSRCFILENAMES += mixersettings
UAVOBJSRCFILENAMES += mixerstatus
UAVOBJSRCFILENAMES += mwratesettings
UAVOBJSRCFILENAMES += pathdesired
UAVOBJSRCFILENAMES += pathstatus
UAVOBJSRCFILENAMES += poilocation
UAVOBJSRCFILENAMES += positionactual
UAVOBJSRCFILENAMES += ratedesired
UAVOBJSRCFILENAMES += sensorsettings
UAVOBJSRCFILENAMES += stabilizationdesired
UAVOBJSRCFILENAMES += stabilizationsettings
UAVOBJSRCFILENAMES += systemident
UAVOBJSRCFILENAMES += tabletinfo
UAVOBJSRCFILENAMES += trimangles
UAVOBJSRCFILENAMES += trimanglessettings
UAVOBJSRCFILENAMES += txpidsettings
UAVOBJSRCFILENAMES += ubloxinfo
UAVOBJSRCFILENAMES += velocityactual

ifeq ($(UAVO_NAV),YES)
UAVOBJSRCFILENAMES += airspeedsettings
UAVOBJSRCFILENAMES += altitudeholddesired
UAVOBJSRCFILENAMES += altitudeholdsettings
UAVOBJSRCFILENAMES += baroairspeed
UAVOBJSRCFILENAMES += fixedwingpathfollowersettings
UAVOBJSRCFILENAMES += flightplancontrol
UAVOBJSRCFILENAMES += flightplansettings
UAVOBJSRCFILENAMES += flightplanstatus
UAVOBJSRCFILENAMES += gpssatellites
UAVOBJSRCFILENAMES += gpstime
UAVOBJSRCFILENAMES += gyrosbias
UAVOBJSRCFILENAMES += inssettings
UAVOBJSRCFILENAMES += insstate
UAVOBJSRCFILENAMES += loitercommand
UAVOBJSRCFILENAMES += magbias
UAVOBJSRCFILENAMES += magnetometer
UAVOBJSRCFILENAMES += nedaccel
UAVOBJSRCFILENAMES += nedposition
UAVOBJSRCFILENAMES += pathplannersettings
UAVOBJSRCFILENAMES += sonaraltitude
UAVOBJSRCFILENAMES += stateestimation
UAVOBJSRCFILENAMES += velocitydesired
UAVOBJSRCFILENAMES += vibrationanalysisoutput
UAVOBJSRCFILENAMES += vibrationanalysissettings
UAVOBJSRCFILENAMES += vtolpathfollowersettings
UAVOBJSRCFILENAMES += vtolpathfollowerstatus
UAVOBJSRCFILENAMES += waypoint
UAVOBJSRCFILENAMES += waypointactive
endif # UAVO_NAV

endif # !UAVO_MINIMAL

endif # !NO_AUTO_UAVO

# Define programs and commands.
REMOVE  = rm -f

SRC += $(foreach UAVOBJSRCFILE,$(UAVOBJSRCFILENAMES),$(OPUAVSYNTHDIR)/$(UAVOBJSRCFILE).c )

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

build: elf

ifneq ($(BUILD_FWFILES), NO)
build: hex bin lss sym
endif

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
