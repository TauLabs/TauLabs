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
