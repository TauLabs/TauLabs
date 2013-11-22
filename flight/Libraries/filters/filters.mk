#
# Rules to add the filter architecture
#

FILTERLIBINC = $(FILTERLIB)
EXTRAINCDIRS += $(FILTERLIBINC)

SRC += $(FILTERLIB)/cfnav_interface.c
SRC += $(FILTERLIB)/cf_interface.c
SRC += $(FILTERLIB)/filter_infrastructure_se3.c
SRC += $(FILTERLIB)/filter_interface.c
