
LQGLIB = $(FLIGHTLIB)/lqg_rate
LQGINC = $(FLIGHTLIB)/lqg_rate

ifeq ($(INCLUDE_RT_LQG), YES)
CDEFS += -DINCLUDE_LQG
SRC += $(LQGLIB)/rate_torque_kf.c
SRC += $(LQGLIB)/rate_torque_lqr.c
endif

SRC += $(LQGLIB)/rate_torque_si.c

EXTRAINCDIRS  += $(LQGINC)
