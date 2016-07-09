
LQGLIB = $(FLIGHTLIB)/lqg_rate
LQGINC = $(FLIGHTLIB)/lqg_rate

ifeq ($(INCLUDE_RT_LQG), YES)
CDEFS += -DINCLUDE_LQG
SRC += $(LQGLIB)/rate_torque_kf.c
SRC += $(LQGLIB)/rate_torque_lqr.c
endif

ifeq ($(INCLUDE_RT_LQR_SOLVER), YES)
CDEFS += -DINCLUDE_LQG_SOLVER
CPPSRC = $(LQGLIB)/rate_torque_lqr_optimize.cpp
CPPSRC += $(LQGLIB)/dare.cpp
endif


# for F3 targets include a stripped
ifeq ($(MCU),cortex-m4)
SRC += $(LQGLIB)/rate_torque_si.c
else
ifeq ($(MCU), cortex-m3)
SRC += $(LQGLIB)/rate_torque_si_2d.c
endif
$(info $$MCU is [${MCU}])
endif

EXTRAINCDIRS  += $(LQGINC)
