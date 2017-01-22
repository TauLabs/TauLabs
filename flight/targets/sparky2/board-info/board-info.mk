BOARD_TYPE          := 0x92
BOARD_REVISION      := 0x01
BOOTLOADER_VERSION  := 0x84
HW_TYPE             := 0x00

MCU                 := cortex-m4
CHIP                := STM32F405RGT
STM32_TYPE          := STM32F40_41xxx
BOARD               := STM32F4xx_RM
MODEL               := HD
MODEL_SUFFIX        := 

OPENOCD_JTAG_CONFIG ?= stlink-v2.cfg
OPENOCD_CONFIG      := stm32f4x.cfg

# Note: These must match the values in link_$(BOARD)_memory.ld
BL_BANK_BASE        := 0x08000000  # Start of bootloader flash
BL_BANK_SIZE        := 0x00008000  # Should include BD_INFO region (32kb)

# Leave the remaining 16KB and 64KB sectors for other uses

FW_BANK_BASE        := 0x08020000  # Start of firmware flash
FW_BANK_SIZE        := 0x00060000  # Should include FW_DESC_SIZE (256kb)

FW_DESC_SIZE        := 0x00000064

EE_BANK_BASE        := 0x00000000
EE_BANK_SIZE        := 0x00000000

EF_BANK_BASE        := 0x08000000  # Start of entire flash image (usually start of bootloader as well)
EF_BANK_SIZE        := 0x00080000  # Size of the entire flash image (from bootloader until end of firmware)

OSCILLATOR_FREQ     :=   8000000
SYSCLK_FREQ         := 168000000
