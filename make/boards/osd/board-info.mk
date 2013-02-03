BOARD_TYPE          := 0x05
BOARD_REVISION      := 0x01
BOOTLOADER_VERSION  := 0x01
HW_TYPE             := 0x00

MCU                 := cortex-m4
CHIP                := STM32F405RGT6
BOARD               := STM32F4xx_OSD
MODEL               := HD
MODEL_SUFFIX        := 

OPENOCD_JTAG_CONFIG := stlink-v2.cfg
OPENOCD_CONFIG      := stm32f4xx.stlink.cfg
#OPENOCD_CONFIG      := stm32f4xx.cfg

# Note: These must match the values in link_$(BOARD)_memory.ld
BL_BANK_BASE        := 0x08000000  # Start of bootloader flash
BL_BANK_SIZE        := 0x00008000  # Should include BD_INFO region

# Leave the remaining 16KB and 64KB sectors for other uses

FW_BANK_BASE        := 0x08020000  # Start of firmware flash
FW_BANK_SIZE        := 0x00040000  # Should include FW_DESC_SIZE

FW_DESC_SIZE        := 0x00000064

EF_BANK_BASE        := 0x08000000  # Start of entire flash image (usually start of bootloader as well)
EF_BANK_SIZE        := 0x00060000  # Size of the entire flash image (from bootloader until end of firmware)

OSCILLATOR_FREQ     := 8000000
SYSCLK_FREQ         := 108000000
