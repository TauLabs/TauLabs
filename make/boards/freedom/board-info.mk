BOARD_TYPE          := 0x81
BOARD_REVISION      := 0x02
BOOTLOADER_VERSION  := 0x01
HW_TYPE             := 0x00

MCU                 := cortex-m4
CHIP                := STM32F405RGT
BOARD               := STM32F4xx_FREEDOM
MODEL               := HD
MODEL_SUFFIX        := 

OPENOCD_JTAG_CONFIG := stlink-v2.cfg
OPENOCD_CONFIG      := stm32f4xx.stlink.cfg

# Note: These must match the values in link_$(BOARD)_memory.ld
BL_BANK_BASE        := 0x08000000  # Start of bootloader flash
BL_BANK_SIZE        := 0x00008000  # Should include BD_INFO region

EE_BANK_BASE        := 0x08008000  # EEPROM storage area
EE_BANK_SIZE        := 0x00008000  # Size of EEPROM storage area

# Leave the 64KB sector for other uses

FW_BANK_BASE        := 0x08020000  # Start of firmware flash
FW_BANK_SIZE        := 0x00050000  # Should include FW_DESC_SIZE

FW_DESC_SIZE        := 0x00000064

EF_BANK_BASE        := 0x08000000  # Start of entire flash image (usually start of bootloader as well)
EF_BANK_SIZE        := 0x00060000  # Size of the entire flash image (from bootloader until end of firmware)

OSCILLATOR_FREQ     :=   8000000
SYSCLK_FREQ         := 168000000
