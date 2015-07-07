BOARD_TYPE          := 0x04
BOARD_REVISION      := 0x02
BOOTLOADER_VERSION  := 0x80
HW_TYPE             := 0x01

MCU                 := cortex-m3
CHIP                := STM32F103CBT
BOARD               := STM32103CB_Naze32
MODEL               := MD
MODEL_SUFFIX        := _CC

OPENOCD_JTAG_CONFIG := stlink-v2-norst.cfg
OPENOCD_CONFIG      := stm32f1xx.stlink.cfg

# Note: These must match the values in link_$(BOARD)_memory.ld
FW_BANK_BASE        := 0x08000000  # Start of firmware flash
FW_BANK_SIZE        := 0x0001D000  # Should include FW_DESC_SIZE (208kB)

FW_DESC_SIZE        := 0x00000064

EE_BANK_BASE        := 0x0801D000  # EEPROM storage area (@111kb)
EE_BANK_SIZE        := 0x00003000  # Size of EEPROM storage area (12kb)

EF_BANK_BASE        := 0x08000000  # Start of entire flash image (usually start of bootloader as well)
EF_BANK_SIZE        := 0x00020000  # Size of the entire flash image (from bootloader until end of firmware)
