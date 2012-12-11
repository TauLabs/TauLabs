BOARD_TYPE          := 0x09		# necessary to match revo to get proper tabs in gcs
BOARD_REVISION      := 0x03		# necessary to match revo revision else we won't get the correct sensors being used
BOOTLOADER_VERSION  := 0x01
HW_TYPE             := 0x00		# seems to be unused

MCU                 := cortex-m4
CHIP                := STM32F303VCT
BOARD               := STM32F30x_FLYINGF3
MODEL               := HD
MODEL_SUFFIX        := 

OPENOCD_JTAG_CONFIG := stlink-v2.cfg
OPENOCD_CONFIG      := stm32f3xx.stlink.cfg

# Note: These must match the values in link_$(BOARD)_memory.ld
BL_BANK_BASE        := 0x08000000  # Start of bootloader flash
BL_BANK_SIZE        := 0x00004000  # Should include BD_INFO region (16kb)

# Leave the remaining 16KB and 64KB sectors for other uses
FW_BANK_BASE        := 0x08008000  # Start of firmware flash (32kb)
FW_BANK_SIZE        := 0x00038000  # Should include FW_DESC_SIZE (208kb)

FW_DESC_SIZE        := 0x00000064

OSCILLATOR_FREQ     :=   8000000
SYSCLK_FREQ         :=  72000000
