BOARD_TYPE          := 0x04
#must be rev.2 to match cc3d ini attitude.c which leads to using of the mpu_6000 sensor
BOARD_REVISION      := 0x02
BOOTLOADER_VERSION  := 0x03
HW_TYPE             := 0x01

MCU                 := cortex-m4
CHIP                := STM32F303VCT
BOARD               := STM32F30x_Discovery
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

