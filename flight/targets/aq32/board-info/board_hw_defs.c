/**
 ******************************************************************************
 * @addtogroup AeroQuadTargets AeroQuad Targets
 * @{
 * @addtogroup Aq32 Aq32 support files
 * @{
 *
 * @file       board_hw_defs.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      Defines board specific static initializers for hardware for the
 *             AQ32 board.
 * @see        The GNU Public License (GPL) Version 3
 *
 *****************************************************************************/
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include <pios_config.h>
#include <pios_board_info.h>

#if defined(PIOS_INCLUDE_LED)

#include <pios_led_priv.h>
static const struct pios_led pios_leds[] = {
	[PIOS_LED_HEARTBEAT] = {
		.pin = {
			.gpio = GPIOE,
			.init = {
				.GPIO_Pin   = GPIO_Pin_5,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_OUT,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL,
			},
		},
		.remap = 0,
		.active_high = true,
	},
	[PIOS_LED_ALARM] = {
		.pin = {
			.gpio = GPIOE,
			.init = {
				.GPIO_Pin   = GPIO_Pin_6,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_OUT,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL,
			},
		},
		.remap = 0,
		.active_high = true,
	},
};

static const struct pios_led_cfg pios_led_cfg = {
	.leds     = pios_leds,
	.num_leds = NELEMENTS(pios_leds),
};

const struct pios_led_cfg * PIOS_BOARD_HW_DEFS_GetLedCfg (uint32_t board_revision)
{
	return &pios_led_cfg;
}

#endif	/* PIOS_INCLUDE_LED */

#if defined(PIOS_INCLUDE_SPI)
#include <pios_spi_priv.h>

/* SPI3 Interface
 *      - Used for gyro communications
 */

void PIOS_SPI_internal_irq_handler(void);

void DMA2_Stream0_IRQHandler(void) __attribute__((alias("PIOS_SPI_internal_irq_handler")));

static const struct pios_spi_cfg pios_spi_internal_cfg = {
	.regs = SPI3,
	.remap = GPIO_AF_SPI3,
	.init = {
		.SPI_Mode              = SPI_Mode_Master,
		.SPI_Direction         = SPI_Direction_2Lines_FullDuplex,
		.SPI_DataSize          = SPI_DataSize_8b,
		.SPI_NSS               = SPI_NSS_Soft,
		.SPI_FirstBit          = SPI_FirstBit_MSB,
		.SPI_CRCPolynomial     = 7,
		.SPI_CPOL              = SPI_CPOL_High,
		.SPI_CPHA              = SPI_CPHA_2Edge,
		.SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_32,		//@ APB2 PCLK1 82MHz / 32 == 2.6MHz
	},
	.use_crc = false,
	.dma = {
		.irq = {
			// Note this is the stream ID that triggers interrupts (in this case RX)
			.flags = (DMA_IT_TCIF0 | DMA_IT_TEIF0 | DMA_IT_HTIF0),
			.init = {
				.NVIC_IRQChannel                   = DMA1_Stream0_IRQn,
				.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGH,
				.NVIC_IRQChannelSubPriority        = 0,
				.NVIC_IRQChannelCmd                = ENABLE,
			},
		},

		.rx = {
			.channel = DMA1_Stream0,
			.init = {
				.DMA_Channel            = DMA_Channel_0,
				.DMA_PeripheralBaseAddr = (uint32_t) & (SPI3->DR),
				.DMA_DIR                = DMA_DIR_PeripheralToMemory,
				.DMA_PeripheralInc      = DMA_PeripheralInc_Disable,
				.DMA_MemoryInc          = DMA_MemoryInc_Enable,
				.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte,
				.DMA_MemoryDataSize     = DMA_MemoryDataSize_Byte,
				.DMA_Mode               = DMA_Mode_Normal,
				.DMA_Priority           = DMA_Priority_Medium,
				//TODO: Enable FIFO
				.DMA_FIFOMode           = DMA_FIFOMode_Disable,
                .DMA_FIFOThreshold      = DMA_FIFOThreshold_Full,
                .DMA_MemoryBurst        = DMA_MemoryBurst_Single,
                .DMA_PeripheralBurst    = DMA_PeripheralBurst_Single,
			},
		},
		.tx = {
			.channel = DMA1_Stream5,
			.init = {
				.DMA_Channel            = DMA_Channel_0,
				.DMA_PeripheralBaseAddr = (uint32_t) & (SPI3->DR),
				.DMA_DIR                = DMA_DIR_MemoryToPeripheral,
				.DMA_PeripheralInc      = DMA_PeripheralInc_Disable,
				.DMA_MemoryInc          = DMA_MemoryInc_Enable,
				.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte,
				.DMA_MemoryDataSize     = DMA_MemoryDataSize_Byte,
				.DMA_Mode               = DMA_Mode_Normal,
				.DMA_Priority           = DMA_Priority_Medium,
				.DMA_FIFOMode           = DMA_FIFOMode_Disable,
                .DMA_FIFOThreshold      = DMA_FIFOThreshold_Full,
                .DMA_MemoryBurst        = DMA_MemoryBurst_Single,
                .DMA_PeripheralBurst    = DMA_PeripheralBurst_Single,
			},
		},
	},
	.sclk = {
		.gpio = GPIOC,
		.init = {
			.GPIO_Pin   = GPIO_Pin_10,
			.GPIO_Speed = GPIO_Speed_100MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_NOPULL
		},
		.pin_source = GPIO_PinSource10,
	},
	.miso = {
		.gpio = GPIOC,
		.init = {
			.GPIO_Pin   = GPIO_Pin_11,
			.GPIO_Speed = GPIO_Speed_100MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_NOPULL
		},
		.pin_source = GPIO_PinSource11,
	},
	.mosi = {
		.gpio = GPIOC,
		.init = {
			.GPIO_Pin   = GPIO_Pin_12,
			.GPIO_Speed = GPIO_Speed_100MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_NOPULL
		},
		.pin_source = GPIO_PinSource12,
	},
	.slave_count = 1,
	.ssel = { {
		.gpio = GPIOB,
		.init = {
			.GPIO_Pin   = GPIO_Pin_8,
			.GPIO_Speed = GPIO_Speed_100MHz,
			.GPIO_Mode  = GPIO_Mode_OUT,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_NOPULL
		},
	} },
};

uint32_t pios_spi_internal_id;
void PIOS_SPI_internal_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_SPI_IRQ_Handler(pios_spi_internal_id);
}

#endif	/* PIOS_INCLUDE_SPI */

#if defined(PIOS_INCLUDE_I2C)

#include <pios_i2c_priv.h>

void PIOS_I2C_internal_ev_irq_handler(void);
void PIOS_I2C_internal_er_irq_handler(void);

void I2C1_EV_IRQHandler() __attribute__ ((alias ("PIOS_I2C_internal_ev_irq_handler")));
void I2C1_ER_IRQHandler() __attribute__ ((alias ("PIOS_I2C_internal_er_irq_handler")));

static const struct pios_i2c_adapter_cfg pios_i2c_internal_cfg = {
  .regs = I2C1,
  .remap = GPIO_AF_I2C1,
  .init = {
    .I2C_Mode                = I2C_Mode_I2C,
    .I2C_OwnAddress1         = 0,
    .I2C_Ack                 = I2C_Ack_Enable,
    .I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit,
    .I2C_DutyCycle           = I2C_DutyCycle_2,
    .I2C_ClockSpeed          = 400000,	/* bits/s */
  },
  .transfer_timeout_ms = 50,
  .scl = {
    .gpio = GPIOB,
    .init = {
			.GPIO_Pin   = GPIO_Pin_6,
            .GPIO_Mode  = GPIO_Mode_AF,
            .GPIO_Speed = GPIO_Speed_50MHz,
            .GPIO_OType = GPIO_OType_OD,
            .GPIO_PuPd  = GPIO_PuPd_UP,
    },
	.pin_source = GPIO_PinSource6,
  },
  .sda = {
    .gpio = GPIOB,
    .init = {
			.GPIO_Pin   = GPIO_Pin_7,
            .GPIO_Mode  = GPIO_Mode_AF,
            .GPIO_Speed = GPIO_Speed_50MHz,
            .GPIO_OType = GPIO_OType_OD,
            .GPIO_PuPd  = GPIO_PuPd_UP,
    },
	.pin_source = GPIO_PinSource7,
  },
  .event = {
    .flags   = 0,		/* FIXME: check this */
    .init = {
			.NVIC_IRQChannel                   = I2C1_EV_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGHEST,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
    },
  },
  .error = {
    .flags   = 0,		/* FIXME: check this */
    .init = {
			.NVIC_IRQChannel                   = I2C1_ER_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGHEST,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
    },
  },
};

uint32_t pios_i2c_internal_id;
void PIOS_I2C_internal_ev_irq_handler(void)
{
  /* Call into the generic code to handle the IRQ for this specific device */
  PIOS_I2C_EV_IRQ_Handler(pios_i2c_internal_id);
}

void PIOS_I2C_internal_er_irq_handler(void)
{
  /* Call into the generic code to handle the IRQ for this specific device */
  PIOS_I2C_ER_IRQ_Handler(pios_i2c_internal_id);
}

void PIOS_I2C_external_ev_irq_handler(void);
void PIOS_I2C_external_er_irq_handler(void);

void I2C2_EV_IRQHandler() __attribute__ ((alias ("PIOS_I2C_external_ev_irq_handler")));
void I2C2_ER_IRQHandler() __attribute__ ((alias ("PIOS_I2C_external_er_irq_handler")));

static const struct pios_i2c_adapter_cfg pios_i2c_external_cfg = {
  .regs = I2C2,
  .remap = GPIO_AF_I2C2,
  .init = {
    .I2C_Mode                = I2C_Mode_I2C,
    .I2C_OwnAddress1         = 0,
    .I2C_Ack                 = I2C_Ack_Enable,
    .I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit,
    .I2C_DutyCycle           = I2C_DutyCycle_2,
    .I2C_ClockSpeed          = 400000,	/* bits/s */
  },
  .transfer_timeout_ms = 50,
  .scl = {
    .gpio = GPIOB,
    .init = {
			.GPIO_Pin   = GPIO_Pin_10,
            .GPIO_Mode  = GPIO_Mode_AF,
            .GPIO_Speed = GPIO_Speed_50MHz,
            .GPIO_OType = GPIO_OType_OD,
            .GPIO_PuPd  = GPIO_PuPd_UP,
    },
	.pin_source = GPIO_PinSource10,
  },
  .sda = {
    .gpio = GPIOB,
    .init = {
			.GPIO_Pin   = GPIO_Pin_11,
            .GPIO_Mode  = GPIO_Mode_AF,
            .GPIO_Speed = GPIO_Speed_50MHz,
            .GPIO_OType = GPIO_OType_OD,
            .GPIO_PuPd  = GPIO_PuPd_UP,
    },
	.pin_source = GPIO_PinSource11,
  },
  .event = {
    .flags   = 0,		/* FIXME: check this */
    .init = {
			.NVIC_IRQChannel                   = I2C2_EV_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGHEST,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
    },
  },
  .error = {
    .flags   = 0,		/* FIXME: check this */
    .init = {
			.NVIC_IRQChannel                   = I2C2_ER_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGHEST,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
    },
  },
};

uint32_t pios_i2c_external_id;
void PIOS_I2C_external_ev_irq_handler(void)
{
  /* Call into the generic code to handle the IRQ for this specific device */
  PIOS_I2C_EV_IRQ_Handler(pios_i2c_external_id);
}

void PIOS_I2C_external_er_irq_handler(void)
{
  /* Call into the generic code to handle the IRQ for this specific device */
  PIOS_I2C_ER_IRQ_Handler(pios_i2c_external_id);
}

#endif /* PIOS_INCLUDE_I2C */

#if defined(PIOS_INCLUDE_FLASH)
#include "pios_flashfs_logfs_priv.h"

static const struct flashfs_logfs_cfg flashfs_settings_cfg = {
	.fs_magic      = 0x3b1b14cf,
	.arena_size    = 0x00004000, /* 64 * slot size = 16K bytes = 1 sector */
	.slot_size     = 0x00000100, /* 256 bytes */
};

static const struct flashfs_logfs_cfg flashfs_waypoints_cfg = {
	.fs_magic      = 0x93a566a4,
	.arena_size    = 0x00020000,	/* 2048 * slot size = 128K bytes = 1 sector */
	.slot_size     = 0x00000040,	/* 64 bytes */
};

#include "pios_flash_internal_priv.h"

static const struct pios_flash_internal_cfg flash_internal_cfg = {
};

#include "pios_flash_priv.h"

static const struct pios_flash_sector_range stm32f4_sectors[] = {
	{
		.base_sector = 0,
		.last_sector = 3,
		.sector_size = FLASH_SECTOR_16KB,
	},
	{
		.base_sector = 4,
		.last_sector = 4,
		.sector_size = FLASH_SECTOR_64KB,
	},
	{
		.base_sector = 5,
		.last_sector = 11,
		.sector_size = FLASH_SECTOR_128KB,
	},

};

uintptr_t pios_internal_flash_id;
static const struct pios_flash_chip pios_flash_chip_internal = {
	.driver        = &pios_internal_flash_driver,
	.chip_id       = &pios_internal_flash_id,
	.page_size     = 16, /* 128-bit rows */
	.sector_blocks = stm32f4_sectors,
	.num_blocks    = NELEMENTS(stm32f4_sectors),
};

static const struct pios_flash_partition pios_flash_partition_table[] = {
	{
		.label        = FLASH_PARTITION_LABEL_BL,
		.chip_desc    = &pios_flash_chip_internal,
		.first_sector = 0,
		.last_sector  = 1,
		.chip_offset  = 0,                               // 0x0800 0000
		.size         = (1 - 0 + 1) * FLASH_SECTOR_16KB, // 32KB
	},

	{
		.label        = FLASH_PARTITION_LABEL_SETTINGS,
		.chip_desc    = &pios_flash_chip_internal,
		.first_sector = 2,
		.last_sector  = 3,
		.chip_offset  = (2 * FLASH_SECTOR_16KB),         // 0x0800 8000
		.size         = (3 - 2 + 1) * FLASH_SECTOR_16KB, // 32KB
	},

    /* NOTE: sector 4 of internal flash is currently unallocated */

	{
		.label        = FLASH_PARTITION_LABEL_FW,
		.chip_desc    = &pios_flash_chip_internal,
		.first_sector = 5,
		.last_sector  = 7,
		.chip_offset  = (4 * FLASH_SECTOR_16KB) + (1 * FLASH_SECTOR_64KB), // 0x0802 0000
		.size         = (7 - 5 + 1) * FLASH_SECTOR_128KB,                  // 384KB
	},

	/* NOTE: sectors 8-9 of the internal flash are currently unallocated */
    
	{
		.label        = FLASH_PARTITION_LABEL_WAYPOINTS,
		.chip_desc    = &pios_flash_chip_internal,
		.first_sector = 10,
		.last_sector  = 11,
		.chip_offset  = (4 * FLASH_SECTOR_16KB) + (1 * FLASH_SECTOR_64KB) + (5 * FLASH_SECTOR_128KB), // 0x080C 0000
		.size         = (11 - 10 + 1) * FLASH_SECTOR_128KB,                                           // 256KB
	}, 
};

const struct pios_flash_partition * PIOS_BOARD_HW_DEFS_GetPartitionTable (uint32_t board_revision, uint32_t * num_partitions)
{
	PIOS_Assert(num_partitions);

	*num_partitions = NELEMENTS(pios_flash_partition_table);
	return pios_flash_partition_table;
}

#endif	/* PIOS_INCLUDE_FLASH */

#if defined(PIOS_INCLUDE_USART)

#include "pios_usart_priv.h"

#if defined(PIOS_INCLUDE_DSM)
/*
 * Spektrum/JR DSM USART
 */
#include <pios_dsm_priv.h>

static const struct pios_dsm_cfg pios_usart4_dsm_aux_cfg = {
	.bind = {
		.gpio = GPIOA,
		.init = {
			.GPIO_Pin   = GPIO_Pin_1,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_OUT,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_NOPULL
		},
	},
};

static const struct pios_dsm_cfg pios_usart6_dsm_aux_cfg = {
	.bind = {
		.gpio = GPIOC,
		.init = {
			.GPIO_Pin   = GPIO_Pin_7,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_OUT,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_NOPULL
		},
	},
};

#endif	/* PIOS_INCLUDE_DSM */

#if defined(PIOS_INCLUDE_HSUM)
/*
 * Graupner HoTT SUMD/SUMH USART
 */
#include <pios_hsum_priv.h>

#endif	/* PIOS_INCLUDE_HSUM */

#if (defined(PIOS_INCLUDE_DSM) || defined(PIOS_INCLUDE_HSUM))
/*
 * Spektrum/JR DSM or Graupner HoTT SUMD/SUMH USART
 */

static const struct pios_usart_cfg pios_usart3_dsm_hsum_cfg = {
	.regs = USART3,
	.remap = GPIO_AF_USART3,
	.init = {
		.USART_BaudRate            = 115200,
		.USART_WordLength          = USART_WordLength_8b,
		.USART_Parity              = USART_Parity_No,
		.USART_StopBits            = USART_StopBits_1,
		.USART_HardwareFlowControl = USART_HardwareFlowControl_None,
		.USART_Mode                = USART_Mode_Rx,
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = USART3_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
	.rx = {
		.gpio = GPIOD,
		.init = {
			.GPIO_Pin   = GPIO_Pin_9,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource9,
	},
};

static const struct pios_usart_cfg pios_usart4_dsm_hsum_cfg = {
	.regs = UART4,
	.remap = GPIO_AF_UART4,
	.init = {
		.USART_BaudRate            = 115200,
		.USART_WordLength          = USART_WordLength_8b,
		.USART_Parity              = USART_Parity_No,
		.USART_StopBits            = USART_StopBits_1,
		.USART_HardwareFlowControl = USART_HardwareFlowControl_None,
		.USART_Mode                = USART_Mode_Rx,
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = UART4_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
	.rx = {
		.gpio = GPIOA,
		.init = {
			.GPIO_Pin   = GPIO_Pin_1,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource1,
	},
};

static const struct pios_usart_cfg pios_usart6_dsm_hsum_cfg = {
	.regs = USART6,
	.remap = GPIO_AF_USART6,
	.init = {
		.USART_BaudRate            = 115200,
		.USART_WordLength          = USART_WordLength_8b,
		.USART_Parity              = USART_Parity_No,
		.USART_StopBits            = USART_StopBits_1,
		.USART_HardwareFlowControl = USART_HardwareFlowControl_None,
		.USART_Mode                = USART_Mode_Rx,
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = USART6_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
	.rx = {
		.gpio = GPIOC,
		.init = {
			.GPIO_Pin   = GPIO_Pin_7,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource7,
	},
};

#endif	/* PIOS_INCLUDE_DSM || PIOS_INCLUDE_HSUM */

#if defined(PIOS_INCLUDE_SBUS)
/*
 * S.Bus USART
 */
#include <pios_sbus_priv.h>

static const struct pios_usart_cfg pios_usart3_sbus_cfg = {
	.regs = USART3,
	.remap = GPIO_AF_USART3,
	.init = {
		.USART_BaudRate            = 100000,
		.USART_WordLength          = USART_WordLength_8b,
		.USART_Parity              = USART_Parity_Even,
		.USART_StopBits            = USART_StopBits_2,
		.USART_HardwareFlowControl = USART_HardwareFlowControl_None,
		.USART_Mode                = USART_Mode_Rx,
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = USART3_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		  },
	},
	.rx = {
		.gpio = GPIOD,
		.init = {
			.GPIO_Pin   = GPIO_Pin_9,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource9,
	},
};

static const struct pios_sbus_cfg pios_usart3_sbus_aux_cfg = {
	/* Inverter configuration */
	.inv = {
		.gpio = GPIOB,
		.init = {
			.GPIO_Pin   = GPIO_Pin_12,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_OUT,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_NOPULL
		},
	},
	.gpio_inv_enable = Bit_SET,
};

#endif	/* PIOS_INCLUDE_SBUS */

static const struct pios_usart_cfg pios_usart1_cfg = {
	.regs = USART1,
	.remap = GPIO_AF_USART1,
	.init = {
		.USART_BaudRate            = 57600,
		.USART_WordLength          = USART_WordLength_8b,
		.USART_Parity              = USART_Parity_No,
		.USART_StopBits            = USART_StopBits_1,
		.USART_HardwareFlowControl = USART_HardwareFlowControl_None,
		.USART_Mode                = USART_Mode_Rx | USART_Mode_Tx,
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = USART1_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
	.rx = {
		.gpio = GPIOA,
		.init = {
			.GPIO_Pin   = GPIO_Pin_10,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource10,
	},
	.tx = {
		.gpio = GPIOA,
		.init = {
			.GPIO_Pin   = GPIO_Pin_9,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource9,
	},
};

static const struct pios_usart_cfg pios_usart2_cfg = {
	.regs = USART2,
	.remap = GPIO_AF_USART2,
	.init = {
		.USART_BaudRate            = 57600,
		.USART_WordLength          = USART_WordLength_8b,
		.USART_Parity              = USART_Parity_No,
		.USART_StopBits            = USART_StopBits_1,
		.USART_HardwareFlowControl = USART_HardwareFlowControl_None,
		.USART_Mode                = USART_Mode_Rx | USART_Mode_Tx,
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = USART2_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
	.rx = {
		.gpio = GPIOD,
		.init = {
			.GPIO_Pin   = GPIO_Pin_6,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource6,
	},
	.tx = {
		.gpio = GPIOD,
		.init = {
			.GPIO_Pin   = GPIO_Pin_5,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource5,
	},
};

static const struct pios_usart_cfg pios_usart3_cfg = {
	.regs = USART3,
	.remap = GPIO_AF_USART3,
	.init = {
		.USART_BaudRate            = 57600,
		.USART_WordLength          = USART_WordLength_8b,
		.USART_Parity              = USART_Parity_No,
		.USART_StopBits            = USART_StopBits_1,
		.USART_HardwareFlowControl = USART_HardwareFlowControl_None,
		.USART_Mode                = USART_Mode_Rx | USART_Mode_Tx,
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = USART3_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
	.rx = {
		.gpio = GPIOD,
		.init = {
			.GPIO_Pin   = GPIO_Pin_9,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource9,
	},
	.tx = {
		.gpio = GPIOD,
		.init = {
			.GPIO_Pin   = GPIO_Pin_8,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource8,
	},
};

static const struct pios_usart_cfg pios_usart4_cfg = {
	.regs = UART4,
	.remap = GPIO_AF_UART4,
	.init = {
		.USART_BaudRate            = 57600,
		.USART_WordLength          = USART_WordLength_8b,
		.USART_Parity              = USART_Parity_No,
		.USART_StopBits            = USART_StopBits_1,
		.USART_HardwareFlowControl = USART_HardwareFlowControl_None,
		.USART_Mode                = USART_Mode_Rx | USART_Mode_Tx,
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = UART4_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
	.rx = {
		.gpio = GPIOA,
		.init = {
			.GPIO_Pin   = GPIO_Pin_1,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource1,
	},
	.tx = {
		.gpio = GPIOA,
		.init = {
			.GPIO_Pin   = GPIO_Pin_0,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource0,
	},
};

static const struct pios_usart_cfg pios_usart6_cfg = {
	.regs = USART6,
	.remap = GPIO_AF_USART6,
	.init = {
		.USART_BaudRate            = 57600,
		.USART_WordLength          = USART_WordLength_8b,
		.USART_Parity              = USART_Parity_No,
		.USART_StopBits            = USART_StopBits_1,
		.USART_HardwareFlowControl = USART_HardwareFlowControl_None,
		.USART_Mode                = USART_Mode_Rx | USART_Mode_Tx,
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = USART6_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
	.rx = {
		.gpio = GPIOC,
		.init = {
			.GPIO_Pin   = GPIO_Pin_7,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource7,
	},
	.tx = {
		.gpio = GPIOC,
		.init = {
			.GPIO_Pin   = GPIO_Pin_6,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd  = GPIO_PuPd_UP
		},
		.pin_source = GPIO_PinSource6,
	},
};

#endif  /* PIOS_INCLUDE_USART */

#if defined(PIOS_INCLUDE_COM)

#include "pios_com_priv.h"

#endif	/* PIOS_INCLUDE_COM */

#if defined(PIOS_INCLUDE_RTC)
/*
 * Realtime Clock (RTC)
 */
#include <pios_rtc_priv.h>

void PIOS_RTC_IRQ_Handler (void);
void RTC_WKUP_IRQHandler() __attribute__ ((alias ("PIOS_RTC_IRQ_Handler")));
static const struct pios_rtc_cfg pios_rtc_main_cfg = {
	.clksrc = RCC_RTCCLKSource_HSE_Div8, // Divide 8 MHz HSE by 8 = 1 MHz
	// 1 MHz resulting clock is then divided
	// by another 16 to give a nominal 62.5 kHz clock
	.prescaler = 100, // Every 100 cycles gives 625 Hz
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = RTC_WKUP_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
};

void PIOS_RTC_IRQ_Handler (void)
{
	PIOS_RTC_irq_handler ();
}

#endif

#include "pios_tim_priv.h"

//Timers used for inputs (4)

// Set up timers that only have outputs on APB1
static const TIM_TimeBaseInitTypeDef tim_4_time_base = {
	.TIM_Prescaler = (PIOS_PERIPHERAL_APB1_CLOCK / 1000000) - 1,
	.TIM_ClockDivision = TIM_CKD_DIV1,
	.TIM_CounterMode = TIM_CounterMode_Up,
	.TIM_Period = 0xFFFF,
	.TIM_RepetitionCounter = 0x0000,
};

static const struct pios_tim_clock_cfg tim_4_cfg = {
	.timer = TIM4,
	.time_base_init = &tim_4_time_base,
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = TIM4_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
};

// Timers used for outputs (1,2,3,8)

// Set up timers that only have outputs on APB1
static const TIM_TimeBaseInitTypeDef tim_2_3_time_base = {
	.TIM_Prescaler         = (PIOS_PERIPHERAL_APB1_CLOCK / 1000000) - 1,
	.TIM_ClockDivision     = TIM_CKD_DIV1,
	.TIM_CounterMode       = TIM_CounterMode_Up,
	.TIM_Period            = ((1000000 / PIOS_SERVO_UPDATE_HZ) - 1),
	.TIM_RepetitionCounter = 0x0000,
};

// Set up timers that only have outputs on APB2
static const TIM_TimeBaseInitTypeDef tim_1_8_time_base = {
	.TIM_Prescaler         = (PIOS_PERIPHERAL_APB2_CLOCK / 1000000) - 1,
	.TIM_ClockDivision     = TIM_CKD_DIV1,
	.TIM_CounterMode       = TIM_CounterMode_Up,
	.TIM_Period            = ((1000000 / PIOS_SERVO_UPDATE_HZ) - 1),
	.TIM_RepetitionCounter = 0x0000,
};

static const struct pios_tim_clock_cfg tim_1_cfg = {
	.timer = TIM1,
	.time_base_init = &tim_1_8_time_base,
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = TIM1_CC_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
};

static const struct pios_tim_clock_cfg tim_2_cfg = {
	.timer = TIM2,
	.time_base_init = &tim_2_3_time_base,
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = TIM2_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
};

static const struct pios_tim_clock_cfg tim_3_cfg = {
	.timer = TIM3,
	.time_base_init = &tim_2_3_time_base,
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = TIM3_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
};

static const struct pios_tim_clock_cfg tim_8_cfg = {
	.timer = TIM8,
	.time_base_init = &tim_1_8_time_base,
	.irq = {
		.init = {
			.NVIC_IRQChannel                   = TIM8_CC_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
};

/**
 * Pios servo configuration structures
 */

/*
 * Available outputs with PPM RX
	1:  TIM8_CH4 (PC9)
	2:  TIM8_CH3 (PC8)
	3:  TIM2_CH1 (PA15)
	4:  TIM2_CH2 (PB3)
	5:  TIM3_CH1 (PB4)
	6:  TIM3_CH2 (PB5)
	7:  TIM1_CH1 (PE9)
	8:  TIM1_CH2 (PE11)
	9:  TIM1_CH3 (PE13)
	10: TIM1_CH4 (PE14)
 */
static const struct pios_tim_channel pios_tim_outputs_pins_ppm_rx[] = {
	{
		.timer = TIM8,
		.timer_chan = TIM_Channel_4,
		.remap = GPIO_AF_TIM8,
		.pin = {
			.gpio = GPIOC,
			.init = {
				.GPIO_Pin   = GPIO_Pin_9,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource9,
		},
	},
	{
		.timer = TIM8,
		.timer_chan = TIM_Channel_3,
		.remap = GPIO_AF_TIM8,
		.pin = {
			.gpio = GPIOC,
			.init = {
				.GPIO_Pin   = GPIO_Pin_8,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource8,
		},
	},
	{
		.timer = TIM2,
		.timer_chan = TIM_Channel_1,
		.remap = GPIO_AF_TIM2,
		.pin = {
			.gpio = GPIOA,
			.init = {
				.GPIO_Pin   = GPIO_Pin_15,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource15,
		},
	},
	{
		.timer = TIM2,
		.timer_chan = TIM_Channel_2,
		.remap = GPIO_AF_TIM2,
		.pin = {
			.gpio = GPIOB,
			.init = {
				.GPIO_Pin   = GPIO_Pin_3,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource3,
		},
	},
	{
		.timer = TIM3,
		.timer_chan = TIM_Channel_1,
		.remap = GPIO_AF_TIM3,
		.pin = {
			.gpio = GPIOB,
			.init = {
				.GPIO_Pin   = GPIO_Pin_4,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource4,
		},
	},
	{
		.timer = TIM3,
		.timer_chan = TIM_Channel_2,
		.remap = GPIO_AF_TIM3,
		.pin = {
			.gpio = GPIOB,
			.init = {
				.GPIO_Pin   = GPIO_Pin_5,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource5,
		},
	},
	{
		.timer = TIM1,
		.timer_chan = TIM_Channel_1,
		.remap = GPIO_AF_TIM1,
		.pin = {
			.gpio = GPIOE,
			.init = {
				.GPIO_Pin   = GPIO_Pin_9,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource9,
		},
	},
	{
		.timer = TIM1,
		.timer_chan = TIM_Channel_2,
		.remap = GPIO_AF_TIM1,
		.pin = {
			.gpio = GPIOE,
			.init = {
				.GPIO_Pin   = GPIO_Pin_11,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource11,
		},
	},
	{
		.timer = TIM1,
		.timer_chan = TIM_Channel_3,
		.remap = GPIO_AF_TIM1,
		.pin = {
			.gpio = GPIOE,
			.init = {
				.GPIO_Pin   = GPIO_Pin_13,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource13,
		},
	},
	{
		.timer = TIM1,
		.timer_chan = TIM_Channel_4,
		.remap = GPIO_AF_TIM1,
		.pin = {
			.gpio = GPIOE,
			.init = {
				.GPIO_Pin   = GPIO_Pin_14,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource14,
		},
	},
};

/*
 * Available outputs with PWM RX
	1:  TIM8_CH4 (PC9)
	2:  TIM8_CH3 (PC8)
	3:  TIM2_CH1 (PA15)
	4:  TIM2_CH2 (PB3)
	5:  TIM3_CH1 (PB4)
	6:  TIM3_CH2 (PB5)
 */
static const struct pios_tim_channel pios_tim_outputs_pins_pwm_rx[] = {
	{
		.timer = TIM8,
		.timer_chan = TIM_Channel_4,
		.remap = GPIO_AF_TIM8,
		.pin = {
			.gpio = GPIOC,
			.init = {
				.GPIO_Pin   = GPIO_Pin_9,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource9,
		},
	},
	{
		.timer = TIM8,
		.timer_chan = TIM_Channel_3,
		.remap = GPIO_AF_TIM8,
		.pin = {
			.gpio = GPIOC,
			.init = {
				.GPIO_Pin   = GPIO_Pin_8,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource8,
		},
	},
	{
		.timer = TIM2,
		.timer_chan = TIM_Channel_1,
		.remap = GPIO_AF_TIM2,
		.pin = {
			.gpio = GPIOA,
			.init = {
				.GPIO_Pin   = GPIO_Pin_15,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource15,
		},
	},
	{
		.timer = TIM2,
		.timer_chan = TIM_Channel_2,
		.remap = GPIO_AF_TIM2,
		.pin = {
			.gpio = GPIOB,
			.init = {
				.GPIO_Pin   = GPIO_Pin_3,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource3,
		},
	},
	{
		.timer = TIM3,
		.timer_chan = TIM_Channel_1,
		.remap = GPIO_AF_TIM3,
		.pin = {
			.gpio = GPIOB,
			.init = {
				.GPIO_Pin   = GPIO_Pin_4,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource4,
		},
	},
	{
		.timer = TIM3,
		.timer_chan = TIM_Channel_2,
		.remap = GPIO_AF_TIM3,
		.pin = {
			.gpio = GPIOB,
			.init = {
				.GPIO_Pin   = GPIO_Pin_5,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource5,
		},
	},
};

#if defined(PIOS_INCLUDE_SERVO) && defined(PIOS_INCLUDE_TIM)
/*
 * Servo outputs
 */
#include <pios_servo_priv.h>

const struct pios_servo_cfg pios_servo_cfg_ppm_rx = {
	.tim_oc_init = {
		.TIM_OCMode       = TIM_OCMode_PWM1,
		.TIM_OutputState  = TIM_OutputState_Enable,
		.TIM_OutputNState = TIM_OutputNState_Disable,
		.TIM_Pulse        = PIOS_SERVOS_INITIAL_POSITION,
		.TIM_OCPolarity   = TIM_OCPolarity_High,
		.TIM_OCNPolarity  = TIM_OCPolarity_High,
		.TIM_OCIdleState  = TIM_OCIdleState_Reset,
		.TIM_OCNIdleState = TIM_OCNIdleState_Reset,
	},
	.channels = pios_tim_outputs_pins_ppm_rx,
	.num_channels = NELEMENTS(pios_tim_outputs_pins_ppm_rx),
};

const struct pios_servo_cfg pios_servo_cfg_pwm_rx = {
	.tim_oc_init = {
		.TIM_OCMode       = TIM_OCMode_PWM1,
		.TIM_OutputState  = TIM_OutputState_Enable,
		.TIM_OutputNState = TIM_OutputNState_Disable,
		.TIM_Pulse        = PIOS_SERVOS_INITIAL_POSITION,
		.TIM_OCPolarity   = TIM_OCPolarity_High,
		.TIM_OCNPolarity  = TIM_OCPolarity_High,
		.TIM_OCIdleState  = TIM_OCIdleState_Reset,
		.TIM_OCNIdleState = TIM_OCNIdleState_Reset,
	},
	.channels = pios_tim_outputs_pins_pwm_rx,
	.num_channels = NELEMENTS(pios_tim_outputs_pins_pwm_rx),
};
#endif	/* PIOS_INCLUDE_SERVO && PIOS_INCLUDE_TIM */

/*
 * 	PWM INPUTS
	1: TIM4_CH1 (PD12)
	2: TIM4_CH2 (PD13)
	3: TIM4_CH3 (PD14)
	4: TIM4_CH4 (PD15)
	5: TIM1_CH1 (PE9)
	6: TIM1_CH2 (PE11)
	7: TIM1_CH3 (PE13)
	8: TIM1_CH4 (PE14)
 */
static const struct pios_tim_channel pios_tim_rcvrport_pwm[] = {
	{
		.timer = TIM4,
		.timer_chan = TIM_Channel_1,
		.remap = GPIO_AF_TIM4,
		.pin = {
			.gpio = GPIOD,
			.init = {
				.GPIO_Pin = GPIO_Pin_12,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_UP
			},
			.pin_source = GPIO_PinSource12,
		},
	},
	{
		.timer = TIM4,
		.timer_chan = TIM_Channel_2,
		.remap = GPIO_AF_TIM4,
		.pin = {
			.gpio = GPIOD,
			.init = {
				.GPIO_Pin = GPIO_Pin_13,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_UP
			},
			.pin_source = GPIO_PinSource13,
		},
	},
	{
		.timer = TIM4,
		.timer_chan = TIM_Channel_3,
		.remap = GPIO_AF_TIM4,
		.pin = {
			.gpio = GPIOD,
			.init = {
				.GPIO_Pin = GPIO_Pin_14,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_UP
			},
			.pin_source = GPIO_PinSource14,
		},
	},
	{
		.timer = TIM4,
		.timer_chan = TIM_Channel_4,
		.remap = GPIO_AF_TIM4,
		.pin = {
			.gpio = GPIOD,
			.init = {
				.GPIO_Pin = GPIO_Pin_15,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_UP
			},
			.pin_source = GPIO_PinSource15,
		},
	},
	{
		.timer = TIM1,
		.timer_chan = TIM_Channel_1,
		.remap = GPIO_AF_TIM1,
		.pin = {
			.gpio = GPIOE,
			.init = {
				.GPIO_Pin = GPIO_Pin_9,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_UP
			},
			.pin_source = GPIO_PinSource9,
		},
	},
	{
		.timer = TIM1,
		.timer_chan = TIM_Channel_2,
		.remap = GPIO_AF_TIM1,
		.pin = {
			.gpio = GPIOE,
			.init = {
				.GPIO_Pin = GPIO_Pin_11,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_UP
			},
			.pin_source = GPIO_PinSource11,
		},
	},
	{
		.timer = TIM1,
		.timer_chan = TIM_Channel_3,
		.remap = GPIO_AF_TIM1,
		.pin = {
			.gpio = GPIOE,
			.init = {
				.GPIO_Pin = GPIO_Pin_13,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_UP
			},
			.pin_source = GPIO_PinSource13,
		},
	},
	{
		.timer = TIM1,
		.timer_chan = TIM_Channel_4,
		.remap = GPIO_AF_TIM1,
		.pin = {
			.gpio = GPIOE,
			.init = {
				.GPIO_Pin = GPIO_Pin_14,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_AF,
				.GPIO_OType = GPIO_OType_PP,
				.GPIO_PuPd  = GPIO_PuPd_UP
			},
			.pin_source = GPIO_PinSource14,
		},
	},
};

/*
 * PWM Inputs
 */
#if defined(PIOS_INCLUDE_PWM) || defined(PIOS_INCLUDE_PPM)
#include <pios_pwm_priv.h>
#include <pios_ppm_priv.h>

/*
 * PWM Inputs
 */
const struct pios_pwm_cfg pios_pwm_cfg = {
	.tim_ic_init = {
		.TIM_ICPolarity = TIM_ICPolarity_Rising,
		.TIM_ICSelection = TIM_ICSelection_DirectTI,
		.TIM_ICPrescaler = TIM_ICPSC_DIV1,
		.TIM_ICFilter = 0x0,
	},
	.channels = pios_tim_rcvrport_pwm,
	.num_channels = NELEMENTS(pios_tim_rcvrport_pwm),
};

/*
 * PPM Input
 */
const struct pios_ppm_cfg pios_ppm_cfg = {
	.tim_ic_init = {
		.TIM_ICPolarity = TIM_ICPolarity_Rising,
		.TIM_ICSelection = TIM_ICSelection_DirectTI,
		.TIM_ICPrescaler = TIM_ICPSC_DIV1,
		.TIM_ICFilter = 0x0,
		.TIM_Channel = TIM_Channel_4,
	},
	/* Channel 4 for PPM use */
	.channels = &pios_tim_rcvrport_pwm[3],
	.num_channels = 1,
};

#endif

#if defined(PIOS_INCLUDE_GCSRCVR)
#include "pios_gcsrcvr_priv.h"
#endif	/* PIOS_INCLUDE_GCSRCVR */

#if defined(PIOS_INCLUDE_RCVR)
#include "pios_rcvr_priv.h"
#endif /* PIOS_INCLUDE_RCVR */

#if defined(PIOS_INCLUDE_USB)
#include "pios_usb_priv.h"

static const struct pios_usb_cfg pios_usb_main_cfg = {
	.irq = {
		.init    = {
			.NVIC_IRQChannel                   = OTG_FS_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGHEST,
			.NVIC_IRQChannelSubPriority        = 0,
			.NVIC_IRQChannelCmd                = ENABLE,
		},
	},
	.vsense = {
		.gpio = NULL,
	},
	.disconnect = {
		.gpio = GPIOD,
		.init = {
			.GPIO_Pin   = GPIO_Pin_11,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_OUT,
			.GPIO_OType = GPIO_OType_OD,
			.GPIO_PuPd  = GPIO_PuPd_NOPULL,
		},
	}
};

const struct pios_usb_cfg * PIOS_BOARD_HW_DEFS_GetUsbCfg (uint32_t board_revision)
{
	return &pios_usb_main_cfg;
}

#include "pios_usb_board_data_priv.h"
#include "pios_usb_desc_hid_cdc_priv.h"
#include "pios_usb_desc_hid_only_priv.h"
#include "pios_usbhook.h"

#endif	/* PIOS_INCLUDE_USB */

#if defined(PIOS_INCLUDE_COM_MSG)

#include <pios_com_msg_priv.h>

#endif /* PIOS_INCLUDE_COM_MSG */

#if defined(PIOS_INCLUDE_USB_HID) && !defined(PIOS_INCLUDE_USB_CDC)
#include <pios_usb_hid_priv.h>

const struct pios_usb_hid_cfg pios_usb_hid_cfg = {
	.data_if = 0,
	.data_rx_ep = 1,
	.data_tx_ep = 1,
};
#endif /* PIOS_INCLUDE_USB_HID && !PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_USB_HID) && defined(PIOS_INCLUDE_USB_CDC)
#include <pios_usb_cdc_priv.h>

const struct pios_usb_cdc_cfg pios_usb_cdc_cfg = {
	.ctrl_if = 0,
	.ctrl_tx_ep = 2,

	.data_if = 1,
	.data_rx_ep = 3,
	.data_tx_ep = 3,
};

#include <pios_usb_hid_priv.h>

const struct pios_usb_hid_cfg pios_usb_hid_cfg = {
	.data_if = 2,
	.data_rx_ep = 1,
	.data_tx_ep = 1,
};
#endif	/* PIOS_INCLUDE_USB_HID && PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_ADC)
#include "pios_adc_priv.h"
#include "pios_internal_adc_priv.h"

void PIOS_ADC_DMA_irq_handler(void);
void DMA2_Stream4_IRQHandler(void) __attribute__((alias("PIOS_ADC_DMA_irq_handler")));

struct pios_internal_adc_cfg pios_adc_cfg = {
	.dma = {
		.irq = {
			.flags = (DMA_FLAG_TCIF4 | DMA_FLAG_TEIF4 | DMA_FLAG_HTIF4),
			.init = {
				.NVIC_IRQChannel = DMA2_Stream4_IRQn,
				.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_LOW,
				.NVIC_IRQChannelSubPriority = 0,
				.NVIC_IRQChannelCmd = ENABLE,
			},
		},
		.rx = {
			.channel = DMA2_Stream4,
			.init = {
				.DMA_Channel = DMA_Channel_0,
				.DMA_PeripheralBaseAddr = (uint32_t)&ADC1->DR
			},
		}
	},
	.half_flag = DMA_IT_HTIF4,
	.full_flag = DMA_IT_TCIF4,

	.adc_dev_master = ADC1,
	.adc_dev_slave  = NULL,
};

void PIOS_ADC_DMA_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_INTERNAL_ADC_DMA_Handler();
}

#endif /* PIOS_INCLUDE_ADC */

/**
 * @}
 * @}
 */
