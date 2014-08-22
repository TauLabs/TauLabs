/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup Colibri Colibri support files
 * @{
 *
 * @file       board_hw_defs.c 
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      Defines board specific static initializers for hardware for the
 *             Colibri board.
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
	[PIOS_LED_RED] = {
			  .pin = {
				  .gpio = GPIOC,
				  .init = {
					   .GPIO_Pin = GPIO_Pin_13,
					   .GPIO_Speed = GPIO_Speed_2MHz,
					   .GPIO_Mode = GPIO_Mode_OUT,
					   .GPIO_OType = GPIO_OType_PP,
					   .GPIO_PuPd = GPIO_PuPd_DOWN},
				  },
			  .remap = 0,
			  .active_high = false,
			  },
	[PIOS_LED_BLUE] = {
			   .pin = {
				   .gpio = GPIOC,
				   .init = {
					    .GPIO_Pin = GPIO_Pin_14,
					    .GPIO_Speed = GPIO_Speed_2MHz,
					    .GPIO_Mode = GPIO_Mode_OUT,
					    .GPIO_OType = GPIO_OType_PP,
					    .GPIO_PuPd = GPIO_PuPd_DOWN},
				   },
			   .remap = 0,
			   .active_high = false,
			   },
};

static const struct pios_led_cfg pios_led_cfg = {
	.leds = pios_leds,
	.num_leds = NELEMENTS(pios_leds),
};

const struct pios_led_cfg *PIOS_BOARD_HW_DEFS_GetLedCfg(uint32_t
							board_revision)
{
	return &pios_led_cfg;
}

#endif /* PIOS_INCLUDE_LED */

#if defined(PIOS_INCLUDE_SPI)
#include <pios_spi_priv.h>

/* SPI2 Interface
 *      - Used for flash communications
 */
void PIOS_SPI_flash_irq_handler(void);
void DMA1_Stream3_IRQHandler(void)
    __attribute__ ((alias("PIOS_SPI_flash_irq_handler")));

static const struct pios_spi_cfg pios_spi_flash_cfg = {
	.regs = SPI2,
	.remap = GPIO_AF_SPI2,
	.init = {
		 .SPI_Mode = SPI_Mode_Master,
		 .SPI_Direction = SPI_Direction_2Lines_FullDuplex,
		 .SPI_DataSize = SPI_DataSize_8b,
		 .SPI_NSS = SPI_NSS_Soft,
		 .SPI_FirstBit = SPI_FirstBit_MSB,
		 .SPI_CRCPolynomial = 7,
		 .SPI_CPOL = SPI_CPOL_High,
		 .SPI_CPHA = SPI_CPHA_2Edge,
		 .SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_2,	//@ APB1 PCLK1 42MHz / 2 == 21MHz
		 },
	.use_crc = false,
	.dma = {
		.irq = {
			// Note this is the stream ID that triggers interrupts (in this case RX)
			.flags =
			(DMA_IT_TCIF3 | DMA_IT_TEIF3 | DMA_IT_HTIF3),
			.init = {
				 .NVIC_IRQChannel = DMA1_Stream3_IRQn,
				 .NVIC_IRQChannelPreemptionPriority =
				 PIOS_IRQ_PRIO_HIGH,
				 .NVIC_IRQChannelSubPriority = 0,
				 .NVIC_IRQChannelCmd = ENABLE,
				 },
			},

		.rx = {
		       .channel = DMA1_Stream3,
		       .init = {
				.DMA_Channel = DMA_Channel_0,
				.DMA_PeripheralBaseAddr =
				(uint32_t) & (SPI2->DR),
				.DMA_DIR = DMA_DIR_PeripheralToMemory,
				.DMA_PeripheralInc =
				DMA_PeripheralInc_Disable,
				.DMA_MemoryInc = DMA_MemoryInc_Enable,
				.DMA_PeripheralDataSize =
				DMA_PeripheralDataSize_Byte,
				.DMA_MemoryDataSize =
				DMA_MemoryDataSize_Byte,
				.DMA_Mode = DMA_Mode_Normal,
				.DMA_Priority = DMA_Priority_Medium,
				//TODO: Enable FIFO
				.DMA_FIFOMode = DMA_FIFOMode_Disable,
				.DMA_FIFOThreshold =
				DMA_FIFOThreshold_Full,
				.DMA_MemoryBurst = DMA_MemoryBurst_Single,
				.DMA_PeripheralBurst =
				DMA_PeripheralBurst_Single,
				},
		       },
		.tx = {
		       .channel = DMA1_Stream4,
		       .init = {
				.DMA_Channel = DMA_Channel_0,
				.DMA_PeripheralBaseAddr =
				(uint32_t) & (SPI2->DR),
				.DMA_DIR = DMA_DIR_MemoryToPeripheral,
				.DMA_PeripheralInc =
				DMA_PeripheralInc_Disable,
				.DMA_MemoryInc = DMA_MemoryInc_Enable,
				.DMA_PeripheralDataSize =
				DMA_PeripheralDataSize_Byte,
				.DMA_MemoryDataSize =
				DMA_MemoryDataSize_Byte,
				.DMA_Mode = DMA_Mode_Normal,
				.DMA_Priority = DMA_Priority_Medium,
				.DMA_FIFOMode = DMA_FIFOMode_Disable,
				.DMA_FIFOThreshold =
				DMA_FIFOThreshold_Full,
				.DMA_MemoryBurst = DMA_MemoryBurst_Single,
				.DMA_PeripheralBurst =
				DMA_PeripheralBurst_Single,
				},
		       },
		},
	.sclk = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_13,
			  .GPIO_Speed = GPIO_Speed_100MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 .pin_source = GPIO_PinSource13,
		 },
	.miso = {
		 .gpio = GPIOC,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_2,
			  .GPIO_Speed = GPIO_Speed_50MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 .pin_source = GPIO_PinSource2,
		 },
	.mosi = {
		 .gpio = GPIOC,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_3,
			  .GPIO_Speed = GPIO_Speed_50MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 .pin_source = GPIO_PinSource3,
		 },
	.slave_count = 1,
	.ssel = {{
		  .gpio = GPIOB,
		  .init = {
			   .GPIO_Pin = GPIO_Pin_12,
			   .GPIO_Speed = GPIO_Speed_50MHz,
			   .GPIO_Mode = GPIO_Mode_OUT,
			   .GPIO_OType = GPIO_OType_PP,
			   .GPIO_PuPd = GPIO_PuPd_UP},
		  }},
};

uint32_t pios_spi_flash_id;
void PIOS_SPI_flash_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_SPI_IRQ_Handler(pios_spi_flash_id);
}

/* SPI1 Interface
 *      - Used for gyro communications
 */
void PIOS_SPI_gyro_accel_irq_handler(void);
void DMA2_Stream0_IRQHandler(void)
    __attribute__ ((alias("PIOS_SPI_gyro_accel_irq_handler")));

static const struct pios_spi_cfg pios_spi_gyro_accel_cfg = {
	.regs = SPI1,
	.remap = GPIO_AF_SPI1,
	.init = {
		 .SPI_Mode = SPI_Mode_Master,
		 .SPI_Direction = SPI_Direction_2Lines_FullDuplex,
		 .SPI_DataSize = SPI_DataSize_8b,
		 .SPI_NSS = SPI_NSS_Soft,
		 .SPI_FirstBit = SPI_FirstBit_MSB,
		 .SPI_CRCPolynomial = 7,
		 .SPI_CPOL = SPI_CPOL_High,
		 .SPI_CPHA = SPI_CPHA_2Edge,
		 .SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_32,	//@ APB2 PCLK1 82MHz / 32 == 2.6MHz
		 },
	.use_crc = false,
	.dma = {
		.irq = {
			// Note this is the stream ID that triggers interrupts (in this case RX)
			.flags =
			(DMA_IT_TCIF0 | DMA_IT_TEIF0 | DMA_IT_HTIF0),
			.init = {
				 .NVIC_IRQChannel = DMA2_Stream0_IRQn,
				 .NVIC_IRQChannelPreemptionPriority =
				 PIOS_IRQ_PRIO_HIGH,
				 .NVIC_IRQChannelSubPriority = 0,
				 .NVIC_IRQChannelCmd = ENABLE,
				 },
			},

		.rx = {
		       .channel = DMA2_Stream0,
		       .init = {
				.DMA_Channel = DMA_Channel_3,
				.DMA_PeripheralBaseAddr =
				(uint32_t) & (SPI1->DR),
				.DMA_DIR = DMA_DIR_PeripheralToMemory,
				.DMA_PeripheralInc =
				DMA_PeripheralInc_Disable,
				.DMA_MemoryInc = DMA_MemoryInc_Enable,
				.DMA_PeripheralDataSize =
				DMA_PeripheralDataSize_Byte,
				.DMA_MemoryDataSize =
				DMA_MemoryDataSize_Byte,
				.DMA_Mode = DMA_Mode_Normal,
				.DMA_Priority = DMA_Priority_Medium,
				//TODO: Enable FIFO
				.DMA_FIFOMode = DMA_FIFOMode_Disable,
				.DMA_FIFOThreshold =
				DMA_FIFOThreshold_Full,
				.DMA_MemoryBurst = DMA_MemoryBurst_Single,
				.DMA_PeripheralBurst =
				DMA_PeripheralBurst_Single,
				},
		       },
		.tx = {
		       .channel = DMA2_Stream3,
		       .init = {
				.DMA_Channel = DMA_Channel_3,
				.DMA_PeripheralBaseAddr =
				(uint32_t) & (SPI1->DR),
				.DMA_DIR = DMA_DIR_MemoryToPeripheral,
				.DMA_PeripheralInc =
				DMA_PeripheralInc_Disable,
				.DMA_MemoryInc = DMA_MemoryInc_Enable,
				.DMA_PeripheralDataSize =
				DMA_PeripheralDataSize_Byte,
				.DMA_MemoryDataSize =
				DMA_MemoryDataSize_Byte,
				.DMA_Mode = DMA_Mode_Normal,
				.DMA_Priority = DMA_Priority_Medium,
				.DMA_FIFOMode = DMA_FIFOMode_Disable,
				.DMA_FIFOThreshold =
				DMA_FIFOThreshold_Full,
				.DMA_MemoryBurst = DMA_MemoryBurst_Single,
				.DMA_PeripheralBurst =
				DMA_PeripheralBurst_Single,
				},
		       },
		},
	.sclk = {
		 .gpio = GPIOA,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_5,
			  .GPIO_Speed = GPIO_Speed_100MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 .pin_source = GPIO_PinSource5,
		 },
	.miso = {
		 .gpio = GPIOA,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_6,
			  .GPIO_Speed = GPIO_Speed_100MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 .pin_source = GPIO_PinSource6,
		 },
	.mosi = {
		 .gpio = GPIOA,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_7,
			  .GPIO_Speed = GPIO_Speed_100MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 .pin_source = GPIO_PinSource7,
		 },
	.slave_count = 1,
	.ssel = {{
		  .gpio = GPIOC,
		  .init = {
			   .GPIO_Pin = GPIO_Pin_4,
			   .GPIO_Speed = GPIO_Speed_100MHz,
			   .GPIO_Mode = GPIO_Mode_OUT,
			   .GPIO_OType = GPIO_OType_PP,
			   .GPIO_PuPd = GPIO_PuPd_UP},
		  }},
};

uint32_t pios_spi_gyro_accel_id;
void PIOS_SPI_gyro_accel_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_SPI_IRQ_Handler(pios_spi_gyro_accel_id);
}

#endif /* PIOS_INCLUDE_SPI */

#if defined(PIOS_INCLUDE_I2C)

#include <pios_i2c_priv.h>

/*
 * I2C Adapters
 */

void PIOS_I2C_usart1_ev_irq_handler(void);
void PIOS_I2C_usart1_er_irq_handler(void);
void I2C1_EV_IRQHandler()
    __attribute__ ((alias("PIOS_I2C_usart1_ev_irq_handler")));
void I2C1_ER_IRQHandler()
    __attribute__ ((alias("PIOS_I2C_usart1_er_irq_handler")));

static const struct pios_i2c_adapter_cfg pios_i2c_usart1_adapter_cfg = {
	.regs = I2C1,
	.remap = GPIO_AF_I2C1,
	.init = {
		 .I2C_Mode = I2C_Mode_I2C,
		 .I2C_OwnAddress1 = 0,
		 .I2C_Ack = I2C_Ack_Enable,
		 .I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit,
		 .I2C_DutyCycle = I2C_DutyCycle_2,
		 .I2C_ClockSpeed = 400000,	/* bits/s */
		 },
	.transfer_timeout_ms = 50,
	.scl = {
		.gpio = GPIOB,
		.init = {
			 .GPIO_Pin = GPIO_Pin_6,
			 .GPIO_Mode = GPIO_Mode_AF,
			 .GPIO_Speed = GPIO_Speed_50MHz,
			 .GPIO_OType = GPIO_OType_OD,
			 .GPIO_PuPd = GPIO_PuPd_NOPULL,
			 },
		.pin_source = GPIO_PinSource6,
		},
	.sda = {
		.gpio = GPIOB,
		.init = {
			 .GPIO_Pin = GPIO_Pin_7,
			 .GPIO_Mode = GPIO_Mode_AF,
			 .GPIO_Speed = GPIO_Speed_50MHz,
			 .GPIO_OType = GPIO_OType_OD,
			 .GPIO_PuPd = GPIO_PuPd_NOPULL,
			 },
		.pin_source = GPIO_PinSource7,
		},
	.event = {
		  .flags = 0,	/* FIXME: check this */
		  .init = {
			   .NVIC_IRQChannel = I2C1_EV_IRQn,
			   .NVIC_IRQChannelPreemptionPriority =
			   PIOS_IRQ_PRIO_HIGHEST,
			   .NVIC_IRQChannelSubPriority = 0,
			   .NVIC_IRQChannelCmd = ENABLE,
			   },
		  },
	.error = {
		  .flags = 0,	/* FIXME: check this */
		  .init = {
			   .NVIC_IRQChannel = I2C1_ER_IRQn,
			   .NVIC_IRQChannelPreemptionPriority =
			   PIOS_IRQ_PRIO_HIGHEST,
			   .NVIC_IRQChannelSubPriority = 0,
			   .NVIC_IRQChannelCmd = ENABLE,
			   },
		  },
};

uint32_t pios_i2c_usart1_adapter_id;
void PIOS_I2C_usart1_ev_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_I2C_EV_IRQ_Handler(pios_i2c_usart1_adapter_id);
}

void PIOS_I2C_usart1_er_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_I2C_ER_IRQ_Handler(pios_i2c_usart1_adapter_id);
}

void PIOS_I2C_usart3_ev_irq_handler(void);
void PIOS_I2C_usart3_er_irq_handler(void);
void I2C2_EV_IRQHandler()
    __attribute__ ((alias("PIOS_I2C_usart3_ev_irq_handler")));
void I2C2_ER_IRQHandler()
    __attribute__ ((alias("PIOS_I2C_usart3_er_irq_handler")));

static const struct pios_i2c_adapter_cfg pios_i2c_usart3_adapter_cfg = {
	.regs = I2C2,
	.remap = GPIO_AF_I2C2,
	.init = {
		 .I2C_Mode = I2C_Mode_I2C,
		 .I2C_OwnAddress1 = 0,
		 .I2C_Ack = I2C_Ack_Enable,
		 .I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit,
		 .I2C_DutyCycle = I2C_DutyCycle_2,
		 .I2C_ClockSpeed = 400000,	/* bits/s */
		 },
	.transfer_timeout_ms = 50,
	.scl = {
		.gpio = GPIOB,
		.init = {
			 .GPIO_Pin = GPIO_Pin_10,
			 .GPIO_Mode = GPIO_Mode_AF,
			 .GPIO_Speed = GPIO_Speed_50MHz,
			 .GPIO_OType = GPIO_OType_OD,
			 .GPIO_PuPd = GPIO_PuPd_NOPULL,
			 },
		.pin_source = GPIO_PinSource10,
		},
	.sda = {
		.gpio = GPIOB,
		.init = {
			 .GPIO_Pin = GPIO_Pin_11,
			 .GPIO_Mode = GPIO_Mode_AF,
			 .GPIO_Speed = GPIO_Speed_50MHz,
			 .GPIO_OType = GPIO_OType_OD,
			 .GPIO_PuPd = GPIO_PuPd_NOPULL,
			 },
		.pin_source = GPIO_PinSource11,
		},
	.event = {
		  .flags = 0,	/* FIXME: check this */
		  .init = {
			   .NVIC_IRQChannel = I2C2_EV_IRQn,
			   .NVIC_IRQChannelPreemptionPriority =
			   PIOS_IRQ_PRIO_HIGHEST,
			   .NVIC_IRQChannelSubPriority = 0,
			   .NVIC_IRQChannelCmd = ENABLE,
			   },
		  },
	.error = {
		  .flags = 0,	/* FIXME: check this */
		  .init = {
			   .NVIC_IRQChannel = I2C2_ER_IRQn,
			   .NVIC_IRQChannelPreemptionPriority =
			   PIOS_IRQ_PRIO_HIGHEST,
			   .NVIC_IRQChannelSubPriority = 0,
			   .NVIC_IRQChannelCmd = ENABLE,
			   },
		  },
};

uint32_t pios_i2c_usart3_adapter_id;
void PIOS_I2C_usart3_ev_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_I2C_EV_IRQ_Handler(pios_i2c_usart3_adapter_id);
}

void PIOS_I2C_usart3_er_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_I2C_ER_IRQ_Handler(pios_i2c_usart3_adapter_id);
}

void PIOS_I2C_internal_ev_irq_handler(void);
void PIOS_I2C_internal_er_irq_handler(void);
void I2C3_EV_IRQHandler()
    __attribute__ ((alias("PIOS_I2C_internal_ev_irq_handler")));
void I2C3_ER_IRQHandler()
    __attribute__ ((alias("PIOS_I2C_internal_er_irq_handler")));

static const struct pios_i2c_adapter_cfg pios_i2c_internal_adapter_cfg = {
	.regs = I2C3,
	.remap = GPIO_AF_I2C3,
	.init = {
		 .I2C_Mode = I2C_Mode_I2C,
		 .I2C_OwnAddress1 = 0,
		 .I2C_Ack = I2C_Ack_Enable,
		 .I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit,
		 .I2C_DutyCycle = I2C_DutyCycle_2,
		 .I2C_ClockSpeed = 400000,	/* bits/s */
		 },
	.transfer_timeout_ms = 50,
	.scl = {
		.gpio = GPIOA,
		.init = {
			 .GPIO_Pin = GPIO_Pin_8,
			 .GPIO_Mode = GPIO_Mode_AF,
			 .GPIO_Speed = GPIO_Speed_50MHz,
			 .GPIO_OType = GPIO_OType_OD,
			 .GPIO_PuPd = GPIO_PuPd_NOPULL,
			 },
		.pin_source = GPIO_PinSource8,
		},
	.sda = {
		.gpio = GPIOC,
		.init = {
			 .GPIO_Pin = GPIO_Pin_9,
			 .GPIO_Mode = GPIO_Mode_AF,
			 .GPIO_Speed = GPIO_Speed_50MHz,
			 .GPIO_OType = GPIO_OType_OD,
			 .GPIO_PuPd = GPIO_PuPd_NOPULL,
			 },
		.pin_source = GPIO_PinSource9,
		},
	.event = {
		  .flags = 0,	/* FIXME: check this */
		  .init = {
			   .NVIC_IRQChannel = I2C3_EV_IRQn,
			   .NVIC_IRQChannelPreemptionPriority =
			   PIOS_IRQ_PRIO_HIGHEST,
			   .NVIC_IRQChannelSubPriority = 0,
			   .NVIC_IRQChannelCmd = ENABLE,
			   },
		  },
	.error = {
		  .flags = 0,	/* FIXME: check this */
		  .init = {
			   .NVIC_IRQChannel = I2C3_ER_IRQn,
			   .NVIC_IRQChannelPreemptionPriority =
			   PIOS_IRQ_PRIO_HIGHEST,
			   .NVIC_IRQChannelSubPriority = 0,
			   .NVIC_IRQChannelCmd = ENABLE,
			   },
		  },
};

uint32_t pios_i2c_internal_adapter_id;
void PIOS_I2C_internal_ev_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_I2C_EV_IRQ_Handler(pios_i2c_internal_adapter_id);
}

void PIOS_I2C_internal_er_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_I2C_ER_IRQ_Handler(pios_i2c_internal_adapter_id);
}

#endif /* PIOS_INCLUDE_I2C */

#if defined(PIOS_INCLUDE_FLASH)
#include "pios_flashfs_logfs_priv.h"

static const struct flashfs_logfs_cfg flashfs_settings_cfg = {
	.fs_magic = 0x3bb141cf,
	.arena_size = 0x00010000,	/* 256 * slot size */
	.slot_size = 0x00000100,	/* 256 bytes */
};

static const struct flashfs_logfs_cfg flashfs_waypoints_cfg = {
	.fs_magic = 0x9a365a64,
	.arena_size = 0x00010000,	/* 1024 * slot size */
	.slot_size = 0x00000040,	/* 64 bytes */
};

#if defined(PIOS_INCLUDE_FLASH_JEDEC)
#include "pios_flash_jedec_priv.h"

static const struct pios_flash_jedec_cfg flash_mx25_cfg = {
	.expect_manufacturer = JEDEC_MANUFACTURER_MACRONIX,
	.expect_memorytype = 0x20,
	.expect_capacity = 0x16,
	.sector_erase = 0x20,
};
#endif /* PIOS_INCLUDE_FLASH_JEDEC */

#if defined(PIOS_INCLUDE_FLASH_INTERNAL)
#include "pios_flash_internal_priv.h"

static const struct pios_flash_internal_cfg flash_internal_cfg = {
};
#endif /* PIOS_INCLUDE_FLASH_INTERNAL */

#include "pios_flash_priv.h"

#if defined(PIOS_INCLUDE_FLASH_INTERNAL)
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
	.driver = &pios_internal_flash_driver,
	.chip_id = &pios_internal_flash_id,
	.page_size = 16,	/* 128-bit rows */
	.sector_blocks = stm32f4_sectors,
	.num_blocks = NELEMENTS(stm32f4_sectors),
};
#endif /* PIOS_INCLUDE_FLASH_INTERNAL */

#if defined(PIOS_INCLUDE_FLASH_JEDEC)
static const struct pios_flash_sector_range mx25_sectors[] = {
	{
	 .base_sector = 0,
	 .last_sector = 1023,
	 .sector_size = FLASH_SECTOR_4KB,
	 },
};

uintptr_t pios_external_flash_id;
static const struct pios_flash_chip pios_flash_chip_external = {
	.driver = &pios_jedec_flash_driver,
	.chip_id = &pios_external_flash_id,
	.page_size = 256,
	.sector_blocks = mx25_sectors,
	.num_blocks = NELEMENTS(mx25_sectors),
};
#endif /* PIOS_INCLUDE_FLASH_JEDEC */

static const struct pios_flash_partition pios_flash_partition_table[] = {
#if defined(PIOS_INCLUDE_FLASH_INTERNAL)
	{
	 .label = FLASH_PARTITION_LABEL_BL,
	 .chip_desc = &pios_flash_chip_internal,
	 .first_sector = 0,
	 .last_sector = 1,
	 .chip_offset = 0,
	 .size = (1 - 0 + 1) * FLASH_SECTOR_16KB,
	 },

	/* NOTE: sectors 2-4 of the internal flash are currently unallocated */

	{
	 .label = FLASH_PARTITION_LABEL_FW,
	 .chip_desc = &pios_flash_chip_internal,
	 .first_sector = 5,
	 .last_sector = 6,
	 .chip_offset = (4 * FLASH_SECTOR_16KB) + (1 * FLASH_SECTOR_64KB),
	 .size = (6 - 5 + 1) * FLASH_SECTOR_128KB,
	 },

	/* NOTE: sectors 7-11 of the internal flash are currently unallocated */

#endif /* PIOS_INCLUDE_FLASH_INTERNAL */

#if defined(PIOS_INCLUDE_FLASH_JEDEC)
	{
	 .label = FLASH_PARTITION_LABEL_SETTINGS,
	 .chip_desc = &pios_flash_chip_external,
	 .first_sector = 0,
	 .last_sector = 511,
	 .chip_offset = 0,
	 .size = (511 - 0 + 1) * FLASH_SECTOR_4KB,
	 },

	{
	 .label = FLASH_PARTITION_LABEL_WAYPOINTS,
	 .chip_desc = &pios_flash_chip_external,
	 .first_sector = 512,
	 .last_sector = 1023,
	 .chip_offset = (512 * FLASH_SECTOR_4KB),
	 .size = (1023 - 512 + 1) * FLASH_SECTOR_4KB,
	 },
#endif /* PIOS_INCLUDE_FLASH_JEDEC */
};

const struct pios_flash_partition
    *PIOS_BOARD_HW_DEFS_GetPartitionTable(uint32_t board_revision,
					  uint32_t * num_partitions)
{
	PIOS_Assert(num_partitions);

	*num_partitions = NELEMENTS(pios_flash_partition_table);
	return pios_flash_partition_table;
}

#endif /* PIOS_INCLUDE_FLASH */

#if defined(PIOS_INCLUDE_USART)

#include "pios_usart_priv.h"

#if defined(PIOS_INCLUDE_DSM)
/*
 * Spektrum/JR DSM USART
 */
#include <pios_dsm_priv.h>

static const struct pios_dsm_cfg pios_usart1_dsm_aux_cfg = {
	.bind = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_7,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_OUT,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 },
};

static const struct pios_dsm_cfg pios_usart2_dsm_aux_cfg = {
	.bind = {
		 .gpio = GPIOA,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_3,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_OUT,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 },
};

static const struct pios_dsm_cfg pios_usart3_dsm_aux_cfg = {
	.bind = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_11,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_OUT,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 },
};

static const struct pios_dsm_cfg pios_usart4_dsm_aux_cfg = {
	.bind = {
		 .gpio = GPIOC,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_11,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_OUT,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 },
};

static const struct pios_dsm_cfg pios_usart5_dsm_aux_cfg = {
	.bind = {
		 .gpio = GPIOD,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_2,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_OUT,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_NOPULL},
		 },
};

#endif /* PIOS_INCLUDE_DSM */

#if defined(PIOS_INCLUDE_HSUM)
/*
 * Graupner HoTT SUMD/SUMH USART
 */
#include <pios_hsum_priv.h>

#endif /* PIOS_INCLUDE_HSUM */

#if (defined(PIOS_INCLUDE_DSM) || defined(PIOS_INCLUDE_HSUM))
/*
 * Spektrum/JR DSM or Graupner HoTT SUMD/SUMH USART
 */

static const struct pios_usart_cfg pios_usart1_dsm_hsum_cfg = {
	.regs = USART1,
	.remap = GPIO_AF_USART1,
	.init = {
		 .USART_BaudRate = 115200,
		 .USART_WordLength = USART_WordLength_8b,
		 .USART_Parity = USART_Parity_No,
		 .USART_StopBits = USART_StopBits_1,
		 .USART_HardwareFlowControl =
		 USART_HardwareFlowControl_None,
		 .USART_Mode = USART_Mode_Rx,
		 },
	.irq = {
		.init = {
			 .NVIC_IRQChannel = USART1_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.rx = {
	       .gpio = GPIOB,
	       .init = {
			.GPIO_Pin = GPIO_Pin_7,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource7,
	       },
};

static const struct pios_usart_cfg pios_usart2_dsm_hsum_cfg = {
	.regs = USART2,
	.remap = GPIO_AF_USART2,
	.init = {
		 .USART_BaudRate = 115200,
		 .USART_WordLength = USART_WordLength_8b,
		 .USART_Parity = USART_Parity_No,
		 .USART_StopBits = USART_StopBits_1,
		 .USART_HardwareFlowControl =
		 USART_HardwareFlowControl_None,
		 .USART_Mode = USART_Mode_Rx,
		 },
	.irq = {
		.init = {
			 .NVIC_IRQChannel = USART2_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.rx = {
	       .gpio = GPIOA,
	       .init = {
			.GPIO_Pin = GPIO_Pin_3,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource3,
	       },
};

static const struct pios_usart_cfg pios_usart3_dsm_hsum_cfg = {
	.regs = USART3,
	.remap = GPIO_AF_USART3,
	.init = {
		 .USART_BaudRate = 115200,
		 .USART_WordLength = USART_WordLength_8b,
		 .USART_Parity = USART_Parity_No,
		 .USART_StopBits = USART_StopBits_1,
		 .USART_HardwareFlowControl =
		 USART_HardwareFlowControl_None,
		 .USART_Mode = USART_Mode_Rx,
		 },
	.irq = {
		.init = {
			 .NVIC_IRQChannel = USART3_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.rx = {
	       .gpio = GPIOB,
	       .init = {
			.GPIO_Pin = GPIO_Pin_11,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource11,
	       },
};

static const struct pios_usart_cfg pios_usart4_dsm_hsum_cfg = {
	.regs = UART4,
	.remap = GPIO_AF_UART4,
	.init = {
		 .USART_BaudRate = 115200,
		 .USART_WordLength = USART_WordLength_8b,
		 .USART_Parity = USART_Parity_No,
		 .USART_StopBits = USART_StopBits_1,
		 .USART_HardwareFlowControl =
		 USART_HardwareFlowControl_None,
		 .USART_Mode = USART_Mode_Rx,
		 },
	.irq = {
		.init = {
			 .NVIC_IRQChannel = UART4_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.rx = {
	       .gpio = GPIOC,
	       .init = {
			.GPIO_Pin = GPIO_Pin_11,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource11,
	       },
};

#endif /* PIOS_INCLUDE_DSM || PIOS_INCLUDE_HSUM */

#if defined(PIOS_INCLUDE_SBUS)
/*
 * S.Bus USART
 */
#include <pios_sbus_priv.h>

static const struct pios_usart_cfg pios_usart2_sbus_cfg = {
	.regs = USART2,
	.remap = GPIO_AF_USART2,
	.init = {
		 .USART_BaudRate = 100000,
		 .USART_WordLength = USART_WordLength_8b,
		 .USART_Parity = USART_Parity_Even,
		 .USART_StopBits = USART_StopBits_2,
		 .USART_HardwareFlowControl =
		 USART_HardwareFlowControl_None,
		 .USART_Mode = USART_Mode_Rx,
		 },
	.irq = {
		.init = {
			 .NVIC_IRQChannel = USART2_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.rx = {
	       .gpio = GPIOA,
	       .init = {
			.GPIO_Pin = GPIO_Pin_3,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource3,
	       },
};

static const struct pios_sbus_cfg pios_usart2_sbus_aux_cfg = {
	/* Inverter configuration */
	.inv = {
		.gpio = GPIOB,
		.init = {
			 .GPIO_Pin = GPIO_Pin_2,
			 .GPIO_Speed = GPIO_Speed_2MHz,
			 .GPIO_Mode = GPIO_Mode_OUT,
			 .GPIO_OType = GPIO_OType_PP,
			 .GPIO_PuPd = GPIO_PuPd_NOPULL},
		},
	.gpio_inv_enable = Bit_SET,
};

#endif /* PIOS_INCLUDE_SBUS */

static const struct pios_usart_cfg pios_usart1_cfg = {
	.regs = USART1,
	.remap = GPIO_AF_USART1,
	.init = {
		 .USART_BaudRate = 57600,
		 .USART_WordLength = USART_WordLength_8b,
		 .USART_Parity = USART_Parity_No,
		 .USART_StopBits = USART_StopBits_1,
		 .USART_HardwareFlowControl =
		 USART_HardwareFlowControl_None,
		 .USART_Mode = USART_Mode_Rx | USART_Mode_Tx,
		 },
	.irq = {
		.init = {
			 .NVIC_IRQChannel = USART1_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.rx = {
	       .gpio = GPIOB,
	       .init = {
			.GPIO_Pin = GPIO_Pin_7,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource7,
	       },
	.tx = {
	       .gpio = GPIOB,
	       .init = {
			.GPIO_Pin = GPIO_Pin_6,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource6,
	       },
};

static const struct pios_usart_cfg pios_usart2_cfg = {
	.regs = USART2,
	.remap = GPIO_AF_USART2,
	.init = {
		 .USART_BaudRate = 57600,
		 .USART_WordLength = USART_WordLength_8b,
		 .USART_Parity = USART_Parity_No,
		 .USART_StopBits = USART_StopBits_1,
		 .USART_HardwareFlowControl =
		 USART_HardwareFlowControl_None,
		 .USART_Mode = USART_Mode_Rx | USART_Mode_Tx,
		 },
	.irq = {
		.init = {
			 .NVIC_IRQChannel = USART2_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.rx = {
	       .gpio = GPIOA,
	       .init = {
			.GPIO_Pin = GPIO_Pin_3,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource3,
	       },
	.tx = {
	       .gpio = GPIOA,
	       .init = {
			.GPIO_Pin = GPIO_Pin_2,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource2,
	       },
};

static const struct pios_usart_cfg pios_usart3_cfg = {
	.regs = USART3,
	.remap = GPIO_AF_USART3,
	.init = {
		 .USART_BaudRate = 57600,
		 .USART_WordLength = USART_WordLength_8b,
		 .USART_Parity = USART_Parity_No,
		 .USART_StopBits = USART_StopBits_1,
		 .USART_HardwareFlowControl =
		 USART_HardwareFlowControl_None,
		 .USART_Mode = USART_Mode_Rx | USART_Mode_Tx,
		 },
	.irq = {
		.init = {
			 .NVIC_IRQChannel = USART3_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.rx = {
	       .gpio = GPIOB,
	       .init = {
			.GPIO_Pin = GPIO_Pin_11,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource11,
	       },
	.tx = {
	       .gpio = GPIOB,
	       .init = {
			.GPIO_Pin = GPIO_Pin_10,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource10,
	       },
};

static const struct pios_usart_cfg pios_usart4_cfg = {
	.regs = UART4,
	.remap = GPIO_AF_UART4,
	.init = {
		 .USART_BaudRate = 57600,
		 .USART_WordLength = USART_WordLength_8b,
		 .USART_Parity = USART_Parity_No,
		 .USART_StopBits = USART_StopBits_1,
		 .USART_HardwareFlowControl =
		 USART_HardwareFlowControl_None,
		 .USART_Mode = USART_Mode_Rx | USART_Mode_Tx,
		 },
	.irq = {
		.init = {
			 .NVIC_IRQChannel = UART4_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.rx = {
	       .gpio = GPIOC,
	       .init = {
			.GPIO_Pin = GPIO_Pin_11,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource11,
	       },
	.tx = {
	       .gpio = GPIOC,
	       .init = {
			.GPIO_Pin = GPIO_Pin_10,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_AF,
			.GPIO_OType = GPIO_OType_PP,
			.GPIO_PuPd = GPIO_PuPd_UP},
	       .pin_source = GPIO_PinSource10,
	       },
};

#endif /* PIOS_INCLUDE_USART */

#if defined(PIOS_INCLUDE_COM)

#include "pios_com_priv.h"

#endif /* PIOS_INCLUDE_COM */

#if defined(PIOS_INCLUDE_RTC)
/*
 * Realtime Clock (RTC)
 */
#include <pios_rtc_priv.h>

void PIOS_RTC_IRQ_Handler(void);
void RTC_WKUP_IRQHandler() __attribute__ ((alias("PIOS_RTC_IRQ_Handler")));
static const struct pios_rtc_cfg pios_rtc_main_cfg = {
	.clksrc = RCC_RTCCLKSource_HSE_Div16,	// Divide 8 Mhz crystal down to 1
	// For some reason it's acting like crystal is 16 Mhz.  This clock is then divided
	// by another 16 to give a nominal 62.5 khz clock
	.prescaler = 100,	// Every 100 cycles gives 625 Hz
	.irq = {
		.init = {
			 .NVIC_IRQChannel = RTC_WKUP_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
};

void PIOS_RTC_IRQ_Handler(void)
{
	PIOS_RTC_irq_handler();
}

#endif

#include "pios_tim_priv.h"

//Timers used for inputs (1, 2, 5, 8)

static const TIM_TimeBaseInitTypeDef tim_2_5_time_base = {
	.TIM_Prescaler = (PIOS_PERIPHERAL_APB1_CLOCK / 1000000) - 1,
	.TIM_ClockDivision = TIM_CKD_DIV1,
	.TIM_CounterMode = TIM_CounterMode_Up,
	.TIM_Period = 0xFFFF,
	.TIM_RepetitionCounter = 0x0000,
};

static const TIM_TimeBaseInitTypeDef tim_1_8_time_base = {
	.TIM_Prescaler = (PIOS_PERIPHERAL_APB2_CLOCK / 1000000) - 1,
	.TIM_ClockDivision = TIM_CKD_DIV1,
	.TIM_CounterMode = TIM_CounterMode_Up,
	.TIM_Period = 0xFFFF,
	.TIM_RepetitionCounter = 0x0000,
};

static const struct pios_tim_clock_cfg tim_2_cfg = {
	.timer = TIM2,
	.time_base_init = &tim_2_5_time_base,
	.irq = {
		.init = {
			 .NVIC_IRQChannel = TIM2_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
};

static const struct pios_tim_clock_cfg tim_5_cfg = {
	.timer = TIM5,
	.time_base_init = &tim_2_5_time_base,
	.irq = {
		.init = {
			 .NVIC_IRQChannel = TIM5_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
};

static const struct pios_tim_clock_cfg tim_1_cfg = {
	.timer = TIM1,
	.time_base_init = &tim_1_8_time_base,
	.irq = {
		.init = {
			 .NVIC_IRQChannel = TIM1_CC_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.irq2 = {
		 .init = {
			  .NVIC_IRQChannel = TIM1_UP_TIM10_IRQn,
			  .NVIC_IRQChannelPreemptionPriority =
			  PIOS_IRQ_PRIO_MID,
			  .NVIC_IRQChannelSubPriority = 0,
			  .NVIC_IRQChannelCmd = ENABLE,
			  },
		 },
};

static const struct pios_tim_clock_cfg tim_8_cfg = {
	.timer = TIM8,
	.time_base_init = &tim_1_8_time_base,
	.irq = {
		.init = {
			 .NVIC_IRQChannel = TIM8_CC_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.irq2 = {
		 .init = {
			  .NVIC_IRQChannel = TIM8_UP_TIM13_IRQn,
			  .NVIC_IRQChannelPreemptionPriority =
			  PIOS_IRQ_PRIO_MID,
			  .NVIC_IRQChannelSubPriority = 0,
			  .NVIC_IRQChannelCmd = ENABLE,
			  },
		 },
};

// Timers used for outputs (3, 10, 11, 12)

// Set up timers that only have inputs on APB1
static const TIM_TimeBaseInitTypeDef tim_3_12_time_base = {
	.TIM_Prescaler = (PIOS_PERIPHERAL_APB1_CLOCK / 1000000) - 1,
	.TIM_ClockDivision = TIM_CKD_DIV1,
	.TIM_CounterMode = TIM_CounterMode_Up,
	.TIM_Period = ((1000000 / PIOS_SERVO_UPDATE_HZ) - 1),
	.TIM_RepetitionCounter = 0x0000,
};

// Set up timers that only have inputs on APB2
static const TIM_TimeBaseInitTypeDef tim_10_11_time_base = {
	.TIM_Prescaler = (PIOS_PERIPHERAL_APB2_CLOCK / 1000000) - 1,
	.TIM_ClockDivision = TIM_CKD_DIV1,
	.TIM_CounterMode = TIM_CounterMode_Up,
	.TIM_Period = ((1000000 / PIOS_SERVO_UPDATE_HZ) - 1),
	.TIM_RepetitionCounter = 0x0000,
};

static const struct pios_tim_clock_cfg tim_3_cfg = {
	.timer = TIM3,
	.time_base_init = &tim_3_12_time_base,
	.irq = {
		.init = {
			 .NVIC_IRQChannel = TIM3_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
};

static const struct pios_tim_clock_cfg tim_10_cfg = {
	.timer = TIM10,
	.time_base_init = &tim_10_11_time_base,
	.irq = {
		.init = {
			 .NVIC_IRQChannel = TIM1_UP_TIM10_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
};

static const struct pios_tim_clock_cfg tim_11_cfg = {
	.timer = TIM11,
	.time_base_init = &tim_10_11_time_base,
	.irq = {
		.init = {
			 .NVIC_IRQChannel = TIM1_TRG_COM_TIM11_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
};

static const struct pios_tim_clock_cfg tim_12_cfg = {
	.timer = TIM12,
	.time_base_init = &tim_3_12_time_base,
	.irq = {
		.init = {
			 .NVIC_IRQChannel = TIM8_BRK_TIM12_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_MID,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
};

/**
 * Pios servo configuration structures
 */

/*
 * 	OUTPUTS
	1: TIM3_CH1 (PB4)
	2: TIM3_CH2 (PB5)
	3: TIM3_CH3 (PB0)
	4: TIM3_CH4 (PB1)
	5: TIM12_CH1 (PB14)
	6: TIM12_CH2 (PB15)
	7: TIM10_CH1 (PB8)
	8: TIM11_CH1 (PB9)
 */
static const struct pios_tim_channel pios_tim_servoport_all_pins[] = {
	{
	 .timer = TIM3,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM3,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_4,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
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
			  .GPIO_Pin = GPIO_Pin_5,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource5,
		 },
	 },
	{
	 .timer = TIM3,
	 .timer_chan = TIM_Channel_3,
	 .remap = GPIO_AF_TIM3,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_0,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource0,
		 },
	 },
	{
	 .timer = TIM3,
	 .timer_chan = TIM_Channel_4,
	 .remap = GPIO_AF_TIM3,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_1,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource1,
		 },
	 },
	{
	 .timer = TIM12,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM12,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_14,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource14,
		 },
	 },
	{
	 .timer = TIM12,
	 .timer_chan = TIM_Channel_2,
	 .remap = GPIO_AF_TIM12,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_15,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource15,
		 },
	 },
	{
	 .timer = TIM10,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM10,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_8,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource8,
		 },
	 },
	{
	 .timer = TIM11,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM11,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_9,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource9,
		 },
	 },
};

/*
 * 	OUTPUTS with extra outputs on receiverport
	1:  TIM3_CH1  (PB4)
	2:  TIM3_CH2  (PB5)
	3:  TIM3_CH3  (PB0)
	4:  TIM3_CH4  (PB1)
	5:  TIM12_CH1 (PB14)
	6:  TIM12_CH2 (PB15)
	7:  TIM10_CH1 (PB8)
	8:  TIM11_CH1 (PB9)
	9:  TIM8_CH1  (PC6)		(IN2)
	10: TIM8_CH2  (PC7)		(IN3)
	11: TIM8_CH3  (PC8)		(IN4)
	12: TIM2_CH1  (PA15)	(IN5)
	13: TIM2_CH2  (PB3)		(IN6)
	14: TIM5_CH1  (PA0)		(IN7)
	15: TIM5_CH2  (PA1)		(IN8)
 */

static const struct pios_tim_channel pios_tim_servoport_rcvrport_pins[] = {
	{
	 .timer = TIM3,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM3,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_4,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
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
			  .GPIO_Pin = GPIO_Pin_5,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource5,
		 },
	 },
	{
	 .timer = TIM3,
	 .timer_chan = TIM_Channel_3,
	 .remap = GPIO_AF_TIM3,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_0,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource0,
		 },
	 },
	{
	 .timer = TIM3,
	 .timer_chan = TIM_Channel_4,
	 .remap = GPIO_AF_TIM3,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_1,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource1,
		 },
	 },
	{
	 .timer = TIM12,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM12,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_14,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource14,
		 },
	 },
	{
	 .timer = TIM12,
	 .timer_chan = TIM_Channel_2,
	 .remap = GPIO_AF_TIM12,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_15,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource15,
		 },
	 },
	{
	 .timer = TIM10,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM10,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_8,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource8,
		 },
	 },
	{
	 .timer = TIM11,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM11,
	 .pin = {
		 .gpio = GPIOB,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_9,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource9,
		 },
	 },
	{
	 .timer = TIM8,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM8,
	 .pin = {
		 .gpio = GPIOC,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_6,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource6,
		 },
	 },
	{
	 .timer = TIM8,
	 .timer_chan = TIM_Channel_2,
	 .remap = GPIO_AF_TIM8,
	 .pin = {
		 .gpio = GPIOC,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_7,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource7,
		 },
	 },
	{
	 .timer = TIM8,
	 .timer_chan = TIM_Channel_3,
	 .remap = GPIO_AF_TIM8,
	 .pin = {
		 .gpio = GPIOC,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_8,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
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
			  .GPIO_Pin = GPIO_Pin_15,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
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
			  .GPIO_Pin = GPIO_Pin_3,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource3,
		 },
	 },
	{
	 .timer = TIM5,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM5,
	 .pin = {
		 .gpio = GPIOA,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_0,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource0,
		 },
	 },
	{
	 .timer = TIM5,
	 .timer_chan = TIM_Channel_2,
	 .remap = GPIO_AF_TIM5,
	 .pin = {
		 .gpio = GPIOA,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_1,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource1,
		 },
	 },
};

#if defined(PIOS_INCLUDE_SERVO) && defined(PIOS_INCLUDE_TIM)
/*
 * Servo outputs
 */
#include <pios_servo_priv.h>

const struct pios_servo_cfg pios_servo_cfg = {
	.tim_oc_init = {
			.TIM_OCMode = TIM_OCMode_PWM1,
			.TIM_OutputState = TIM_OutputState_Enable,
			.TIM_OutputNState = TIM_OutputNState_Disable,
			.TIM_Pulse = PIOS_SERVOS_INITIAL_POSITION,
			.TIM_OCPolarity = TIM_OCPolarity_High,
			.TIM_OCNPolarity = TIM_OCPolarity_High,
			.TIM_OCIdleState = TIM_OCIdleState_Reset,
			.TIM_OCNIdleState = TIM_OCNIdleState_Reset,
			},
	.channels = pios_tim_servoport_all_pins,
	.num_channels = NELEMENTS(pios_tim_servoport_all_pins),
};

const struct pios_servo_cfg pios_servo_with_adc_cfg = {
	.tim_oc_init = {
			.TIM_OCMode = TIM_OCMode_PWM1,
			.TIM_OutputState = TIM_OutputState_Enable,
			.TIM_OutputNState = TIM_OutputNState_Disable,
			.TIM_Pulse = PIOS_SERVOS_INITIAL_POSITION,
			.TIM_OCPolarity = TIM_OCPolarity_High,
			.TIM_OCNPolarity = TIM_OCPolarity_High,
			.TIM_OCIdleState = TIM_OCIdleState_Reset,
			.TIM_OCNIdleState = TIM_OCNIdleState_Reset,
			},
	/* Leave the last two for ADC use */
	.channels = pios_tim_servoport_all_pins,
	.num_channels = NELEMENTS(pios_tim_servoport_all_pins) - 2,
};

const struct pios_servo_cfg pios_servo_with_rcvr_cfg = {
	.tim_oc_init = {
			.TIM_OCMode = TIM_OCMode_PWM1,
			.TIM_OutputState = TIM_OutputState_Enable,
			.TIM_OutputNState = TIM_OutputNState_Disable,
			.TIM_Pulse = PIOS_SERVOS_INITIAL_POSITION,
			.TIM_OCPolarity = TIM_OCPolarity_High,
			.TIM_OCNPolarity = TIM_OCPolarity_High,
			.TIM_OCIdleState = TIM_OCIdleState_Reset,
			.TIM_OCNIdleState = TIM_OCNIdleState_Reset,
			},
	.channels = pios_tim_servoport_rcvrport_pins,
	.num_channels = NELEMENTS(pios_tim_servoport_rcvrport_pins),
};

const struct pios_servo_cfg pios_servo_with_rcvr_with_adc_cfg = {
	.tim_oc_init = {
			.TIM_OCMode = TIM_OCMode_PWM1,
			.TIM_OutputState = TIM_OutputState_Enable,
			.TIM_OutputNState = TIM_OutputNState_Disable,
			.TIM_Pulse = PIOS_SERVOS_INITIAL_POSITION,
			.TIM_OCPolarity = TIM_OCPolarity_High,
			.TIM_OCNPolarity = TIM_OCPolarity_High,
			.TIM_OCIdleState = TIM_OCIdleState_Reset,
			.TIM_OCNIdleState = TIM_OCNIdleState_Reset,
			},
	/* Leave the last two for ADC use */
	.channels = pios_tim_servoport_rcvrport_pins,
	.num_channels = NELEMENTS(pios_tim_servoport_rcvrport_pins) - 2,
};

#endif /* PIOS_INCLUDE_SERVO && PIOS_INCLUDE_TIM */

/*
 * 	INPUTS
	1: TIM1_CH3 (PA10)
	2: TIM8_CH1 (PC6)
	3: TIM8_CH2 (PC7)
	4: TIM8_CH3 (PC8)
	5: TIM2_CH1 (PA15)
	6: TIM2_CH2 (PB3)
	7: TIM5_CH1 (PA0)
	8: TIM5_CH2 (PA1)
 */
static const struct pios_tim_channel pios_tim_rcvrport_all_channels[] = {
	{
	 .timer = TIM1,
	 .timer_chan = TIM_Channel_3,
	 .remap = GPIO_AF_TIM1,
	 .pin = {
		 .gpio = GPIOA,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_10,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource10,
		 },
	 },
	{
	 .timer = TIM8,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM8,
	 .pin = {
		 .gpio = GPIOC,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_6,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource6,
		 },
	 },
	{
	 .timer = TIM8,
	 .timer_chan = TIM_Channel_2,
	 .remap = GPIO_AF_TIM8,
	 .pin = {
		 .gpio = GPIOC,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_7,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource7,
		 },
	 },
	{
	 .timer = TIM8,
	 .timer_chan = TIM_Channel_3,
	 .remap = GPIO_AF_TIM8,
	 .pin = {
		 .gpio = GPIOC,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_8,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
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
			  .GPIO_Pin = GPIO_Pin_15,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
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
			  .GPIO_Pin = GPIO_Pin_3,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource3,
		 },
	 },
	{
	 .timer = TIM5,
	 .timer_chan = TIM_Channel_1,
	 .remap = GPIO_AF_TIM5,
	 .pin = {
		 .gpio = GPIOA,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_0,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource0,
		 },
	 },
	{
	 .timer = TIM5,
	 .timer_chan = TIM_Channel_2,
	 .remap = GPIO_AF_TIM5,
	 .pin = {
		 .gpio = GPIOA,
		 .init = {
			  .GPIO_Pin = GPIO_Pin_1,
			  .GPIO_Speed = GPIO_Speed_2MHz,
			  .GPIO_Mode = GPIO_Mode_AF,
			  .GPIO_OType = GPIO_OType_PP,
			  .GPIO_PuPd = GPIO_PuPd_UP},
		 .pin_source = GPIO_PinSource1,
		 },
	 },
};

/*
 * PWM Inputs
 */
#if defined(PIOS_INCLUDE_PWM) || defined(PIOS_INCLUDE_PPM)
#include <pios_pwm_priv.h>

const struct pios_pwm_cfg pios_pwm_cfg = {
	.tim_ic_init = {
			.TIM_ICPolarity = TIM_ICPolarity_Rising,
			.TIM_ICSelection = TIM_ICSelection_DirectTI,
			.TIM_ICPrescaler = TIM_ICPSC_DIV1,
			.TIM_ICFilter = 0x0,
			},
	.channels = pios_tim_rcvrport_all_channels,
	.num_channels = NELEMENTS(pios_tim_rcvrport_all_channels),
};

const struct pios_pwm_cfg pios_pwm_with_adc_cfg = {
	.tim_ic_init = {
			.TIM_ICPolarity = TIM_ICPolarity_Rising,
			.TIM_ICSelection = TIM_ICSelection_DirectTI,
			.TIM_ICPrescaler = TIM_ICPSC_DIV1,
			.TIM_ICFilter = 0x0,
			},
	/* Leave the last two for ADC use */
	.channels = pios_tim_rcvrport_all_channels,
	.num_channels = NELEMENTS(pios_tim_rcvrport_all_channels) - 2,
};

const struct pios_pwm_cfg pios_pwm_with_ppm_cfg = {
	.tim_ic_init = {
			.TIM_ICPolarity = TIM_ICPolarity_Rising,
			.TIM_ICSelection = TIM_ICSelection_DirectTI,
			.TIM_ICPrescaler = TIM_ICPSC_DIV1,
			.TIM_ICFilter = 0x0,
			},
	/* Leave the first channel for PPM use and use the rest for PWM */
	.channels = &pios_tim_rcvrport_all_channels[1],
	.num_channels = NELEMENTS(pios_tim_rcvrport_all_channels) - 1,
};

const struct pios_pwm_cfg pios_pwm_with_ppm_with_adc_cfg = {
	.tim_ic_init = {
			.TIM_ICPolarity = TIM_ICPolarity_Rising,
			.TIM_ICSelection = TIM_ICSelection_DirectTI,
			.TIM_ICPrescaler = TIM_ICPSC_DIV1,
			.TIM_ICFilter = 0x0,
			},
	/* Leave the first channel for PPM, the last two for ADC use and use the rest for PWM */
	.channels = &pios_tim_rcvrport_all_channels[1],
	.num_channels = NELEMENTS(pios_tim_rcvrport_all_channels) - 1 - 2,
};

#endif

/*
 * PPM Input
 */
#if defined(PIOS_INCLUDE_PPM)
#include <pios_ppm_priv.h>

static const struct pios_ppm_cfg pios_ppm_cfg = {
	.tim_ic_init = {
			.TIM_ICPolarity = TIM_ICPolarity_Rising,
			.TIM_ICSelection = TIM_ICSelection_DirectTI,
			.TIM_ICPrescaler = TIM_ICPSC_DIV1,
			.TIM_ICFilter = 0x0,
			.TIM_Channel = TIM_Channel_3,
			},
	/* Use only the first channel for ppm */
	.channels = &pios_tim_rcvrport_all_channels[0],
	.num_channels = 1,
};

#endif //PPM

#if defined(PIOS_INCLUDE_GCSRCVR)
#include "pios_gcsrcvr_priv.h"
#endif /* PIOS_INCLUDE_GCSRCVR */

#if defined(PIOS_INCLUDE_RCVR)
#include "pios_rcvr_priv.h"
#endif /* PIOS_INCLUDE_RCVR */

#if defined(PIOS_INCLUDE_USB)
#include "pios_usb_priv.h"

static const struct pios_usb_cfg pios_usb_main_cfg = {
	.irq = {
		.init = {
			 .NVIC_IRQChannel = OTG_FS_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_HIGHEST,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.vsense = {
		   .gpio = GPIOA,
		   .init = {
			    .GPIO_Pin = GPIO_Pin_9,
			    .GPIO_Speed = GPIO_Speed_25MHz,
			    .GPIO_Mode = GPIO_Mode_IN,
			    .GPIO_OType = GPIO_OType_OD,
			    .GPIO_PuPd = GPIO_PuPd_NOPULL,
			    },
		   }
};

const struct pios_usb_cfg *PIOS_BOARD_HW_DEFS_GetUsbCfg(uint32_t
							board_revision)
{
	return &pios_usb_main_cfg;
}

#include "pios_usb_board_data_priv.h"
#include "pios_usb_desc_hid_cdc_priv.h"
#include "pios_usb_desc_hid_only_priv.h"
#include "pios_usbhook.h"

#endif /* PIOS_INCLUDE_USB */

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
#endif /* PIOS_INCLUDE_USB_HID && PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_ADC)
#include "pios_adc_priv.h"
#include "pios_internal_adc_priv.h"

void PIOS_ADC_DMA_irq_handler(void);
void DMA2_Stream4_IRQHandler(void)
    __attribute__ ((alias("PIOS_ADC_DMA_irq_handler")));
struct pios_internal_adc_cfg pios_adc_cfg = {
	.adc_dev_master = ADC1,
	.dma = {
		.irq = {
			.flags =
			(DMA_FLAG_TCIF4 | DMA_FLAG_TEIF4 | DMA_FLAG_HTIF4),
			.init = {
				 .NVIC_IRQChannel = DMA2_Stream4_IRQn,
				 .NVIC_IRQChannelPreemptionPriority =
				 PIOS_IRQ_PRIO_LOW,
				 .NVIC_IRQChannelSubPriority = 0,
				 .NVIC_IRQChannelCmd = ENABLE,
				 },
			},
		.rx = {
		       .channel = DMA2_Stream4,
		       .init = {
				.DMA_Channel = DMA_Channel_0,
				.DMA_PeripheralBaseAddr =
				(uint32_t) & ADC1->DR},
		       }
		},
	.half_flag = DMA_IT_HTIF4,
	.full_flag = DMA_IT_TCIF4,
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
