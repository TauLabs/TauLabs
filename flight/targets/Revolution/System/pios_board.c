/**
 *****************************************************************************
 * @file       plop_board.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://www.taulabs.org, Copyright (C) 2012-2013
 * @addtogroup OpenPilotSystem OpenPilot System
 * @{
 * @addtogroup OpenPilotCore OpenPilot Core
 * @{
 * @brief Defines board specific static initializers for hardware for the revolution board.
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

/* Pull in the board-specific static HW definitions.
 * Including .c files is a bit ugly but this allows all of
 * the HW definitions to be const and static to limit their
 * scope.  
 *
 * NOTE: THIS IS THE ONLY PLACE THAT SHOULD EVER INCLUDE THIS FILE
 */
#include "board_hw_defs.c"

#include <plop.h>
#include <openpilot.h>
#include <uavobjectsinit.h>
#include "hwrevolution.h"
#include "modulesettings.h"
#include "manualcontrolsettings.h"

/**
 * Sensor configurations 
 */

#if defined(plop_INCLUDE_ADC)
#include "plop_adc_priv.h"
void plop_ADC_DMC_irq_handler(void);
void DMA2_Stream4_IRQHandler(void) __attribute__((alias("plop_ADC_DMC_irq_handler")));
struct plop_adc_cfg plop_adc_cfg = {
	.adc_dev = ADC1,
	.dma = {
		.irq = {
			.flags   = (DMA_FLAG_TCIF4 | DMA_FLAG_TEIF4 | DMA_FLAG_HTIF4),
			.init    = {
				.NVIC_IRQChannel                   = DMA2_Stream4_IRQn,
				.NVIC_IRQChannelPreemptionPriority = plop_IRQ_PRIO_LOW,
				.NVIC_IRQChannelSubPriority        = 0,
				.NVIC_IRQChannelCmd                = ENABLE,
			},
		},
		.rx = {
			.channel = DMA2_Stream4,
			.init    = {
				.DMA_Channel                    = DMA_Channel_0,
				.DMA_PeripheralBaseAddr = (uint32_t) & ADC1->DR
			},
		}
	},
	.half_flag = DMA_IT_HTIF4,
	.full_flag = DMA_IT_TCIF4,

};
void plop_ADC_DMC_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	plop_ADC_DMA_Handler();
}

#endif

#if defined(plop_INCLUDE_HMC5883)
#include "plop_hmc5883.h"
static const struct plop_exti_cfg plop_exti_hmc5883_cfg __exti_config = {
	.vector = plop_HMC5883_IRQHandler,
	.line = EXTI_Line5,
	.pin = {
		.gpio = GPIOB,
		.init = {
			.GPIO_Pin = GPIO_Pin_5,
			.GPIO_Speed = GPIO_Speed_100MHz,
			.GPIO_Mode = GPIO_Mode_IN,
			.GPIO_OType = GPIO_OType_OD,
			.GPIO_PuPd = GPIO_PuPd_NOPULL,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI9_5_IRQn,
			.NVIC_IRQChannelPreemptionPriority = plop_IRQ_PRIO_LOW,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line5, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};

static const struct plop_hmc5883_cfg plop_hmc5883_cfg = {
	.exti_cfg = &plop_exti_hmc5883_cfg,
	.M_ODR = plop_HMC5883_ODR_75,
	.Meas_Conf = plop_HMC5883_MEASCONF_NORMAL,
	.Gain = plop_HMC5883_GAIN_1_9,
	.Mode = plop_HMC5883_MODE_CONTINUOUS,
	.orientation = plop_HMC5883_TOP_270DEG,
};
#endif /* plop_INCLUDE_HMC5883 */

/**
 * Configuration for the MS5611 chip
 */
#if defined(plop_INCLUDE_MS5611)
#include "plop_ms5611_priv.h"
static const struct plop_ms5611_cfg plop_ms5611_cfg = {
	.oversampling = MS5611_OSR_512,
	.temperature_interleaving = 1,
};
#endif /* plop_INCLUDE_MS5611 */

/**
 * Configuration for the BMA180 chip
 */
#if defined(plop_INCLUDE_BMA180)
#include "plop_bma180.h"
static const struct plop_exti_cfg plop_exti_bma180_cfg __exti_config = {
	.vector = plop_BMA180_IRQHandler,
	.line = EXTI_Line4,
	.pin = {
		.gpio = GPIOC,
		.init = {
			.GPIO_Pin = GPIO_Pin_4,
			.GPIO_Speed = GPIO_Speed_100MHz,
			.GPIO_Mode = GPIO_Mode_IN,
			.GPIO_OType = GPIO_OType_OD,
			.GPIO_PuPd = GPIO_PuPd_NOPULL,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI4_IRQn,
			.NVIC_IRQChannelPreemptionPriority = plop_IRQ_PRIO_LOW,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line4, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};
static const struct plop_bma180_cfg plop_bma180_cfg = {
	.exti_cfg = &plop_exti_bma180_cfg,
	.bandwidth = BMA_BW_600HZ,
	.range = BMA_RANGE_8G,
};
#endif /* plop_INCLUDE_BMA180 */

/**
 * Configuration for the MPU6000 chip
 */
#if defined(plop_INCLUDE_MPU6000)
#include "plop_mpu6000.h"
static const struct plop_exti_cfg plop_exti_mpu6000_cfg __exti_config = {
	.vector = plop_MPU6000_IRQHandler,
	.line = EXTI_Line8,
	.pin = {
		.gpio = GPIOD,
		.init = {
			.GPIO_Pin = GPIO_Pin_8,
			.GPIO_Speed = GPIO_Speed_100MHz,
			.GPIO_Mode = GPIO_Mode_IN,
			.GPIO_OType = GPIO_OType_OD,
			.GPIO_PuPd = GPIO_PuPd_NOPULL,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI9_5_IRQn,
			.NVIC_IRQChannelPreemptionPriority = plop_IRQ_PRIO_HIGH,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line8, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};

static const struct plop_mpu60x0_cfg plop_mpu6000_cfg = {
	.exti_cfg = &plop_exti_mpu6000_cfg,
	.default_samplerate = 666,
	.interrupt_cfg = plop_MPU60X0_INT_CLR_ANYRD,
	.interrupt_en = plop_MPU60X0_INTEN_DATA_RDY,
	.User_ctl = plop_MPU60X0_USERCTL_DIS_I2C,
	.Pwr_mgmt_clk = plop_MPU60X0_PWRMGMT_PLL_Z_CLK,
	.default_filter = plop_MPU60X0_LOWPASS_256_HZ,
	.orientation = plop_MPU60X0_TOP_0DEG
};
#endif /* plop_INCLUDE_MPU6000 */

/**
 * Configuration for L3GD20 chip
 */
#if defined(plop_INCLUDE_L3GD20)
#include "plop_l3gd20.h"
static const struct plop_exti_cfg plop_exti_l3gd20_cfg __exti_config = {
	.vector = plop_L3GD20_IRQHandler,
	.line = EXTI_Line8,
	.pin = {
		.gpio = GPIOD,
		.init = {
			.GPIO_Pin = GPIO_Pin_8,
			.GPIO_Speed = GPIO_Speed_100MHz,
			.GPIO_Mode = GPIO_Mode_IN,
			.GPIO_OType = GPIO_OType_OD,
			.GPIO_PuPd = GPIO_PuPd_NOPULL,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI9_5_IRQn,
			.NVIC_IRQChannelPreemptionPriority = plop_IRQ_PRIO_HIGH,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line8, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};

static const struct plop_l3gd20_cfg plop_l3gd20_cfg = {
	.exti_cfg = &plop_exti_l3gd20_cfg,
	.range = plop_L3GD20_SCALE_500_DEG,
};
#endif /* plop_INCLUDE_L3GD20 */

/* One slot per selectable receiver group.
 *  eg. PWM, PPM, GCS, SPEKTRUM1, SPEKTRUM2, SBUS
 * NOTE: No slot in this map for NONE.
 */
uint32_t plop_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE];

#define plop_COM_TELEM_RF_RX_BUF_LEN 512
#define plop_COM_TELEM_RF_TX_BUF_LEN 512

#define plop_COM_GPS_RX_BUF_LEN 32

#define plop_COM_TELEM_USB_RX_BUF_LEN 65
#define plop_COM_TELEM_USB_TX_BUF_LEN 65

#define plop_COM_BRIDGE_RX_BUF_LEN 65
#define plop_COM_BRIDGE_TX_BUF_LEN 12

#if defined(plop_INCLUDE_DEBUG_CONSOLE)
#define plop_COM_DEBUGCONSOLE_TX_BUF_LEN 40
uintptr_t plop_com_debug_id;
#endif /* plop_INCLUDE_DEBUG_CONSOLE */

uintptr_t plop_com_gps_id;
uintptr_t plop_com_telem_usb_id;
uintptr_t plop_com_telem_rf_id;
uintptr_t plop_com_vcp_id;
uintptr_t plop_com_bridge_id;
uintptr_t plop_com_overo_id;

uintptr_t plop_uavo_settings_fs_id;
uintptr_t plop_waypoints_settings_fs_id;

/*
 * Setup a com port based on the passed cfg, driver and buffer sizes. rx or tx size of 0 disables rx or tx
 */
#if defined(plop_INCLUDE_USART) && defined(plop_INCLUDE_COM)
static void plop_Board_configure_com (const struct plop_usart_cfg *usart_port_cfg, size_t rx_buf_len, size_t tx_buf_len,
		const struct plop_com_driver *com_driver, uintptr_t *plop_com_id)
{
	uint32_t plop_usart_id;
	if (plop_USART_Init(&plop_usart_id, usart_port_cfg)) {
		plop_Assert(0);
	}

	uint8_t * rx_buffer;
	if (rx_buf_len > 0) {
		rx_buffer = (uint8_t *) pvPortMalloc(rx_buf_len);
		plop_Assert(rx_buffer);
	} else {
		rx_buffer = NULL;
	}

	uint8_t * tx_buffer;
	if (tx_buf_len > 0) {
		tx_buffer = (uint8_t *) pvPortMalloc(tx_buf_len);
		plop_Assert(tx_buffer);
	} else {
		tx_buffer = NULL;
	}

	if (plop_COM_Init(plop_com_id, com_driver, plop_usart_id,
				rx_buffer, rx_buf_len,
				tx_buffer, tx_buf_len)) {
		plop_Assert(0);
	}
}
#endif	/* plop_INCLUDE_USART && plop_INCLUDE_COM */

static void plop_Board_configure_dsm(const struct plop_usart_cfg *plop_usart_dsm_cfg, const struct plop_dsm_cfg *plop_dsm_cfg, 
		const struct plop_com_driver *plop_usart_com_driver,enum plop_dsm_proto *proto, 
		ManualControlSettingsChannelGroupsOptions channelgroup,uint8_t *bind)
{
	uint32_t plop_usart_dsm_id;
	if (plop_USART_Init(&plop_usart_dsm_id, plop_usart_dsm_cfg)) {
		plop_Assert(0);
	}
	
	uint32_t plop_dsm_id;
	if (plop_DSM_Init(&plop_dsm_id, plop_dsm_cfg, plop_usart_com_driver,
			plop_usart_dsm_id, *proto, *bind)) {
		plop_Assert(0);
	}
	
	uint32_t plop_dsm_rcvr_id;
	if (plop_RCVR_Init(&plop_dsm_rcvr_id, &plop_dsm_rcvr_driver, plop_dsm_id)) {
		plop_Assert(0);
	}
	plop_rcvr_group_map[channelgroup] = plop_dsm_rcvr_id;
}

/**
 * plop_Board_Init()
 * initializes all the core subsystems on this specific hardware
 * called from System/openpilot.c
 */

#include <plop_board_info.h>

void plop_Board_Init(void) {

	const struct plop_board_info * bdinfo = &plop_board_info_blob;	
	
	/* Delay system */
	plop_DELAY_Init();

	plop_LED_Init(&plop_led_cfg);

	/* Set up the SPI interface to the accelerometer*/
	if (plop_SPI_Init(&plop_spi_accel_id, &plop_spi_accel_cfg)) {
		plop_DEBUG_Assert(0);
	}
	
	/* Set up the SPI interface to the gyro */
	if (plop_SPI_Init(&plop_spi_gyro_id, &plop_spi_gyro_cfg)) {
		plop_DEBUG_Assert(0);
	}
#if !defined(plop_FLASH_ON_ACCEL)
	/* Set up the SPI interface to the flash */
	if (plop_SPI_Init(&plop_spi_flash_id, &plop_spi_flash_cfg)) {
		plop_DEBUG_Assert(0);
	}
	/* Connect flash to the appropriate interface and configure it */
	uintptr_t flash_id;
	plop_Flash_Jedec_Init(&flash_id, plop_spi_flash_id, 0, &flash_m25p_cfg);
#else
	/* Connect flash to the appropriate interface and configure it */
	uintptr_t flash_id;
	plop_Flash_Jedec_Init(&flash_id, plop_spi_accel_id, 1, &flash_m25p_cfg);
#endif
	plop_FLASHFS_Logfs_Init(&plop_uavo_settings_fs_id, &flashfs_m25p_settings_cfg, &plop_jedec_flash_driver, flash_id);
	plop_FLASHFS_Logfs_Init(&plop_waypoints_settings_fs_id, &flashfs_m25p_waypoints_cfg, &plop_jedec_flash_driver, flash_id);

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();
	
	HwRevolutionInitialize();
	ModuleSettingsInitialize();
	
#if defined(plop_INCLUDE_RTC)
	plop_RTC_Init(&plop_rtc_main_cfg);
#endif

	/* Initialize the alarms library */
	AlarmsInitialize();

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Set up pulse timers */
	plop_TIM_InitClock(&tim_1_cfg);
	plop_TIM_InitClock(&tim_2_cfg);
	plop_TIM_InitClock(&tim_3_cfg);
	plop_TIM_InitClock(&tim_4_cfg);
	plop_TIM_InitClock(&tim_5_cfg);
	plop_TIM_InitClock(&tim_9_cfg);
	plop_TIM_InitClock(&tim_10_cfg);
	plop_TIM_InitClock(&tim_11_cfg);

	/* IAP System Setup */
	plop_IAP_Init();
	uint16_t boot_count = plop_IAP_ReadBootCount();
	if (boot_count < 3) {
		plop_IAP_WriteBootCount(++boot_count);
		AlarmsClear(SYSTEMALARMS_ALARM_BOOTFAULT);
	} else {
		/* Too many failed boot attempts, force hw config to defaults */
		HwRevolutionSetDefaults(HwRevolutionHandle(), 0);
		ModuleSettingsSetDefaults(ModuleSettingsHandle(),0);
		AlarmsSet(SYSTEMALARMS_ALARM_BOOTFAULT, SYSTEMALARMS_ALARM_CRITICAL);
	}
	
	
	//plop_IAP_Init();

#if defined(plop_INCLUDE_USB)
	/* Initialize board specific USB data */
	plop_USB_BOARD_DATA_Init();

	/* Flags to determine if various USB interfaces are advertised */
	bool usb_hid_present = false;
	bool usb_cdc_present = false;

#if defined(plop_INCLUDE_USB_CDC)
	if (plop_USB_DESC_HID_CDC_Init()) {
		plop_Assert(0);
	}
	usb_hid_present = true;
	usb_cdc_present = true;
#else
	if (plop_USB_DESC_HID_ONLY_Init()) {
		plop_Assert(0);
	}
	usb_hid_present = true;
#endif

	uint32_t plop_usb_id;
	plop_USB_Init(&plop_usb_id, &plop_usb_main_cfg);

#if defined(plop_INCLUDE_USB_CDC)

	uint8_t hw_usb_vcpport;
	/* Configure the USB VCP port */
	HwRevolutionUSB_VCPPortGet(&hw_usb_vcpport);

	if (!usb_cdc_present) {
		/* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
		hw_usb_vcpport = HWREVOLUTION_USB_VCPPORT_DISABLED;
	}

	switch (hw_usb_vcpport) {
	case HWREVOLUTION_USB_VCPPORT_DISABLED:
		break;
	case HWREVOLUTION_USB_VCPPORT_USBTELEMETRY:
#if defined(plop_INCLUDE_COM)
		{
			uint32_t plop_usb_cdc_id;
			if (plop_USB_CDC_Init(&plop_usb_cdc_id, &plop_usb_cdc_cfg, plop_usb_id)) {
				plop_Assert(0);
			}
			uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(plop_COM_TELEM_USB_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(plop_COM_TELEM_USB_TX_BUF_LEN);
			plop_Assert(rx_buffer);
			plop_Assert(tx_buffer);
			if (plop_COM_Init(&plop_com_telem_usb_id, &plop_usb_cdc_com_driver, plop_usb_cdc_id,
						rx_buffer, plop_COM_TELEM_USB_RX_BUF_LEN,
						tx_buffer, plop_COM_TELEM_USB_TX_BUF_LEN)) {
				plop_Assert(0);
			}
		}
#endif	/* plop_INCLUDE_COM */
		break;
	case HWREVOLUTION_USB_VCPPORT_COMBRIDGE:
#if defined(plop_INCLUDE_COM)
		{
			uint32_t plop_usb_cdc_id;
			if (plop_USB_CDC_Init(&plop_usb_cdc_id, &plop_usb_cdc_cfg, plop_usb_id)) {
				plop_Assert(0);
			}
			uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(plop_COM_BRIDGE_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(plop_COM_BRIDGE_TX_BUF_LEN);
			plop_Assert(rx_buffer);
			plop_Assert(tx_buffer);
			if (plop_COM_Init(&plop_com_vcp_id, &plop_usb_cdc_com_driver, plop_usb_cdc_id,
						rx_buffer, plop_COM_BRIDGE_RX_BUF_LEN,
						tx_buffer, plop_COM_BRIDGE_TX_BUF_LEN)) {
				plop_Assert(0);
			}
		}
#endif	/* plop_INCLUDE_COM */
		break;
	case HWREVOLUTION_USB_VCPPORT_DEBUGCONSOLE:
#if defined(plop_INCLUDE_COM)
#if defined(plop_INCLUDE_DEBUG_CONSOLE)
		{
			uint32_t plop_usb_cdc_id;
			if (plop_USB_CDC_Init(&plop_usb_cdc_id, &plop_usb_cdc_cfg, plop_usb_id)) {
				plop_Assert(0);
			}
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(plop_COM_DEBUGCONSOLE_TX_BUF_LEN);
			plop_Assert(tx_buffer);
			if (plop_COM_Init(&plop_com_debug_id, &plop_usb_cdc_com_driver, plop_usb_cdc_id,
						NULL, 0,
						tx_buffer, plop_COM_DEBUGCONSOLE_TX_BUF_LEN)) {
				plop_Assert(0);
			}
		}
#endif	/* plop_INCLUDE_DEBUG_CONSOLE */
#endif	/* plop_INCLUDE_COM */
		break;
	}
#endif	/* plop_INCLUDE_USB_CDC */

#if defined(plop_INCLUDE_USB_HID)
	/* Configure the usb HID port */
	uint8_t hw_usb_hidport;
	HwRevolutionUSB_HIDPortGet(&hw_usb_hidport);

	if (!usb_hid_present) {
		/* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
		hw_usb_hidport = HWREVOLUTION_USB_HIDPORT_DISABLED;
	}

	switch (hw_usb_hidport) {
	case HWREVOLUTION_USB_HIDPORT_DISABLED:
		break;
	case HWREVOLUTION_USB_HIDPORT_USBTELEMETRY:
#if defined(plop_INCLUDE_COM)
		{
			uint32_t plop_usb_hid_id;
			if (plop_USB_HID_Init(&plop_usb_hid_id, &plop_usb_hid_cfg, plop_usb_id)) {
				plop_Assert(0);
			}
			uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(plop_COM_TELEM_USB_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(plop_COM_TELEM_USB_TX_BUF_LEN);
			plop_Assert(rx_buffer);
			plop_Assert(tx_buffer);
			if (plop_COM_Init(&plop_com_telem_usb_id, &plop_usb_hid_com_driver, plop_usb_hid_id,
						rx_buffer, plop_COM_TELEM_USB_RX_BUF_LEN,
						tx_buffer, plop_COM_TELEM_USB_TX_BUF_LEN)) {
				plop_Assert(0);
			}
		}
#endif	/* plop_INCLUDE_COM */
		break;
	}

#endif	/* plop_INCLUDE_USB_HID */

	if (usb_hid_present || usb_cdc_present) {
		plop_USBHOOK_Activate();
	}
#endif	/* plop_INCLUDE_USB */

	/* Configure IO ports */
	uint8_t hw_DSMxBind;
	HwRevolutionDSMxBindGet(&hw_DSMxBind);
	
	/* Configure Telemetry port */
	uint8_t hw_telemetryport;
	HwRevolutionTelemetryPortGet(&hw_telemetryport);

	switch (hw_telemetryport){
		case HWREVOLUTION_TELEMETRYPORT_DISABLED:
			break;
		case HWREVOLUTION_TELEMETRYPORT_TELEMETRY:
			plop_Board_configure_com(&plop_usart_telem_cfg, plop_COM_TELEM_RF_RX_BUF_LEN, plop_COM_TELEM_RF_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_telem_rf_id);
			break;
		case HWREVOLUTION_TELEMETRYPORT_DEBUGCONSOLE:
#if defined(plop_INCLUDE_DEBUG_CONSOLE)
			plop_Board_configure_com(&plop_usart_telem_cfg, 0, plop_DEBUGCONSOLE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_debug_id);
#endif	/* plop_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWREVOLUTION_TELEMETRYPORT_COMBRIDGE:
			plop_Board_configure_com(&plop_usart_telem_cfg, plop_COM_BRIDGE_RX_BUF_LEN, plop_COM_BRIDGE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_bridge_id);
			break;
			
	} /* 	hw_telemetryport */

	/* Configure GPS port */
	uint8_t hw_gpsport;
	HwRevolutionGPSPortGet(&hw_gpsport);
	switch (hw_gpsport){
		case HWREVOLUTION_GPSPORT_DISABLED:
			break;
			
		case HWREVOLUTION_GPSPORT_TELEMETRY:
			plop_Board_configure_com(&plop_usart_gps_cfg, plop_COM_TELEM_RF_RX_BUF_LEN, plop_COM_TELEM_RF_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_telem_rf_id);
			break;
			
		case HWREVOLUTION_GPSPORT_GPS:
			plop_Board_configure_com(&plop_usart_gps_cfg, plop_COM_GPS_RX_BUF_LEN, 0,  &plop_usart_com_driver, &plop_com_gps_id);
			break;
		
		case HWREVOLUTION_GPSPORT_DEBUGCONSOLE:
#if defined(plop_INCLUDE_DEBUG_CONSOLE)
			plop_Board_configure_com(&plop_usart_gps_cfg, 0, plop_DEBUGCONSOLE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_debug_id);
#endif	/* plop_INCLUDE_DEBUG_CONSOLE */
			break;
			
		case HWREVOLUTION_GPSPORT_COMBRIDGE:
			plop_Board_configure_com(&plop_usart_gps_cfg, plop_COM_BRIDGE_RX_BUF_LEN, plop_COM_BRIDGE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_bridge_id);
			break;
	}/* hw_gpsport */

	/* Configure AUXPort */
	uint8_t hw_auxport;
	HwRevolutionAuxPortGet(&hw_auxport);

	switch (hw_auxport) {
		case HWREVOLUTION_AUXPORT_DISABLED:
			break;
			
		case HWREVOLUTION_AUXPORT_TELEMETRY:
			plop_Board_configure_com(&plop_usart_aux_cfg, plop_COM_TELEM_RF_RX_BUF_LEN, plop_COM_TELEM_RF_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_telem_rf_id);
			break;
			
		case HWREVOLUTION_AUXPORT_DSM2:
		case HWREVOLUTION_AUXPORT_DSMX10BIT:
		case HWREVOLUTION_AUXPORT_DSMX11BIT:
		{
			enum plop_dsm_proto proto;
			switch (hw_auxport) {
				case HWREVOLUTION_AUXPORT_DSM2:
					proto = plop_DSM_PROTO_DSM2;
					break;
				case HWREVOLUTION_AUXPORT_DSMX10BIT:
					proto = plop_DSM_PROTO_DSMX10BIT;
					break;
				case HWREVOLUTION_AUXPORT_DSMX11BIT:
					proto = plop_DSM_PROTO_DSMX11BIT;
					break;
				default:
					plop_Assert(0);
					break;
			}
			//TODO: Define the various Channelgroup for Revo dsm inputs and handle here
			plop_Board_configure_dsm(&plop_usart_dsm_aux_cfg, &plop_dsm_aux_cfg, 
											 &plop_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT,&hw_DSMxBind);
		}
			break;
		case HWREVOLUTION_AUXPORT_DEBUGCONSOLE:
#if defined(plop_INCLUDE_DEBUG_CONSOLE)
			plop_Board_configure_com(&plop_usart_aux_cfg, 0, plop_DEBUGCONSOLE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_debug_id);
#endif	/* plop_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWREVOLUTION_AUXPORT_COMBRIDGE:
			plop_Board_configure_com(&plop_usart_aux_cfg, plop_COM_BRIDGE_RX_BUF_LEN, plop_COM_BRIDGE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_bridge_id);
			break;
	} /* hw_auxport */
	/* Configure AUXSbusPort */
	//TODO: ensure that the serial invertion pin is setted correctly
	uint8_t hw_auxsbusport;
	HwRevolutionAuxSBusPortGet(&hw_auxsbusport);
	
	switch (hw_auxsbusport) {
		case HWREVOLUTION_AUXSBUSPORT_DISABLED:
			break;
		case HWREVOLUTION_AUXSBUSPORT_SBUS:
#ifdef plop_INCLUDE_SBUS
		{
			uint32_t plop_usart_sbus_id;
			if (plop_USART_Init(&plop_usart_sbus_id, &plop_usart_sbus_auxsbus_cfg)) {
				plop_Assert(0);
			}
			
			uint32_t plop_sbus_id;
			if (plop_SBus_Init(&plop_sbus_id, &plop_sbus_cfg, &plop_usart_com_driver, plop_usart_sbus_id)) {
				plop_Assert(0);
			}
			
			uint32_t plop_sbus_rcvr_id;
			if (plop_RCVR_Init(&plop_sbus_rcvr_id, &plop_sbus_rcvr_driver, plop_sbus_id)) {
				plop_Assert(0);
			}
			plop_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_SBUS] = plop_sbus_rcvr_id;
			
		}
#endif /* plop_INCLUDE_SBUS */
			break;

		case HWREVOLUTION_AUXSBUSPORT_DSM2:
		case HWREVOLUTION_AUXSBUSPORT_DSMX10BIT:
		case HWREVOLUTION_AUXSBUSPORT_DSMX11BIT:
		{
			enum plop_dsm_proto proto;
			switch (hw_auxsbusport) {
				case HWREVOLUTION_AUXSBUSPORT_DSM2:
					proto = plop_DSM_PROTO_DSM2;
					break;
				case HWREVOLUTION_AUXSBUSPORT_DSMX10BIT:
					proto = plop_DSM_PROTO_DSMX10BIT;
					break;
				case HWREVOLUTION_AUXSBUSPORT_DSMX11BIT:
					proto = plop_DSM_PROTO_DSMX11BIT;
					break;
				default:
					plop_Assert(0);
					break;
			}
			//TODO: Define the various Channelgroup for Revo dsm inputs and handle here
			plop_Board_configure_dsm(&plop_usart_dsm_auxsbus_cfg, &plop_dsm_auxsbus_cfg, 
											 &plop_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT,&hw_DSMxBind);
		}
			break;
		case HWREVOLUTION_AUXSBUSPORT_DEBUGCONSOLE:
#if defined(plop_INCLUDE_DEBUG_CONSOLE)
			plop_Board_configure_com(&plop_usart_auxsbus_cfg, 0, plop_DEBUGCONSOLE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_debug_id);
#endif	/* plop_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWREVOLUTION_AUXSBUSPORT_COMBRIDGE:
			plop_Board_configure_com(&plop_usart_auxsbus_cfg, plop_COM_BRIDGE_RX_BUF_LEN, plop_COM_BRIDGE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_bridge_id);
			break;
	} /* hw_auxport */
	
	/* Configure FlexiPort */

	uint8_t hw_flexiport;
	HwRevolutionFlexiPortGet(&hw_flexiport);
	
	switch (hw_flexiport) {
		case HWREVOLUTION_FLEXIPORT_DISABLED:
			break;
		case HWREVOLUTION_FLEXIPORT_I2C:
#if defined(plop_INCLUDE_I2C)
		{
			if (plop_I2C_Init(&plop_i2c_flexiport_adapter_id, &plop_i2c_flexiport_adapter_cfg)) {
				plop_Assert(0);
			}
		}
#endif	/* plop_INCLUDE_I2C */
			break;
			
		case HWREVOLUTION_FLEXIPORT_DSM2:
		case HWREVOLUTION_FLEXIPORT_DSMX10BIT:
		case HWREVOLUTION_FLEXIPORT_DSMX11BIT:
		{
			enum plop_dsm_proto proto;
			switch (hw_flexiport) {
				case HWREVOLUTION_FLEXIPORT_DSM2:
					proto = plop_DSM_PROTO_DSM2;
					break;
				case HWREVOLUTION_FLEXIPORT_DSMX10BIT:
					proto = plop_DSM_PROTO_DSMX10BIT;
					break;
				case HWREVOLUTION_FLEXIPORT_DSMX11BIT:
					proto = plop_DSM_PROTO_DSMX11BIT;
					break;
				default:
					plop_Assert(0);
					break;
			}
			//TODO: Define the various Channelgroup for Revo dsm inputs and handle here
			plop_Board_configure_dsm(&plop_usart_dsm_flexi_cfg, &plop_dsm_flexi_cfg, 
											 &plop_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT,&hw_DSMxBind);
		}
			break;
		case HWREVOLUTION_FLEXIPORT_DEBUGCONSOLE:
#if defined(plop_INCLUDE_DEBUG_CONSOLE)
			plop_Board_configure_com(&plop_usart_flexi_cfg, 0, plop_DEBUGCONSOLE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_aux_id);
#endif	/* plop_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWREVOLUTION_FLEXIPORT_COMBRIDGE:
			plop_Board_configure_com(&plop_usart_flexi_cfg, plop_COM_BRIDGE_RX_BUF_LEN, plop_COM_BRIDGE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_bridge_id);
			break;
	} /* hw_flexiport */
	
	
	/* Configure the receiver port*/
	uint8_t hw_rcvrport;
	HwRevolutionRcvrPortGet(&hw_rcvrport);
	//   
	switch (hw_rcvrport){
		case HWREVOLUTION_RCVRPORT_DISABLED:
			break;
		case HWREVOLUTION_RCVRPORT_PWM:
#if defined(plop_INCLUDE_PWM)
		{
			/* Set up the receiver port.  Later this should be optional */
			uint32_t plop_pwm_id;
			plop_PWM_Init(&plop_pwm_id, &plop_pwm_cfg);
			
			uint32_t plop_pwm_rcvr_id;
			if (plop_RCVR_Init(&plop_pwm_rcvr_id, &plop_pwm_rcvr_driver, plop_pwm_id)) {
				plop_Assert(0);
			}
			plop_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PWM] = plop_pwm_rcvr_id;
		}
#endif	/* plop_INCLUDE_PWM */
			break;
		case HWREVOLUTION_RCVRPORT_PPM:
		case HWREVOLUTION_RCVRPORT_PPMOUTPUTS:
#if defined(plop_INCLUDE_PPM)
		{
			uint32_t plop_ppm_id;
			plop_PPM_Init(&plop_ppm_id, &plop_ppm_cfg);
			
			uint32_t plop_ppm_rcvr_id;
			if (plop_RCVR_Init(&plop_ppm_rcvr_id, &plop_ppm_rcvr_driver, plop_ppm_id)) {
				plop_Assert(0);
			}
			plop_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PPM] = plop_ppm_rcvr_id;
		}
#endif	/* plop_INCLUDE_PPM */
		case HWREVOLUTION_RCVRPORT_OUTPUTS:
		
			break;
	}

#if defined(plop_OVERO_SPI)
	/* Set up the SPI based plop_COM interface to the overo */
	{
		bool overo_enabled = false;
#ifdef MODULE_OveroSync_BUILTIN
		overo_enabled = true;
#else
		uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
		ModuleSettingsAdminStateGet(module_state);
		if (module_state[MODULESETTINGS_ADMINSTATE_OVEROSYNC] == MODULESETTINGS_ADMINSTATE_ENABLED) {
			overo_enabled = true;
		} else {
			overo_enabled = false;
		}
#endif
		if (overo_enabled) {
			if (plop_OVERO_Init(&plop_overo_id, &plop_overo_cfg)) {
				plop_DEBUG_Assert(0);
			}
			const uint32_t PACKET_SIZE = 1024;
			uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(PACKET_SIZE);
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(PACKET_SIZE);
			plop_Assert(rx_buffer);
			plop_Assert(tx_buffer);
			if (plop_COM_Init(&plop_com_overo_id, &plop_overo_com_driver, plop_overo_id,
							  rx_buffer, PACKET_SIZE,
							  tx_buffer, PACKET_SIZE)) {
				plop_Assert(0);
			}
		}
	}

#endif

#if defined(plop_INCLUDE_GCSRCVR)
	GCSReceiverInitialize();
	uint32_t plop_gcsrcvr_id;
	plop_GCSRCVR_Init(&plop_gcsrcvr_id);
	uint32_t plop_gcsrcvr_rcvr_id;
	if (plop_RCVR_Init(&plop_gcsrcvr_rcvr_id, &plop_gcsrcvr_rcvr_driver, plop_gcsrcvr_id)) {
		plop_Assert(0);
	}
	plop_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_GCS] = plop_gcsrcvr_rcvr_id;
#endif	/* plop_INCLUDE_GCSRCVR */
	
#ifndef plop_DEBUG_ENABLE_DEBUG_PINS
	switch (hw_rcvrport) {
		case HWREVOLUTION_RCVRPORT_DISABLED:
		case HWREVOLUTION_RCVRPORT_PWM:
		case HWREVOLUTION_RCVRPORT_PPM:
			/* Set up the servo outputs */
			plop_Servo_Init(&plop_servo_cfg);
			break;
		case HWREVOLUTION_RCVRPORT_PPMOUTPUTS:
		case HWREVOLUTION_RCVRPORT_OUTPUTS:
			//plop_Servo_Init(&plop_servo_rcvr_cfg);
			//TODO: Prepare the configurations on board_hw_defs and handle here:
			plop_Servo_Init(&plop_servo_cfg);
			break;
	}
#else
	plop_DEBUG_Init(&plop_tim_servo_all_channels, NELEMENTS(plop_tim_servo_all_channels));
#endif
	
	if (plop_I2C_Init(&plop_i2c_mag_adapter_id, &plop_i2c_mag_adapter_cfg)) {
		plop_DEBUG_Assert(0);
	}

	if (plop_I2C_Init(&plop_i2c_pressure_adapter_id, &plop_i2c_pressure_adapter_cfg)) {
		plop_DEBUG_Assert(0);
	}
	
	plop_DELAY_WaitmS(50);

	plop_SENSORS_Init();

#if defined(plop_INCLUDE_ADC)
	plop_ADC_Init(&plop_adc_cfg);
#endif

#if defined(plop_INCLUDE_HMC5883)
	plop_HMC5883_Init(plop_I2C_MAIN_ADAPTER, &plop_hmc5883_cfg);
#endif
	
#if defined(plop_INCLUDE_MS5611)
	plop_MS5611_Init(&plop_ms5611_cfg, plop_i2c_pressure_adapter_id);
#endif

	switch(bdinfo->board_rev) {
		case 0x01:
#if defined(plop_INCLUDE_L3GD20)
			plop_L3GD20_Init(plop_spi_gyro_id, 0, &plop_l3gd20_cfg);
			plop_Assert(plop_L3GD20_Test() == 0);
#endif
#if defined(plop_INCLUDE_BMA180)
			plop_BMA180_Init(plop_spi_accel_id, 0, &plop_bma180_cfg);
			plop_Assert(plop_BMA180_Test() == 0);
#endif
			break;
		case 0x02:
#if defined(plop_INCLUDE_MPU6000)
			plop_MPU6000_Init(plop_spi_gyro_id,0, &plop_mpu6000_cfg);

			// To be safe map from UAVO enum to driver enum
			uint8_t hw_gyro_range;
			HwRevolutionGyroRangeGet(&hw_gyro_range);
			switch(hw_gyro_range) {
				case HWREVOLUTION_GYRORANGE_250:
					plop_MPU6000_SetGyroRange(plop_MPU60X0_SCALE_250_DEG);
					break;
				case HWREVOLUTION_GYRORANGE_500:
					plop_MPU6000_SetGyroRange(plop_MPU60X0_SCALE_500_DEG);
					break;
				case HWREVOLUTION_GYRORANGE_1000:
					plop_MPU6000_SetGyroRange(plop_MPU60X0_SCALE_1000_DEG);
					break;
				case HWREVOLUTION_GYRORANGE_2000:
					plop_MPU6000_SetGyroRange(plop_MPU60X0_SCALE_2000_DEG);
					break;
			}

			uint8_t hw_accel_range;
			HwRevolutionAccelRangeGet(&hw_accel_range);
			switch(hw_accel_range) {
				case HWREVOLUTION_ACCELRANGE_2G:
					plop_MPU6000_SetAccelRange(plop_MPU60X0_ACCEL_2G);
					break;
				case HWREVOLUTION_ACCELRANGE_4G:
					plop_MPU6000_SetAccelRange(plop_MPU60X0_ACCEL_4G);
					break;
				case HWREVOLUTION_ACCELRANGE_8G:
					plop_MPU6000_SetAccelRange(plop_MPU60X0_ACCEL_8G);
					break;
				case HWREVOLUTION_ACCELRANGE_16G:
					plop_MPU6000_SetAccelRange(plop_MPU60X0_ACCEL_16G);
					break;
			}

			// the filter has to be set before rate else divisor calculation will fail
			uint8_t hw_mpu6000_dlpf;
			HwRevolutionMPU6000DLPFGet(&hw_mpu6000_dlpf);
			enum plop_mpu60x0_filter mpu6000_dlpf = \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_256) ? plop_MPU60X0_LOWPASS_256_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_188) ? plop_MPU60X0_LOWPASS_188_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_98) ? plop_MPU60X0_LOWPASS_98_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_42) ? plop_MPU60X0_LOWPASS_42_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_20) ? plop_MPU60X0_LOWPASS_20_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_10) ? plop_MPU60X0_LOWPASS_10_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_5) ? plop_MPU60X0_LOWPASS_5_HZ : \
			    plop_mpu6000_cfg.default_filter;
			plop_MPU6000_SetLPF(mpu6000_dlpf);

			uint8_t hw_mpu6000_samplerate;
			HwRevolutionMPU6000RateGet(&hw_mpu6000_samplerate);
			uint16_t mpu6000_samplerate = \
			    (hw_mpu6000_samplerate == HWREVOLUTION_MPU6000RATE_200) ? 200 : \
			    (hw_mpu6000_samplerate == HWREVOLUTION_MPU6000RATE_333) ? 333 : \
			    (hw_mpu6000_samplerate == HWREVOLUTION_MPU6000RATE_500) ? 500 : \
			    (hw_mpu6000_samplerate == HWREVOLUTION_MPU6000RATE_666) ? 666 : \
			    (hw_mpu6000_samplerate == HWREVOLUTION_MPU6000RATE_1000) ? 1000 : \
			    (hw_mpu6000_samplerate == HWREVOLUTION_MPU6000RATE_2000) ? 2000 : \
			    (hw_mpu6000_samplerate == HWREVOLUTION_MPU6000RATE_4000) ? 4000 : \
			    (hw_mpu6000_samplerate == HWREVOLUTION_MPU6000RATE_8000) ? 8000 : \
			    plop_mpu6000_cfg.default_samplerate;
			plop_MPU6000_SetSampleRate(mpu6000_samplerate);
#endif
			break;
		default:
			plop_DEBUG_Assert(0);
	}

}

/**
 * @}
 * @}
 */

