/**
 ******************************************************************************
 * @file       plop_board.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://www.taulabs.org, Copyright (C) 2012-2013
 * @addtogroup OpenPilotSystem OpenPilot System
 * @{
 * @addtogroup OpenPilotCore OpenPilot Core
 * @{
 * @brief Defines board specific static initializers for hardware for the revomini board.
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
#include "hwrevomini.h"
#include "manualcontrolsettings.h"
#include "modulesettings.h"

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

struct stm32_gpio plop_current_sonar_pin ={
    .gpio = GPIOA,
			.init = {
				.GPIO_Pin = GPIO_Pin_8,
				.GPIO_Speed = GPIO_Speed_2MHz,
				.GPIO_Mode  = GPIO_Mode_IN,
				.GPIO_OType = GPIO_OType_OD,
				.GPIO_PuPd  = GPIO_PuPd_NOPULL
			},
			.pin_source = GPIO_PinSource8,
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
	.line = EXTI_Line7,
	.pin = {
		.gpio = GPIOB,
		.init = {
			.GPIO_Pin = GPIO_Pin_7,
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
			.EXTI_Line = EXTI_Line7, // matches above GPIO pin
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
 * Configuration for the MPU6000 chip
 */
#if defined(plop_INCLUDE_MPU6000)
#include "plop_mpu6000.h"
static const struct plop_exti_cfg plop_exti_mpu6000_cfg __exti_config = {
	.vector = plop_MPU6000_IRQHandler,
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
			.NVIC_IRQChannelPreemptionPriority = plop_IRQ_PRIO_HIGH,
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

static const struct plop_mpu60x0_cfg plop_mpu6000_cfg = {
	.exti_cfg = &plop_exti_mpu6000_cfg,
	.default_samplerate = 666,
	.interrupt_cfg = plop_MPU60X0_INT_CLR_ANYRD,
	.interrupt_en = plop_MPU60X0_INTEN_DATA_RDY,
	.User_ctl = plop_MPU60X0_USERCTL_DIS_I2C,
	.Pwr_mgmt_clk = plop_MPU60X0_PWRMGMT_PLL_Z_CLK,
	.default_filter = plop_MPU60X0_LOWPASS_256_HZ,
	.orientation = plop_MPU60X0_TOP_180DEG
};
#endif /* plop_INCLUDE_MPU6000 */

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

#define plop_COM_RFM22B_RF_RX_BUF_LEN 512
#define plop_COM_RFM22B_RF_TX_BUF_LEN 512

#if defined(plop_INCLUDE_DEBUG_CONSOLE)
#define plop_COM_DEBUGCONSOLE_TX_BUF_LEN 40
uintptr_t plop_com_debug_id;
#endif	/* plop_INCLUDE_DEBUG_CONSOLE */

uintptr_t plop_com_gps_id;
uintptr_t plop_com_telem_usb_id;
uintptr_t plop_com_telem_rf_id;
uintptr_t plop_com_vcp_id;
uintptr_t plop_com_bridge_id;
uintptr_t plop_com_overo_id;
uint32_t plop_rfm22b_id;

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

	/* Delay system */
	plop_DELAY_Init();

	const struct plop_board_info * bdinfo = &plop_board_info_blob;
	
#if defined(plop_INCLUDE_LED)
	const struct plop_led_cfg * led_cfg = plop_BOARD_HW_DEFS_GetLedCfg(bdinfo->board_rev);
	plop_Assert(led_cfg);
	plop_LED_Init(led_cfg);
#endif	/* plop_INCLUDE_LED */

	/* Set up the SPI interface to the gyro/acelerometer */
	if (plop_SPI_Init(&plop_spi_gyro_id, &plop_spi_gyro_cfg)) {
		plop_DEBUG_Assert(0);
	}
	
	/* Set up the SPI interface to the flash and rfm22b */
	if (plop_SPI_Init(&plop_spi_telem_flash_id, &plop_spi_telem_flash_cfg)) {
		plop_DEBUG_Assert(0);
	}

#if defined(plop_INCLUDE_FLASH)
	/* Connect flash to the appropriate interface and configure it */
	uintptr_t flash_id;
	plop_Flash_Jedec_Init(&flash_id, plop_spi_telem_flash_id, 1, &flash_m25p_cfg);
	plop_FLASHFS_Logfs_Init(&plop_uavo_settings_fs_id, &flashfs_m25p_settings_cfg, &plop_jedec_flash_driver, flash_id);
	plop_FLASHFS_Logfs_Init(&plop_waypoints_settings_fs_id, &flashfs_m25p_waypoints_cfg, &plop_jedec_flash_driver, flash_id);
#endif

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();
	
	HwRevoMiniInitialize();
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
	plop_TIM_InitClock(&tim_3_cfg);
	plop_TIM_InitClock(&tim_4_cfg);
	plop_TIM_InitClock(&tim_5_cfg);
	plop_TIM_InitClock(&tim_8_cfg);
	plop_TIM_InitClock(&tim_9_cfg);
	plop_TIM_InitClock(&tim_10_cfg);
	plop_TIM_InitClock(&tim_11_cfg);
	plop_TIM_InitClock(&tim_12_cfg);
	/* IAP System Setup */
	plop_IAP_Init();
	uint16_t boot_count = plop_IAP_ReadBootCount();
	if (boot_count < 3) {
		plop_IAP_WriteBootCount(++boot_count);
		AlarmsClear(SYSTEMALARMS_ALARM_BOOTFAULT);
	} else {
		/* Too many failed boot attempts, force hw config to defaults */
		HwRevoMiniSetDefaults(HwRevoMiniHandle(), 0);
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
	plop_USB_Init(&plop_usb_id, plop_BOARD_HW_DEFS_GetUsbCfg(bdinfo->board_rev));

#if defined(plop_INCLUDE_USB_CDC)

	uint8_t hw_usb_vcpport;
	/* Configure the USB VCP port */
	HwRevoMiniUSB_VCPPortGet(&hw_usb_vcpport);

	if (!usb_cdc_present) {
		/* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
		hw_usb_vcpport = HWREVOMINI_USB_VCPPORT_DISABLED;
	}

	switch (hw_usb_vcpport) {
	case HWREVOMINI_USB_VCPPORT_DISABLED:
		break;
	case HWREVOMINI_USB_VCPPORT_USBTELEMETRY:
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
	case HWREVOMINI_USB_VCPPORT_COMBRIDGE:
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
	case HWREVOMINI_USB_VCPPORT_DEBUGCONSOLE:
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
	HwRevoMiniUSB_HIDPortGet(&hw_usb_hidport);

	if (!usb_hid_present) {
		/* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
		hw_usb_hidport = HWREVOMINI_USB_HIDPORT_DISABLED;
	}

	switch (hw_usb_hidport) {
	case HWREVOMINI_USB_HIDPORT_DISABLED:
		break;
	case HWREVOMINI_USB_HIDPORT_USBTELEMETRY:
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
	HwRevoMiniDSMxBindGet(&hw_DSMxBind);
	
	/* Configure main USART port */
	uint8_t hw_mainport;
	HwRevoMiniMainPortGet(&hw_mainport);
	switch (hw_mainport) {
		case HWREVOMINI_MAINPORT_DISABLED:
			break;
		case HWREVOMINI_MAINPORT_TELEMETRY:
			plop_Board_configure_com(&plop_usart_main_cfg, plop_COM_TELEM_RF_RX_BUF_LEN, plop_COM_TELEM_RF_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_telem_rf_id);
			break;
		case HWREVOMINI_MAINPORT_GPS:
			plop_Board_configure_com(&plop_usart_main_cfg, plop_COM_GPS_RX_BUF_LEN, 0, &plop_usart_com_driver, &plop_com_gps_id);
			break;
		case HWREVOMINI_MAINPORT_SBUS:
#if defined(plop_INCLUDE_SBUS)
                        {
                                uint32_t plop_usart_sbus_id;
                                if (plop_USART_Init(&plop_usart_sbus_id, &plop_usart_sbus_main_cfg)) {
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
#endif
                        break;
		case HWREVOMINI_MAINPORT_DSM2:
		case HWREVOMINI_MAINPORT_DSMX10BIT:
		case HWREVOMINI_MAINPORT_DSMX11BIT:
			{
				enum plop_dsm_proto proto;
				switch (hw_mainport) {
				case HWREVOMINI_MAINPORT_DSM2:
					proto = plop_DSM_PROTO_DSM2;
					break;
				case HWREVOMINI_MAINPORT_DSMX10BIT:
					proto = plop_DSM_PROTO_DSMX10BIT;
					break;
				case HWREVOMINI_MAINPORT_DSMX11BIT:
					proto = plop_DSM_PROTO_DSMX11BIT;
					break;
				default:
					plop_Assert(0);
					break;
				}

				// Force binding to zero on the main port
				hw_DSMxBind = 0;

				//TODO: Define the various Channelgroup for Revo dsm inputs and handle here
				plop_Board_configure_dsm(&plop_usart_dsm_main_cfg, &plop_dsm_main_cfg, 
							&plop_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT,&hw_DSMxBind);
			}
			break;
		case HWREVOMINI_MAINPORT_DEBUGCONSOLE:
#if defined(plop_INCLUDE_DEBUG_CONSOLE)
			{
				plop_Board_configure_com(&plop_usart_main_cfg, 0, plop_COM_DEBUGCONSOLE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_debug_id);
			}
#endif	/* plop_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWREVOMINI_MAINPORT_COMBRIDGE:
			plop_Board_configure_com(&plop_usart_main_cfg, plop_COM_BRIDGE_RX_BUF_LEN, plop_COM_BRIDGE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_bridge_id);
			break;
			
	} /* 	hw_mainport */

	if (hw_mainport != HWREVOMINI_MAINPORT_SBUS) {
		GPIO_Init(plop_sbus_cfg.inv.gpio, (GPIO_InitTypeDef*)&plop_sbus_cfg.inv.init);
		GPIO_WriteBit(plop_sbus_cfg.inv.gpio, plop_sbus_cfg.inv.init.GPIO_Pin, plop_sbus_cfg.gpio_inv_disable);
	}

	/* Configure FlexiPort */
	uint8_t hw_flexiport;
	HwRevoMiniFlexiPortGet(&hw_flexiport);
	switch (hw_flexiport) {
		case HWREVOMINI_FLEXIPORT_DISABLED:
			break;
                case HWREVOMINI_FLEXIPORT_TELEMETRY:
                        plop_Board_configure_com(&plop_usart_flexi_cfg, plop_COM_TELEM_RF_RX_BUF_LEN, plop_COM_TELEM_RF_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_telem_rf_id);
			break;
		case HWREVOMINI_FLEXIPORT_I2C:
#if defined(plop_INCLUDE_I2C)
			{
				if (plop_I2C_Init(&plop_i2c_flexiport_adapter_id, &plop_i2c_flexiport_adapter_cfg)) {
					plop_Assert(0);
				}
			}
#endif	/* plop_INCLUDE_I2C */
			break;
		case HWREVOMINI_FLEXIPORT_GPS:
			plop_Board_configure_com(&plop_usart_flexi_cfg, plop_COM_GPS_RX_BUF_LEN, 0, &plop_usart_com_driver, &plop_com_gps_id);
			break;
		case HWREVOMINI_FLEXIPORT_DSM2:
		case HWREVOMINI_FLEXIPORT_DSMX10BIT:
		case HWREVOMINI_FLEXIPORT_DSMX11BIT:
			{
				enum plop_dsm_proto proto;
				switch (hw_flexiport) {
				case HWREVOMINI_FLEXIPORT_DSM2:
					proto = plop_DSM_PROTO_DSM2;
					break;
				case HWREVOMINI_FLEXIPORT_DSMX10BIT:
					proto = plop_DSM_PROTO_DSMX10BIT;
					break;
				case HWREVOMINI_FLEXIPORT_DSMX11BIT:
					proto = plop_DSM_PROTO_DSMX11BIT;
					break;
				default:
					plop_Assert(0);
					break;
				}
				//TODO: Define the various Channelgroup for Revo dsm inputs and handle here
				plop_Board_configure_dsm(&plop_usart_dsm_flexi_cfg, &plop_dsm_flexi_cfg, 
							&plop_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMFLEXIPORT,&hw_DSMxBind);
			}
			break;
		case HWREVOMINI_FLEXIPORT_DEBUGCONSOLE:
#if defined(plop_INCLUDE_DEBUG_CONSOLE)
			{
				plop_Board_configure_com(&plop_usart_main_cfg, 0, plop_COM_DEBUGCONSOLE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_debug_id);
			}
#endif	/* plop_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWREVOMINI_FLEXIPORT_COMBRIDGE:
			plop_Board_configure_com(&plop_usart_flexi_cfg, plop_COM_BRIDGE_RX_BUF_LEN, plop_COM_BRIDGE_TX_BUF_LEN, &plop_usart_com_driver, &plop_com_bridge_id);
			break;
	} /* hwsettings_rv_flexiport */

	/* Initalize the RFM22B radio COM device. */
#if defined(plop_INCLUDE_RFM22B)
	uint8_t hwsettings_radioport;
	HwRevoMiniRadioPortGet(&hwsettings_radioport);
	switch (hwsettings_radioport) {
		case HWREVOMINI_RADIOPORT_DISABLED:
			break;
		case HWREVOMINI_RADIOPORT_TELEMETRY:
		{
			extern const struct plop_rfm22b_cfg * plop_BOARD_HW_DEFS_GetRfm22Cfg (uint32_t board_revision);
			const struct plop_board_info * bdinfo = &plop_board_info_blob;
			const struct plop_rfm22b_cfg *plop_rfm22b_cfg = plop_BOARD_HW_DEFS_GetRfm22Cfg(bdinfo->board_rev);
			if (plop_RFM22B_Init(&plop_rfm22b_id, plop_RFM22_SPI_PORT, plop_rfm22b_cfg->slave_num, plop_rfm22b_cfg)) {
				plop_Assert(0);
			}
			uint8_t *rx_buffer = (uint8_t *) pvPortMalloc(plop_COM_RFM22B_RF_RX_BUF_LEN);
			uint8_t *tx_buffer = (uint8_t *) pvPortMalloc(plop_COM_RFM22B_RF_TX_BUF_LEN);
			plop_Assert(rx_buffer);
			plop_Assert(tx_buffer);
			if (plop_COM_Init(&plop_com_telem_rf_id, &plop_rfm22b_com_driver, plop_rfm22b_id,
					  rx_buffer, plop_COM_RFM22B_RF_RX_BUF_LEN,
					  tx_buffer, plop_COM_RFM22B_RF_TX_BUF_LEN)) {
				plop_Assert(0);
			}
			break;
		}
	}

#endif /* plop_INCLUDE_RFM22B */

	/* Configure the receiver port*/
	uint8_t hw_rcvrport;
	HwRevoMiniRcvrPortGet(&hw_rcvrport);
	//   
	switch (hw_rcvrport){
		case HWREVOMINI_RCVRPORT_DISABLED:
			break;
		case HWREVOMINI_RCVRPORT_PWM:
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
		case HWREVOMINI_RCVRPORT_PPM:
		case HWREVOMINI_RCVRPORT_PPMOUTPUTS:
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
		case HWREVOMINI_RCVRPORT_OUTPUTS:
		
			break;
	}


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
		case HWREVOMINI_RCVRPORT_DISABLED:
		case HWREVOMINI_RCVRPORT_PWM:
		case HWREVOMINI_RCVRPORT_PPM:
			/* Set up the servo outputs */
			plop_Servo_Init(&plop_servo_cfg);
			break;
		case HWREVOMINI_RCVRPORT_PPMOUTPUTS:
		case HWREVOMINI_RCVRPORT_OUTPUTS:
			//plop_Servo_Init(&plop_servo_rcvr_cfg);
			//TODO: Prepare the configurations on board_hw_defs and handle here:
			plop_Servo_Init(&plop_servo_cfg);
			break;
	}
#else
	plop_DEBUG_Init(&plop_tim_servo_all_channels, NELEMENTS(plop_tim_servo_all_channels));
#endif
	
	if (plop_I2C_Init(&plop_i2c_mag_pressure_adapter_id, &plop_i2c_mag_pressure_adapter_cfg)) {
		plop_DEBUG_Assert(0);
	}
	
	plop_DELAY_WaitmS(50);

	plop_SENSORS_Init();

#if defined(plop_INCLUDE_ADC)
	plop_ADC_Init(&plop_adc_cfg);
        // configure the pullup for PA8 (inhibit pullups from current/sonar shared pin)
        GPIO_Init(plop_current_sonar_pin.gpio, &plop_current_sonar_pin.init);
#endif

#if defined(plop_INCLUDE_HMC5883)
	plop_HMC5883_Init(plop_I2C_MAIN_ADAPTER, &plop_hmc5883_cfg);
#endif
	
#if defined(plop_INCLUDE_MS5611)
	plop_MS5611_Init(&plop_ms5611_cfg, plop_i2c_mag_pressure_adapter_id);
#endif

#if defined(plop_INCLUDE_MPU6000)
	plop_MPU6000_Init(plop_spi_gyro_id,0, &plop_mpu6000_cfg);

	// To be safe map from UAVO enum to driver enum
	uint8_t hw_gyro_range;
	HwRevoMiniGyroRangeGet(&hw_gyro_range);
	switch(hw_gyro_range) {
		case HWREVOMINI_GYRORANGE_250:
			plop_MPU6000_SetGyroRange(plop_MPU60X0_SCALE_250_DEG);
			break;
		case HWREVOMINI_GYRORANGE_500:
			plop_MPU6000_SetGyroRange(plop_MPU60X0_SCALE_500_DEG);
			break;
		case HWREVOMINI_GYRORANGE_1000:
			plop_MPU6000_SetGyroRange(plop_MPU60X0_SCALE_1000_DEG);
			break;
		case HWREVOMINI_GYRORANGE_2000:
			plop_MPU6000_SetGyroRange(plop_MPU60X0_SCALE_2000_DEG);
			break;
	}

	uint8_t hw_accel_range;
	HwRevoMiniAccelRangeGet(&hw_accel_range);
	switch(hw_accel_range) {
		case HWREVOMINI_ACCELRANGE_2G:
			plop_MPU6000_SetAccelRange(plop_MPU60X0_ACCEL_2G);
			break;
		case HWREVOMINI_ACCELRANGE_4G:
			plop_MPU6000_SetAccelRange(plop_MPU60X0_ACCEL_4G);
			break;
		case HWREVOMINI_ACCELRANGE_8G:
			plop_MPU6000_SetAccelRange(plop_MPU60X0_ACCEL_8G);
			break;
		case HWREVOMINI_ACCELRANGE_16G:
			plop_MPU6000_SetAccelRange(plop_MPU60X0_ACCEL_16G);
			break;
	}

	// the filter has to be set before rate else divisor calculation will fail
	uint8_t hw_mpu6000_dlpf;
	HwRevoMiniMPU6000DLPFGet(&hw_mpu6000_dlpf);
	enum plop_mpu60x0_filter mpu6000_dlpf = \
	    (hw_mpu6000_dlpf == HWREVOMINI_MPU6000DLPF_256) ? plop_MPU60X0_LOWPASS_256_HZ : \
	    (hw_mpu6000_dlpf == HWREVOMINI_MPU6000DLPF_188) ? plop_MPU60X0_LOWPASS_188_HZ : \
	    (hw_mpu6000_dlpf == HWREVOMINI_MPU6000DLPF_98) ? plop_MPU60X0_LOWPASS_98_HZ : \
	    (hw_mpu6000_dlpf == HWREVOMINI_MPU6000DLPF_42) ? plop_MPU60X0_LOWPASS_42_HZ : \
	    (hw_mpu6000_dlpf == HWREVOMINI_MPU6000DLPF_20) ? plop_MPU60X0_LOWPASS_20_HZ : \
	    (hw_mpu6000_dlpf == HWREVOMINI_MPU6000DLPF_10) ? plop_MPU60X0_LOWPASS_10_HZ : \
	    (hw_mpu6000_dlpf == HWREVOMINI_MPU6000DLPF_5) ? plop_MPU60X0_LOWPASS_5_HZ : \
	    plop_mpu6000_cfg.default_filter;
	plop_MPU6000_SetLPF(mpu6000_dlpf);

	uint8_t hw_mpu6000_samplerate;
	HwRevoMiniMPU6000RateGet(&hw_mpu6000_samplerate);
	uint16_t mpu6000_samplerate = \
	    (hw_mpu6000_samplerate == HWREVOMINI_MPU6000RATE_200) ? 200 : \
	    (hw_mpu6000_samplerate == HWREVOMINI_MPU6000RATE_333) ? 333 : \
	    (hw_mpu6000_samplerate == HWREVOMINI_MPU6000RATE_500) ? 500 : \
	    (hw_mpu6000_samplerate == HWREVOMINI_MPU6000RATE_666) ? 666 : \
	    (hw_mpu6000_samplerate == HWREVOMINI_MPU6000RATE_1000) ? 1000 : \
	    (hw_mpu6000_samplerate == HWREVOMINI_MPU6000RATE_2000) ? 2000 : \
	    (hw_mpu6000_samplerate == HWREVOMINI_MPU6000RATE_4000) ? 4000 : \
	    (hw_mpu6000_samplerate == HWREVOMINI_MPU6000RATE_8000) ? 8000 : \
	    plop_mpu6000_cfg.default_samplerate;
	plop_MPU6000_SetSampleRate(mpu6000_samplerate);
#endif

}

/**
 * @}
 * @}
 */

