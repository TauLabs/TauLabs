/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup Revolution OpenPilot revolution support files
 * @{
 *
 * @file       pios_board.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      The board specific initialization routines
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

/* Pull in the board-specific static HW definitions.
 * Including .c files is a bit ugly but this allows all of
 * the HW definitions to be const and static to limit their
 * scope.  
 *
 * NOTE: THIS IS THE ONLY PLACE THAT SHOULD EVER INCLUDE THIS FILE
 */
#include "board_hw_defs.c"

#include <pios.h>
#include <openpilot.h>
#include <uavobjectsinit.h>
#include "hwrevolution.h"
#include "modulesettings.h"
#include "manualcontrolsettings.h"

/**
 * Sensor configurations 
 */

#if defined(PIOS_INCLUDE_HMC5883)
#include "pios_hmc5883_priv.h"
static const struct pios_exti_cfg pios_exti_hmc5883_cfg __exti_config = {
	.vector = PIOS_HMC5883_IRQHandler,
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
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_LOW,
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

static const struct pios_hmc5883_cfg pios_hmc5883_cfg = {
	.exti_cfg = &pios_exti_hmc5883_cfg,
	.M_ODR = PIOS_HMC5883_ODR_75,
	.Meas_Conf = PIOS_HMC5883_MEASCONF_NORMAL,
	.Gain = PIOS_HMC5883_GAIN_1_9,
	.Mode = PIOS_HMC5883_MODE_CONTINUOUS,
	.Default_Orientation = PIOS_HMC5883_TOP_270DEG,
};
#endif /* PIOS_INCLUDE_HMC5883 */

/**
 * Configuration for the MS5611 chip
 */
#if defined(PIOS_INCLUDE_MS5611)
#include "pios_ms5611_priv.h"
static const struct pios_ms5611_cfg pios_ms5611_cfg = {
	.oversampling = MS5611_OSR_1024,
	.temperature_interleaving = 1,
};
#endif /* PIOS_INCLUDE_MS5611 */

/**
 * Configuration for the BMA180 chip
 */
#if defined(PIOS_INCLUDE_BMA180)
#include "pios_bma180.h"
static const struct pios_exti_cfg pios_exti_bma180_cfg __exti_config = {
	.vector = PIOS_BMA180_IRQHandler,
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
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_LOW,
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
static const struct pios_bma180_cfg pios_bma180_cfg = {
	.exti_cfg = &pios_exti_bma180_cfg,
	.bandwidth = BMA_BW_600HZ,
	.range = BMA_RANGE_8G,
};
#endif /* PIOS_INCLUDE_BMA180 */

/**
 * Configuration for the MPU6000 chip
 */
#if defined(PIOS_INCLUDE_MPU6000)
#include "pios_mpu6000.h"
static const struct pios_exti_cfg pios_exti_mpu6000_cfg __exti_config = {
	.vector = PIOS_MPU6000_IRQHandler,
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
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGH,
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

static const struct pios_mpu60x0_cfg pios_mpu6000_cfg = {
	.exti_cfg = &pios_exti_mpu6000_cfg,
	.default_samplerate = 666,
	.interrupt_cfg = PIOS_MPU60X0_INT_CLR_ANYRD,
	.interrupt_en = PIOS_MPU60X0_INTEN_DATA_RDY,
	.User_ctl = PIOS_MPU60X0_USERCTL_DIS_I2C,
	.Pwr_mgmt_clk = PIOS_MPU60X0_PWRMGMT_PLL_Z_CLK,
	.default_filter = PIOS_MPU60X0_LOWPASS_256_HZ,
	.orientation = PIOS_MPU60X0_TOP_0DEG
};
#endif /* PIOS_INCLUDE_MPU6000 */

/**
 * Configuration for L3GD20 chip
 */
#if defined(PIOS_INCLUDE_L3GD20)
#include "pios_l3gd20.h"
static const struct pios_exti_cfg pios_exti_l3gd20_cfg __exti_config = {
	.vector = PIOS_L3GD20_IRQHandler,
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
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGH,
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

static const struct pios_l3gd20_cfg pios_l3gd20_cfg = {
	.exti_cfg = &pios_exti_l3gd20_cfg,
	.range = PIOS_L3GD20_SCALE_500_DEG,
};
#endif /* PIOS_INCLUDE_L3GD20 */

/* One slot per selectable receiver group.
 *  eg. PWM, PPM, GCS, SPEKTRUM1, SPEKTRUM2, SBUS
 * NOTE: No slot in this map for NONE.
 */
uintptr_t pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE];

#define PIOS_COM_TELEM_RF_RX_BUF_LEN 512
#define PIOS_COM_TELEM_RF_TX_BUF_LEN 512

#define PIOS_COM_GPS_RX_BUF_LEN 32
#define PIOS_COM_GPS_TX_BUF_LEN 16

#define PIOS_COM_TELEM_USB_RX_BUF_LEN 65
#define PIOS_COM_TELEM_USB_TX_BUF_LEN 65

#define PIOS_COM_BRIDGE_RX_BUF_LEN 65
#define PIOS_COM_BRIDGE_TX_BUF_LEN 12

#define PIOS_COM_MAVLINK_TX_BUF_LEN 128

#define PIOS_COM_HOTT_RX_BUF_LEN 16
#define PIOS_COM_HOTT_TX_BUF_LEN 16

#define PIOS_COM_PICOC_RX_BUF_LEN 128
#define PIOS_COM_PICOC_TX_BUF_LEN 128

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
#define PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN 40
uintptr_t pios_com_debug_id;
#endif /* PIOS_INCLUDE_DEBUG_CONSOLE */

uintptr_t pios_com_gps_id;
uintptr_t pios_com_telem_usb_id;
uintptr_t pios_com_telem_rf_id;
uintptr_t pios_com_vcp_id;
uintptr_t pios_com_bridge_id;
uintptr_t pios_com_overo_id;
uintptr_t pios_com_mavlink_id;
uintptr_t pios_com_hott_id;
uintptr_t pios_com_picoc_id;
uintptr_t pios_internal_adc_id = 0;
uintptr_t pios_uavo_settings_fs_id;
uintptr_t pios_waypoints_settings_fs_id;

/*
 * Setup a com port based on the passed cfg, driver and buffer sizes. rx or tx size of 0 disables rx or tx
 */
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
static void PIOS_Board_configure_com (const struct pios_usart_cfg *usart_port_cfg, size_t rx_buf_len, size_t tx_buf_len,
		const struct pios_com_driver *com_driver, uintptr_t *pios_com_id)
{
	uintptr_t pios_usart_id;
	if (PIOS_USART_Init(&pios_usart_id, usart_port_cfg)) {
		PIOS_Assert(0);
	}

	uint8_t * rx_buffer;
	if (rx_buf_len > 0) {
		rx_buffer = (uint8_t *) pvPortMalloc(rx_buf_len);
		PIOS_Assert(rx_buffer);
	} else {
		rx_buffer = NULL;
	}

	uint8_t * tx_buffer;
	if (tx_buf_len > 0) {
		tx_buffer = (uint8_t *) pvPortMalloc(tx_buf_len);
		PIOS_Assert(tx_buffer);
	} else {
		tx_buffer = NULL;
	}

	if (PIOS_COM_Init(pios_com_id, com_driver, pios_usart_id,
				rx_buffer, rx_buf_len,
				tx_buffer, tx_buf_len)) {
		PIOS_Assert(0);
	}
}
#endif	/* PIOS_INCLUDE_USART && PIOS_INCLUDE_COM */

#ifdef PIOS_INCLUDE_DSM
static void PIOS_Board_configure_dsm(const struct pios_usart_cfg *pios_usart_dsm_cfg, const struct pios_dsm_cfg *pios_dsm_cfg, 
		const struct pios_com_driver *pios_usart_com_driver,enum pios_dsm_proto *proto, 
		ManualControlSettingsChannelGroupsOptions channelgroup,uint8_t *bind)
{
	uintptr_t pios_usart_dsm_id;
	if (PIOS_USART_Init(&pios_usart_dsm_id, pios_usart_dsm_cfg)) {
		PIOS_Assert(0);
	}

	uintptr_t pios_dsm_id;
	if (PIOS_DSM_Init(&pios_dsm_id, pios_dsm_cfg, pios_usart_com_driver,
			pios_usart_dsm_id, *proto, *bind)) {
		PIOS_Assert(0);
	}

	uintptr_t pios_dsm_rcvr_id;
	if (PIOS_RCVR_Init(&pios_dsm_rcvr_id, &pios_dsm_rcvr_driver, pios_dsm_id)) {
		PIOS_Assert(0);
	}
	pios_rcvr_group_map[channelgroup] = pios_dsm_rcvr_id;
}
#endif

#ifdef PIOS_INCLUDE_HSUM
static void PIOS_Board_configure_hsum(const struct pios_usart_cfg *pios_usart_hsum_cfg,
		const struct pios_com_driver *pios_usart_com_driver,enum pios_hsum_proto *proto,
		ManualControlSettingsChannelGroupsOptions channelgroup)
{
	uintptr_t pios_usart_hsum_id;
	if (PIOS_USART_Init(&pios_usart_hsum_id, pios_usart_hsum_cfg)) {
		PIOS_Assert(0);
	}
	
	uintptr_t pios_hsum_id;
	if (PIOS_HSUM_Init(&pios_hsum_id, pios_usart_com_driver,
			   pios_usart_hsum_id, *proto)) {
		PIOS_Assert(0);
	}
	
	uintptr_t pios_hsum_rcvr_id;
	if (PIOS_RCVR_Init(&pios_hsum_rcvr_id, &pios_hsum_rcvr_driver, pios_hsum_id)) {
		PIOS_Assert(0);
	}
	pios_rcvr_group_map[channelgroup] = pios_hsum_rcvr_id;
}
#endif

/**
 * PIOS_Board_Init()
 * initializes all the core subsystems on this specific hardware
 * called from System/openpilot.c
 */

#include <pios_board_info.h>

void PIOS_Board_Init(void) {

	const struct pios_board_info * bdinfo = &pios_board_info_blob;

	/* Delay system */
	PIOS_DELAY_Init();

	PIOS_LED_Init(&pios_led_cfg);

	/* Set up the SPI interface to the accelerometer*/
	if (PIOS_SPI_Init(&pios_spi_accel_id, &pios_spi_accel_cfg)) {
		PIOS_DEBUG_Assert(0);
	}

	/* Set up the SPI interface to the gyro */
	if (PIOS_SPI_Init(&pios_spi_gyro_id, &pios_spi_gyro_cfg)) {
		PIOS_DEBUG_Assert(0);
	}

#if !defined(PIOS_FLASH_ON_ACCEL)
	/* Set up the SPI interface to the flash */
	if (PIOS_SPI_Init(&pios_spi_flash_id, &pios_spi_flash_cfg)) {
		PIOS_DEBUG_Assert(0);
	}
	/* Inititialize all flash drivers */
	PIOS_Flash_Jedec_Init(&pios_external_flash_id, pios_spi_flash_id, 0, &flash_m25p_cfg);
#else
	/* Inititialize all flash drivers */
	PIOS_Flash_Jedec_Init(&pios_external_flash_id, pios_spi_accel_id, 1, &flash_m25p_cfg);
#endif

	PIOS_Flash_Internal_Init(&pios_internal_flash_id, &flash_internal_cfg);

	/* Register the partition table */
	const struct pios_flash_partition * flash_partition_table;
	uint32_t num_partitions;
	flash_partition_table = PIOS_BOARD_HW_DEFS_GetPartitionTable(bdinfo->board_rev, &num_partitions);
	PIOS_FLASH_register_partition_table(flash_partition_table, num_partitions);

	/* Mount all filesystems */
	PIOS_FLASHFS_Logfs_Init(&pios_uavo_settings_fs_id, &flashfs_settings_cfg, FLASH_PARTITION_LABEL_SETTINGS);
	PIOS_FLASHFS_Logfs_Init(&pios_waypoints_settings_fs_id, &flashfs_waypoints_cfg, FLASH_PARTITION_LABEL_WAYPOINTS);

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();

	HwRevolutionInitialize();
	ModuleSettingsInitialize();

#if defined(PIOS_INCLUDE_RTC)
	PIOS_RTC_Init(&pios_rtc_main_cfg);
#endif

	/* Initialize the alarms library */
	AlarmsInitialize();

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Set up pulse timers */
	PIOS_TIM_InitClock(&tim_1_cfg);
	PIOS_TIM_InitClock(&tim_2_cfg);
	PIOS_TIM_InitClock(&tim_3_cfg);
	PIOS_TIM_InitClock(&tim_4_cfg);
	PIOS_TIM_InitClock(&tim_5_cfg);
	PIOS_TIM_InitClock(&tim_9_cfg);
	PIOS_TIM_InitClock(&tim_10_cfg);
	PIOS_TIM_InitClock(&tim_11_cfg);

	/* IAP System Setup */
	PIOS_IAP_Init();
	uint16_t boot_count = PIOS_IAP_ReadBootCount();
	if (boot_count < 3) {
		PIOS_IAP_WriteBootCount(++boot_count);
		AlarmsClear(SYSTEMALARMS_ALARM_BOOTFAULT);
	} else {
		/* Too many failed boot attempts, force hw config to defaults */
		HwRevolutionSetDefaults(HwRevolutionHandle(), 0);
		ModuleSettingsSetDefaults(ModuleSettingsHandle(),0);
		AlarmsSet(SYSTEMALARMS_ALARM_BOOTFAULT, SYSTEMALARMS_ALARM_CRITICAL);
	}
	
	
	//PIOS_IAP_Init();

#if defined(PIOS_INCLUDE_USB)
	/* Initialize board specific USB data */
	PIOS_USB_BOARD_DATA_Init();

	/* Flags to determine if various USB interfaces are advertised */
	bool usb_hid_present = false;
	bool usb_cdc_present = false;

#if defined(PIOS_INCLUDE_USB_CDC)
	if (PIOS_USB_DESC_HID_CDC_Init()) {
		PIOS_Assert(0);
	}
	usb_hid_present = true;
	usb_cdc_present = true;
#else
	if (PIOS_USB_DESC_HID_ONLY_Init()) {
		PIOS_Assert(0);
	}
	usb_hid_present = true;
#endif

	uintptr_t pios_usb_id;
	PIOS_USB_Init(&pios_usb_id, &pios_usb_main_cfg);

#if defined(PIOS_INCLUDE_USB_CDC)

	uint8_t hw_usb_vcpport;
	/* Configure the USB VCP port */
	HwRevolutionUSB_VCPPortGet(&hw_usb_vcpport);

	if (!usb_cdc_present) {
		/* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
		hw_usb_vcpport = HWREVOLUTION_USB_VCPPORT_DISABLED;
	}

	uintptr_t pios_usb_cdc_id;
	if (PIOS_USB_CDC_Init(&pios_usb_cdc_id, &pios_usb_cdc_cfg, pios_usb_id)) {
		PIOS_Assert(0);
	}

	switch (hw_usb_vcpport) {
	case HWREVOLUTION_USB_VCPPORT_DISABLED:
		break;
	case HWREVOLUTION_USB_VCPPORT_USBTELEMETRY:
#if defined(PIOS_INCLUDE_COM)
		{
			uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_TELEM_USB_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_TELEM_USB_TX_BUF_LEN);
			PIOS_Assert(rx_buffer);
			PIOS_Assert(tx_buffer);
			if (PIOS_COM_Init(&pios_com_telem_usb_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
						rx_buffer, PIOS_COM_TELEM_USB_RX_BUF_LEN,
						tx_buffer, PIOS_COM_TELEM_USB_TX_BUF_LEN)) {
				PIOS_Assert(0);
			}
		}
#endif	/* PIOS_INCLUDE_COM */
		break;
	case HWREVOLUTION_USB_VCPPORT_COMBRIDGE:
#if defined(PIOS_INCLUDE_COM)
		{
			uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_BRIDGE_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_BRIDGE_TX_BUF_LEN);
			PIOS_Assert(rx_buffer);
			PIOS_Assert(tx_buffer);
			if (PIOS_COM_Init(&pios_com_vcp_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
						rx_buffer, PIOS_COM_BRIDGE_RX_BUF_LEN,
						tx_buffer, PIOS_COM_BRIDGE_TX_BUF_LEN)) {
				PIOS_Assert(0);
			}
		}
#endif	/* PIOS_INCLUDE_COM */
		break;
	case HWREVOLUTION_USB_VCPPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_COM)
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
		{
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN);
			PIOS_Assert(tx_buffer);
			if (PIOS_COM_Init(&pios_com_debug_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
						NULL, 0,
						tx_buffer, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN)) {
				PIOS_Assert(0);
			}
		}
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
#endif	/* PIOS_INCLUDE_COM */
		break;
	case HWREVOLUTION_USB_VCPPORT_PICOC:
#if defined(PIOS_INCLUDE_COM)
		{
			uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_PICOC_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_PICOC_TX_BUF_LEN);
			PIOS_Assert(rx_buffer);
			PIOS_Assert(tx_buffer);
			if (PIOS_COM_Init(&pios_com_picoc_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
						rx_buffer, PIOS_COM_PICOC_RX_BUF_LEN,
						tx_buffer, PIOS_COM_PICOC_TX_BUF_LEN)) {
				PIOS_Assert(0);
			}
		}
#endif	/* PIOS_INCLUDE_COM */
		break;
	}
#endif	/* PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_USB_HID)
	/* Configure the usb HID port */
	uint8_t hw_usb_hidport;
	HwRevolutionUSB_HIDPortGet(&hw_usb_hidport);

	if (!usb_hid_present) {
		/* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
		hw_usb_hidport = HWREVOLUTION_USB_HIDPORT_DISABLED;
	}

	uintptr_t pios_usb_hid_id;
	if (PIOS_USB_HID_Init(&pios_usb_hid_id, &pios_usb_hid_cfg, pios_usb_id)) {
		PIOS_Assert(0);
	}

	switch (hw_usb_hidport) {
	case HWREVOLUTION_USB_HIDPORT_DISABLED:
		break;
	case HWREVOLUTION_USB_HIDPORT_USBTELEMETRY:
#if defined(PIOS_INCLUDE_COM)
		{
			uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_TELEM_USB_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(PIOS_COM_TELEM_USB_TX_BUF_LEN);
			PIOS_Assert(rx_buffer);
			PIOS_Assert(tx_buffer);
			if (PIOS_COM_Init(&pios_com_telem_usb_id, &pios_usb_hid_com_driver, pios_usb_hid_id,
						rx_buffer, PIOS_COM_TELEM_USB_RX_BUF_LEN,
						tx_buffer, PIOS_COM_TELEM_USB_TX_BUF_LEN)) {
				PIOS_Assert(0);
			}
		}
#endif	/* PIOS_INCLUDE_COM */
		break;
	}

#endif	/* PIOS_INCLUDE_USB_HID */

	if (usb_hid_present || usb_cdc_present) {
		PIOS_USBHOOK_Activate();
	}
#endif	/* PIOS_INCLUDE_USB */

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
			PIOS_Board_configure_com(&pios_usart_telem_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
			break;
		case HWREVOLUTION_TELEMETRYPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
			PIOS_Board_configure_com(&pios_usart_telem_cfg, 0, PIOS_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_debug_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWREVOLUTION_TELEMETRYPORT_COMBRIDGE:
			PIOS_Board_configure_com(&pios_usart_telem_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
			break;
			
	} /* 	hw_telemetryport */

	/* Configure GPS port */
	uint8_t hw_gpsport;
	HwRevolutionGPSPortGet(&hw_gpsport);
	switch (hw_gpsport){
		case HWREVOLUTION_GPSPORT_DISABLED:
			break;
			
		case HWREVOLUTION_GPSPORT_TELEMETRY:
			PIOS_Board_configure_com(&pios_usart_gps_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
			break;
			
		case HWREVOLUTION_GPSPORT_GPS:
			PIOS_Board_configure_com(&pios_usart_gps_cfg, PIOS_COM_GPS_RX_BUF_LEN, PIOS_COM_GPS_RX_BUF_LEN,  &pios_usart_com_driver, &pios_com_gps_id);
			break;
		
		case HWREVOLUTION_GPSPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
			PIOS_Board_configure_com(&pios_usart_gps_cfg, 0, PIOS_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_debug_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
			break;
			
		case HWREVOLUTION_GPSPORT_COMBRIDGE:
			PIOS_Board_configure_com(&pios_usart_gps_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
			break;
	}/* hw_gpsport */

	/* Configure AUXPort */
	uint8_t hw_auxport;
	HwRevolutionAuxPortGet(&hw_auxport);

	switch (hw_auxport) {
		case HWREVOLUTION_AUXPORT_DISABLED:
			break;
			
		case HWREVOLUTION_AUXPORT_TELEMETRY:
			PIOS_Board_configure_com(&pios_usart_aux_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
			break;
			
		case HWREVOLUTION_AUXPORT_DSM2:
		case HWREVOLUTION_AUXPORT_DSMX10BIT:
		case HWREVOLUTION_AUXPORT_DSMX11BIT:
#if defined(PIOS_INCLUDE_DSM)
			{
				enum pios_dsm_proto proto;
				switch (hw_auxport) {
				case HWREVOLUTION_AUXPORT_DSM2:
					proto = PIOS_DSM_PROTO_DSM2;
					break;
				case HWREVOLUTION_AUXPORT_DSMX10BIT:
					proto = PIOS_DSM_PROTO_DSMX10BIT;
					break;
				case HWREVOLUTION_AUXPORT_DSMX11BIT:
					proto = PIOS_DSM_PROTO_DSMX11BIT;
					break;
				default:
					PIOS_Assert(0);
					break;
				}
				//TODO: Define the various Channelgroup for Revo dsm inputs and handle here
				PIOS_Board_configure_dsm(&pios_usart_dsm_hsum_aux_cfg, &pios_dsm_aux_cfg,
					&pios_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT,&hw_DSMxBind);
			}
#endif	/* PIOS_INCLUDE_DSM */
			break;

		case HWREVOLUTION_AUXPORT_HOTTSUMD:
		case HWREVOLUTION_AUXPORT_HOTTSUMH:
#if defined(PIOS_INCLUDE_HSUM)
			{
				enum pios_hsum_proto proto;
				switch (hw_auxport) {
				case HWREVOLUTION_AUXPORT_HOTTSUMD:
					proto = PIOS_HSUM_PROTO_SUMD;
					break;
				case HWREVOLUTION_AUXPORT_HOTTSUMH:
					proto = PIOS_HSUM_PROTO_SUMH;
					break;
				default:
					PIOS_Assert(0);
					break;
				}
				PIOS_Board_configure_hsum(&pios_usart_dsm_hsum_aux_cfg, &pios_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_HOTTSUM);
			}
#endif	/* PIOS_INCLUDE_HSUM */
			break;

		case HWREVOLUTION_AUXPORT_MAVLINKTX:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM) && defined(PIOS_INCLUDE_MAVLINK)
			PIOS_Board_configure_com(&pios_usart_aux_cfg, 0, PIOS_COM_MAVLINK_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_mavlink_id);
#endif	/* PIOS_INCLUDE_MAVLINK */
			break;

		case HWREVOLUTION_AUXPORT_MAVLINKTX_GPS_RX:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM) && defined(PIOS_INCLUDE_MAVLINK) && defined(PIOS_INCLUDE_GPS)
			PIOS_Board_configure_com(&pios_usart_aux_cfg, PIOS_COM_GPS_RX_BUF_LEN, PIOS_COM_MAVLINK_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_gps_id);
			pios_com_mavlink_id = pios_com_gps_id;
#endif	/* PIOS_INCLUDE_MAVLINK */
			break;

		case HWREVOLUTION_AUXPORT_HOTTTELEMETRY:
#if defined(PIOS_INCLUDE_HOTT) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usart_aux_cfg, PIOS_COM_HOTT_RX_BUF_LEN, PIOS_COM_HOTT_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_hott_id);
#endif /* PIOS_INCLUDE_HOTT */
			break;

		case HWREVOLUTION_AUXPORT_PICOC:
#if defined(PIOS_INCLUDE_PICOC) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usart_aux_cfg, PIOS_COM_PICOC_RX_BUF_LEN, PIOS_COM_PICOC_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_picoc_id);
#endif /* PIOS_INCLUDE_PICOC */
			break;

		case HWREVOLUTION_AUXPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
			PIOS_Board_configure_com(&pios_usart_aux_cfg, 0, PIOS_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_debug_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWREVOLUTION_AUXPORT_COMBRIDGE:
			PIOS_Board_configure_com(&pios_usart_aux_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
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
#ifdef PIOS_INCLUDE_SBUS
		{
			uintptr_t pios_usart_sbus_id;
			if (PIOS_USART_Init(&pios_usart_sbus_id, &pios_usart_sbus_auxsbus_cfg)) {
				PIOS_Assert(0);
			}
			
			uintptr_t pios_sbus_id;
			if (PIOS_SBus_Init(&pios_sbus_id, &pios_sbus_cfg, &pios_usart_com_driver, pios_usart_sbus_id)) {
				PIOS_Assert(0);
			}
			
			uintptr_t pios_sbus_rcvr_id;
			if (PIOS_RCVR_Init(&pios_sbus_rcvr_id, &pios_sbus_rcvr_driver, pios_sbus_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_SBUS] = pios_sbus_rcvr_id;
			
		}
#endif /* PIOS_INCLUDE_SBUS */
			break;

		case HWREVOLUTION_AUXSBUSPORT_DSM2:
		case HWREVOLUTION_AUXSBUSPORT_DSMX10BIT:
		case HWREVOLUTION_AUXSBUSPORT_DSMX11BIT:
#if defined(PIOS_INCLUDE_DSM)
			{
				enum pios_dsm_proto proto;
				switch (hw_auxsbusport) {
				case HWREVOLUTION_AUXSBUSPORT_DSM2:
					proto = PIOS_DSM_PROTO_DSM2;
					break;
				case HWREVOLUTION_AUXSBUSPORT_DSMX10BIT:
					proto = PIOS_DSM_PROTO_DSMX10BIT;
					break;
				case HWREVOLUTION_AUXSBUSPORT_DSMX11BIT:
					proto = PIOS_DSM_PROTO_DSMX11BIT;
					break;
				default:
					PIOS_Assert(0);
					break;
				}
				//TODO: Define the various Channelgroup for Revo dsm inputs and handle here
				PIOS_Board_configure_dsm(&pios_usart_dsm_hsum_auxsbus_cfg, &pios_dsm_auxsbus_cfg,
					&pios_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT,&hw_DSMxBind);
			}
#endif	/* PIOS_INCLUDE_DSM */
			break;
			
		case HWREVOLUTION_AUXSBUSPORT_HOTTSUMD:
		case HWREVOLUTION_AUXSBUSPORT_HOTTSUMH:
#if defined(PIOS_INCLUDE_HSUM)
			{
				enum pios_hsum_proto proto;
				switch (hw_auxsbusport) {
				case HWREVOLUTION_AUXSBUSPORT_HOTTSUMD:
					proto = PIOS_HSUM_PROTO_SUMD;
					break;
				case HWREVOLUTION_AUXSBUSPORT_HOTTSUMH:
					proto = PIOS_HSUM_PROTO_SUMH;
					break;
				default:
					PIOS_Assert(0);
					break;
				}
			PIOS_Board_configure_hsum(&pios_usart_dsm_hsum_auxsbus_cfg, &pios_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_HOTTSUM);
			}
#endif	/* PIOS_INCLUDE_HSUM */
			break;

		case HWREVOLUTION_AUXSBUSPORT_MAVLINKTX:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM) && defined(PIOS_INCLUDE_MAVLINK)
			PIOS_Board_configure_com(&pios_usart_auxsbus_cfg, 0, PIOS_COM_MAVLINK_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_mavlink_id);
#endif	/* PIOS_INCLUDE_MAVLINK */
			break;

		case HWREVOLUTION_AUXSBUSPORT_MAVLINKTX_GPS_RX:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM) && defined(PIOS_INCLUDE_MAVLINK) && defined(PIOS_INCLUDE_GPS)
			PIOS_Board_configure_com(&pios_usart_auxsbus_cfg, PIOS_COM_GPS_RX_BUF_LEN, PIOS_COM_MAVLINK_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_gps_id);
			pios_com_mavlink_id = pios_com_gps_id;
#endif	/* PIOS_INCLUDE_MAVLINK */
			break;

		case HWREVOLUTION_AUXSBUSPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
			PIOS_Board_configure_com(&pios_usart_auxsbus_cfg, 0, PIOS_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_debug_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWREVOLUTION_AUXSBUSPORT_COMBRIDGE:
			PIOS_Board_configure_com(&pios_usart_auxsbus_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
			break;

		case HWREVOLUTION_AUXSBUSPORT_HOTTTELEMETRY:
#if defined(PIOS_INCLUDE_HOTT) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usart_auxsbus_cfg, PIOS_COM_HOTT_RX_BUF_LEN, PIOS_COM_HOTT_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_hott_id);
#endif /* PIOS_INCLUDE_HOTT */
			break;

		case HWREVOLUTION_AUXSBUSPORT_PICOC:
#if defined(PIOS_INCLUDE_PICOC) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usart_auxsbus_cfg, PIOS_COM_PICOC_RX_BUF_LEN, PIOS_COM_PICOC_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_picoc_id);
#endif /* PIOS_INCLUDE_PICOC */
			break;
	} /* hw_auxport */
	
	/* Configure FlexiPort */

	uint8_t hw_flexiport;
	HwRevolutionFlexiPortGet(&hw_flexiport);
	
	switch (hw_flexiport) {
		case HWREVOLUTION_FLEXIPORT_DISABLED:
			break;
		case HWREVOLUTION_FLEXIPORT_I2C:
#if defined(PIOS_INCLUDE_I2C)
		{
			if (PIOS_I2C_Init(&pios_i2c_flexiport_adapter_id, &pios_i2c_flexiport_adapter_cfg)) {
				PIOS_Assert(0);
			}
		}
#endif	/* PIOS_INCLUDE_I2C */
			break;
			
		case HWREVOLUTION_FLEXIPORT_DSM2:
		case HWREVOLUTION_FLEXIPORT_DSMX10BIT:
		case HWREVOLUTION_FLEXIPORT_DSMX11BIT:
#if defined(PIOS_INCLUDE_DSM)
			{
				enum pios_dsm_proto proto;
				switch (hw_flexiport) {
				case HWREVOLUTION_FLEXIPORT_DSM2:
					proto = PIOS_DSM_PROTO_DSM2;
					break;
				case HWREVOLUTION_FLEXIPORT_DSMX10BIT:
					proto = PIOS_DSM_PROTO_DSMX10BIT;
					break;
				case HWREVOLUTION_FLEXIPORT_DSMX11BIT:
					proto = PIOS_DSM_PROTO_DSMX11BIT;
					break;
				default:
					PIOS_Assert(0);
					break;
				}
				//TODO: Define the various Channelgroup for Revo dsm inputs and handle here
				PIOS_Board_configure_dsm(&pios_usart_dsm_hsum_flexi_cfg, &pios_dsm_flexi_cfg,
					&pios_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT,&hw_DSMxBind);
			}
#endif	/* PIOS_INCLUDE_DSM */
			break;
			
		case HWREVOLUTION_FLEXIPORT_HOTTSUMD:
		case HWREVOLUTION_FLEXIPORT_HOTTSUMH:
#if defined(PIOS_INCLUDE_HSUM)
			{
				enum pios_hsum_proto proto;
				switch (hw_flexiport) {
				case HWREVOLUTION_FLEXIPORT_HOTTSUMD:
					proto = PIOS_HSUM_PROTO_SUMD;
					break;
				case HWREVOLUTION_FLEXIPORT_HOTTSUMH:
					proto = PIOS_HSUM_PROTO_SUMH;
					break;
				default:
					PIOS_Assert(0);
					break;
				}
			PIOS_Board_configure_hsum(&pios_usart_dsm_hsum_flexi_cfg, &pios_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_HOTTSUM);
			}
#endif	/* PIOS_INCLUDE_HSUM */
			break;

		case HWREVOLUTION_FLEXIPORT_MAVLINKTX:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM) && defined(PIOS_INCLUDE_MAVLINK)
			PIOS_Board_configure_com(&pios_usart_flexi_cfg, 0, PIOS_COM_MAVLINK_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_mavlink_id);
#endif	/* PIOS_INCLUDE_MAVLINK */
			break;

		case HWREVOLUTION_FLEXIPORT_MAVLINKTX_GPS_RX:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM) && defined(PIOS_INCLUDE_MAVLINK) && defined(PIOS_INCLUDE_GPS)
			PIOS_Board_configure_com(&pios_usart_flexi_cfg, PIOS_COM_GPS_RX_BUF_LEN, PIOS_COM_MAVLINK_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_gps_id);
			pios_com_mavlink_id = pios_com_gps_id;
#endif	/* PIOS_INCLUDE_MAVLINK */
			break;

		case HWREVOLUTION_FLEXIPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
			PIOS_Board_configure_com(&pios_usart_flexi_cfg, 0, PIOS_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_aux_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWREVOLUTION_FLEXIPORT_COMBRIDGE:
			PIOS_Board_configure_com(&pios_usart_flexi_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
			break;
		case HWREVOLUTION_FLEXIPORT_HOTTTELEMETRY:
#if defined(PIOS_INCLUDE_HOTT) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usart_flexi_cfg, PIOS_COM_HOTT_RX_BUF_LEN, PIOS_COM_HOTT_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_hott_id);
#endif /* PIOS_INCLUDE_HOTT */
			break;

		case HWREVOLUTION_FLEXIPORT_PICOC:
#if defined(PIOS_INCLUDE_PICOC) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usart_flexi_cfg, PIOS_COM_PICOC_RX_BUF_LEN, PIOS_COM_PICOC_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_picoc_id);
#endif /* PIOS_INCLUDE_PICOC */
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
#if defined(PIOS_INCLUDE_PWM)
		{
			/* Set up the receiver port.  Later this should be optional */
			uintptr_t pios_pwm_id;
			PIOS_PWM_Init(&pios_pwm_id, &pios_pwm_cfg);
			
			uintptr_t pios_pwm_rcvr_id;
			if (PIOS_RCVR_Init(&pios_pwm_rcvr_id, &pios_pwm_rcvr_driver, pios_pwm_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PWM] = pios_pwm_rcvr_id;
		}
#endif	/* PIOS_INCLUDE_PWM */
			break;
		case HWREVOLUTION_RCVRPORT_PPM:
		case HWREVOLUTION_RCVRPORT_PPMOUTPUTS:
#if defined(PIOS_INCLUDE_PPM)
		{
			uintptr_t pios_ppm_id;
			PIOS_PPM_Init(&pios_ppm_id, &pios_ppm_cfg);
			
			uintptr_t pios_ppm_rcvr_id;
			if (PIOS_RCVR_Init(&pios_ppm_rcvr_id, &pios_ppm_rcvr_driver, pios_ppm_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PPM] = pios_ppm_rcvr_id;
		}
#endif	/* PIOS_INCLUDE_PPM */
		case HWREVOLUTION_RCVRPORT_OUTPUTS:
		
			break;
	}

#if defined(PIOS_OVERO_SPI)
	/* Set up the SPI based PIOS_COM interface to the overo */
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
			if (PIOS_OVERO_Init(&pios_overo_id, &pios_overo_cfg)) {
				PIOS_DEBUG_Assert(0);
			}
			const uint32_t PACKET_SIZE = 1024;
			uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(PACKET_SIZE);
			uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(PACKET_SIZE);
			PIOS_Assert(rx_buffer);
			PIOS_Assert(tx_buffer);
			if (PIOS_COM_Init(&pios_com_overo_id, &pios_overo_com_driver, pios_overo_id,
							  rx_buffer, PACKET_SIZE,
							  tx_buffer, PACKET_SIZE)) {
				PIOS_Assert(0);
			}
		}
	}

#endif

#if defined(PIOS_INCLUDE_GCSRCVR)
	GCSReceiverInitialize();
	uintptr_t pios_gcsrcvr_id;
	PIOS_GCSRCVR_Init(&pios_gcsrcvr_id);
	uintptr_t pios_gcsrcvr_rcvr_id;
	if (PIOS_RCVR_Init(&pios_gcsrcvr_rcvr_id, &pios_gcsrcvr_rcvr_driver, pios_gcsrcvr_id)) {
		PIOS_Assert(0);
	}
	pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_GCS] = pios_gcsrcvr_rcvr_id;
#endif	/* PIOS_INCLUDE_GCSRCVR */
	
#ifndef PIOS_DEBUG_ENABLE_DEBUG_PINS
	switch (hw_rcvrport) {
		case HWREVOLUTION_RCVRPORT_DISABLED:
		case HWREVOLUTION_RCVRPORT_PWM:
		case HWREVOLUTION_RCVRPORT_PPM:
			/* Set up the servo outputs */
			PIOS_Servo_Init(&pios_servo_cfg);
			break;
		case HWREVOLUTION_RCVRPORT_PPMOUTPUTS:
		case HWREVOLUTION_RCVRPORT_OUTPUTS:
			//PIOS_Servo_Init(&pios_servo_rcvr_cfg);
			//TODO: Prepare the configurations on board_hw_defs and handle here:
			PIOS_Servo_Init(&pios_servo_cfg);
			break;
	}
#else
	PIOS_DEBUG_Init(&pios_tim_servo_all_channels, NELEMENTS(pios_tim_servo_all_channels));
#endif
	
	if (PIOS_I2C_Init(&pios_i2c_mag_adapter_id, &pios_i2c_mag_adapter_cfg)) {
		PIOS_DEBUG_Assert(0);
	}

	if (PIOS_I2C_Init(&pios_i2c_pressure_adapter_id, &pios_i2c_pressure_adapter_cfg)) {
		PIOS_DEBUG_Assert(0);
	}
	
	PIOS_DELAY_WaitmS(50);

	PIOS_SENSORS_Init();

#if defined(PIOS_INCLUDE_ADC)
	uint32_t internal_adc_id;
	if(PIOS_INTERNAL_ADC_Init(&internal_adc_id, &pios_adc_cfg) < 0)
	        PIOS_Assert(0);
	PIOS_ADC_Init(&pios_internal_adc_id, &pios_internal_adc_driver, internal_adc_id);
#endif

#if defined(PIOS_INCLUDE_HMC5883)
	PIOS_HMC5883_Init(PIOS_I2C_MAIN_ADAPTER, &pios_hmc5883_cfg);
#endif
	
#if defined(PIOS_INCLUDE_MS5611)
	PIOS_MS5611_Init(&pios_ms5611_cfg, pios_i2c_pressure_adapter_id);
#endif

	switch(bdinfo->board_rev) {
		case 0x01:
#if defined(PIOS_INCLUDE_L3GD20)
			PIOS_L3GD20_Init(pios_spi_gyro_id, 0, &pios_l3gd20_cfg);
			PIOS_Assert(PIOS_L3GD20_Test() == 0);
#endif
#if defined(PIOS_INCLUDE_BMA180)
			PIOS_BMA180_Init(pios_spi_accel_id, 0, &pios_bma180_cfg);
			PIOS_Assert(PIOS_BMA180_Test() == 0);
#endif
			break;
		case 0x02:
#if defined(PIOS_INCLUDE_MPU6000)
			PIOS_MPU6000_Init(pios_spi_gyro_id,0, &pios_mpu6000_cfg);

			// To be safe map from UAVO enum to driver enum
			uint8_t hw_gyro_range;
			HwRevolutionGyroRangeGet(&hw_gyro_range);
			switch(hw_gyro_range) {
				case HWREVOLUTION_GYRORANGE_250:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_250_DEG);
					break;
				case HWREVOLUTION_GYRORANGE_500:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);
					break;
				case HWREVOLUTION_GYRORANGE_1000:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_1000_DEG);
					break;
				case HWREVOLUTION_GYRORANGE_2000:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_2000_DEG);
					break;
			}

			uint8_t hw_accel_range;
			HwRevolutionAccelRangeGet(&hw_accel_range);
			switch(hw_accel_range) {
				case HWREVOLUTION_ACCELRANGE_2G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_2G);
					break;
				case HWREVOLUTION_ACCELRANGE_4G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_4G);
					break;
				case HWREVOLUTION_ACCELRANGE_8G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
					break;
				case HWREVOLUTION_ACCELRANGE_16G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_16G);
					break;
			}

			// the filter has to be set before rate else divisor calculation will fail
			uint8_t hw_mpu6000_dlpf;
			HwRevolutionMPU6000DLPFGet(&hw_mpu6000_dlpf);
			enum pios_mpu60x0_filter mpu6000_dlpf = \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_256) ? PIOS_MPU60X0_LOWPASS_256_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_188) ? PIOS_MPU60X0_LOWPASS_188_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_98) ? PIOS_MPU60X0_LOWPASS_98_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_42) ? PIOS_MPU60X0_LOWPASS_42_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_20) ? PIOS_MPU60X0_LOWPASS_20_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_10) ? PIOS_MPU60X0_LOWPASS_10_HZ : \
			    (hw_mpu6000_dlpf == HWREVOLUTION_MPU6000DLPF_5) ? PIOS_MPU60X0_LOWPASS_5_HZ : \
			    pios_mpu6000_cfg.default_filter;
			PIOS_MPU6000_SetLPF(mpu6000_dlpf);

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
			    pios_mpu6000_cfg.default_samplerate;
			PIOS_MPU6000_SetSampleRate(mpu6000_samplerate);
#endif
			break;
		default:
			PIOS_DEBUG_Assert(0);
	}

}

/**
 * @}
 * @}
 */

