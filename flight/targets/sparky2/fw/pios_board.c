/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup Sparky2 Tau Labs Sparky2 support files
 * @{
 *
 * @file       pios_board.c 
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @brief      Board initialization file
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
#include "hwsparky2.h"
#include "manualcontrolsettings.h"
#include "modulesettings.h"
#include <rfm22bstatus.h>
#include <rfm22breceiver.h>
#include <pios_rfm22b_rcvr_priv.h>
#include <pios_openlrs_rcvr_priv.h>

/**
 * Sensor configurations 
 */

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
 * Configuration for the MPU9250 chip
 */
#if defined(PIOS_INCLUDE_MPU9250_SPI)
#include "pios_mpu9250.h"
static const struct pios_exti_cfg pios_exti_mpu9250_cfg __exti_config = {
	.vector = PIOS_MPU9250_IRQHandler,
	.line = EXTI_Line5,
	.pin = {
		.gpio = GPIOC,
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
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGH,
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

static struct pios_mpu9250_cfg pios_mpu9250_cfg = {
	.exti_cfg = &pios_exti_mpu9250_cfg,
	.default_samplerate = 500,
	.interrupt_cfg = PIOS_MPU60X0_INT_CLR_ANYRD,

	.use_magnetometer = true,
	.default_gyro_filter = PIOS_MPU9250_GYRO_LOWPASS_184_HZ,
	.default_accel_filter = PIOS_MPU9250_ACCEL_LOWPASS_184_HZ,
	.orientation = PIOS_MPU9250_TOP_180DEG
};
#endif /* PIOS_INCLUDE_MPU9250_SPI */

/**
 * Configuration for the external HMC5883 chip
 */
#if defined(PIOS_INCLUDE_HMC5883)
#include "pios_hmc5883_priv.h"
static const struct pios_hmc5883_cfg pios_hmc5883_external_cfg = {
	.exti_cfg            = NULL,
	.M_ODR               = PIOS_HMC5883_ODR_75,
	.Meas_Conf           = PIOS_HMC5883_MEASCONF_NORMAL,
	.Gain                = PIOS_HMC5883_GAIN_1_9,
	.Mode                = PIOS_HMC5883_MODE_SINGLE,
	.Default_Orientation = PIOS_HMC5883_TOP_0DEG,
};
#endif /* PIOS_INCLUDE_HMC5883 */

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

#define PIOS_COM_FRSKYSENSORHUB_TX_BUF_LEN 128

#define PIOS_COM_LIGHTTELEMETRY_TX_BUF_LEN 19

#define PIOS_COM_PICOC_RX_BUF_LEN 128
#define PIOS_COM_PICOC_TX_BUF_LEN 128

#define PIOS_COM_RFM22B_RF_RX_BUF_LEN 512
#define PIOS_COM_RFM22B_RF_TX_BUF_LEN 512

#define PIOS_COM_CAN_RX_BUF_LEN 256
#define PIOS_COM_CAN_TX_BUF_LEN 256

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
#define PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN 40
uintptr_t pios_com_debug_id;
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */

uintptr_t pios_com_gps_id;
uintptr_t pios_com_telem_usb_id;
uintptr_t pios_com_telem_rf_id;
uintptr_t pios_com_vcp_id;
uintptr_t pios_com_bridge_id;
uintptr_t pios_com_overo_id;
uintptr_t pios_com_mavlink_id;
uintptr_t pios_com_hott_id;
uintptr_t pios_com_frsky_sensor_hub_id;
uintptr_t pios_com_lighttelemetry_id;
uintptr_t pios_com_picoc_id;
uintptr_t pios_com_logging_id;
uintptr_t pios_com_rf_id;
uintptr_t pios_com_can_id;
uint32_t pios_rfm22b_id;
uintptr_t pios_internal_adc_id = 0;
uintptr_t pios_uavo_settings_fs_id;
uintptr_t pios_waypoints_settings_fs_id;

uintptr_t pios_can_id;

uintptr_t streamfs_id;

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
		rx_buffer = (uint8_t *) PIOS_malloc(rx_buf_len);
		PIOS_Assert(rx_buffer);
	} else {
		rx_buffer = NULL;
	}

	uint8_t * tx_buffer;
	if (tx_buf_len > 0) {
		tx_buffer = (uint8_t *) PIOS_malloc(tx_buf_len);
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
static void PIOS_Board_configure_dsm(const struct pios_usart_cfg *pios_usart_dsm_cfg,
		const struct pios_dsm_cfg *pios_dsm_cfg,
		const struct pios_com_driver *pios_usart_com_driver,
		ManualControlSettingsChannelGroupsOptions channelgroup, uint8_t *bind)
{
	uintptr_t pios_usart_dsm_id;
	if (PIOS_USART_Init(&pios_usart_dsm_id, pios_usart_dsm_cfg)) {
		PIOS_Assert(0);
	}
	
	uintptr_t pios_dsm_id;
	if (PIOS_DSM_Init(&pios_dsm_id, pios_dsm_cfg, pios_usart_com_driver,
			pios_usart_dsm_id, *bind)) {
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
 * Indicate a target-specific error code when a component fails to initialize
 * 1 pulse - flash chip
 * 2 pulses - MPU9250
 * 4 pulses - MS5611
 * 6 pulses - external mag
 */
static void panic(int32_t code) {
	while(1){
		for (int32_t i = 0; i < code; i++) {
			PIOS_WDG_Clear();
			PIOS_LED_Toggle(PIOS_LED_ALARM);
			PIOS_DELAY_WaitmS(200);
			PIOS_WDG_Clear();
			PIOS_LED_Toggle(PIOS_LED_ALARM);
			PIOS_DELAY_WaitmS(200);
		}
		PIOS_DELAY_WaitmS(200);
		PIOS_WDG_Clear();
		PIOS_DELAY_WaitmS(200);
		PIOS_WDG_Clear();
		PIOS_DELAY_WaitmS(100);
		PIOS_WDG_Clear();
	}
}

void set_vtx_channel(HwSparky2VTX_ChOptions channel)
{
	uint8_t chan = 0;
	switch (channel) {
	case HWSPARKY2_VTX_CH_1:
		chan = 0;
		break;
	case HWSPARKY2_VTX_CH_2:
		chan = 1;
		break;
	case HWSPARKY2_VTX_CH_3:
		chan = 2;
		break;
	case HWSPARKY2_VTX_CH_4:
		chan = 3;
		break;
	case HWSPARKY2_VTX_CH_5:
		chan = 4;
		break;
	case HWSPARKY2_VTX_CH_6:
		chan = 5;
		break;
	case HWSPARKY2_VTX_CH_7:
		chan = 6;
		break;
	case HWSPARKY2_VTX_CH_8:
		chan = 7;
		break;
	}

	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_OD;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;

	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_12;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_13;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_14;
	GPIO_Init(GPIOB, &GPIO_InitStructure);

	if (chan & 0x01) {
		GPIO_SetBits(GPIOB, GPIO_Pin_14);
	} else {
		GPIO_ResetBits(GPIOB, GPIO_Pin_14);
	}

	if (chan & 0x02) {
		GPIO_SetBits(GPIOB, GPIO_Pin_13);
	} else {
		GPIO_ResetBits(GPIOB, GPIO_Pin_13);
	}

	if (chan & 0x04) {
		GPIO_SetBits(GPIOB, GPIO_Pin_12);
	} else {
		GPIO_ResetBits(GPIOB, GPIO_Pin_12);
	}
}

/**
 * PIOS_Board_Init()
 * initializes all the core subsystems on this specific hardware
 * called from System/openpilot.c
 */

#include <pios_board_info.h>

/**
 * Check the brown out reset threshold is 2.7 volts and if not
 * resets it.  This solves an issue that can prevent boards
 * powering up with some BEC
 */
void check_bor()
{
    uint8_t bor = FLASH_OB_GetBOR();

    if (bor != OB_BOR_LEVEL3) {
        FLASH_OB_Unlock();
        FLASH_OB_BORConfig(OB_BOR_LEVEL3);
        FLASH_OB_Launch();
        while (FLASH_WaitForLastOperation() == FLASH_BUSY) {
            ;
        }
        FLASH_OB_Lock();
        while (FLASH_WaitForLastOperation() == FLASH_BUSY) {
            ;
        }
    }
}

void PIOS_Board_Init(void) {

	check_bor();

	const struct pios_board_info * bdinfo = &pios_board_info_blob;

	// Make sure all the PWM outputs are low
	const struct pios_servo_cfg * servo_cfg = get_servo_cfg(bdinfo->board_rev);
	const struct pios_tim_channel * channels = servo_cfg->channels;
	uint8_t num_channels = servo_cfg->num_channels;
	for (int i = 0; i < num_channels; i++) {
		GPIO_Init(channels[i].pin.gpio, (GPIO_InitTypeDef*) &channels[i].pin.init);
	}

	/* Delay system */
	PIOS_DELAY_Init();

#if defined(PIOS_INCLUDE_LED)
	const struct pios_led_cfg * led_cfg = PIOS_BOARD_HW_DEFS_GetLedCfg(bdinfo->board_rev);
	PIOS_Assert(led_cfg);
	PIOS_LED_Init(led_cfg);
#endif	/* PIOS_INCLUDE_LED */

	/* Set up the SPI interface to the gyro/acelerometer */
	if (PIOS_SPI_Init(&pios_spi_gyro_id, &pios_spi_gyro_cfg)) {
		PIOS_DEBUG_Assert(0);
	}

	/* Set up the SPI interface to the flash and rfm22b */
	if (PIOS_SPI_Init(&pios_spi_telem_flash_id, &pios_spi_telem_flash_cfg)) {
		PIOS_DEBUG_Assert(0);
	}

#if defined(PIOS_INCLUDE_FLASH)
	/* Inititialize all flash drivers */
#if defined(PIOS_INCLUDE_FLASH_JEDEC)
	if (get_external_flash(bdinfo->board_rev)) {
		if (PIOS_Flash_Jedec_Init(&pios_external_flash_id, pios_spi_telem_flash_id, 1, &flash_m25p_cfg) != 0)
			panic(1);
	}
#endif /* PIOS_INCLUDE_FLASH_JEDEC */

	PIOS_Flash_Internal_Init(&pios_internal_flash_id, &flash_internal_cfg);

	/* Register the partition table */
	const struct pios_flash_partition * flash_partition_table;
	uint32_t num_partitions;
	flash_partition_table = PIOS_BOARD_HW_DEFS_GetPartitionTable(bdinfo->board_rev, &num_partitions);
	PIOS_FLASH_register_partition_table(flash_partition_table, num_partitions);

	/* Mount all filesystems */
	if (PIOS_FLASHFS_Logfs_Init(&pios_uavo_settings_fs_id, get_flashfs_settings_cfg(bdinfo->board_rev), FLASH_PARTITION_LABEL_SETTINGS))
		panic(1);
#if defined(PIOS_INCLUDE_FLASH_JEDEC)
	if (get_external_flash(bdinfo->board_rev)) {
		if (PIOS_FLASHFS_Logfs_Init(&pios_waypoints_settings_fs_id, &flashfs_waypoints_cfg, FLASH_PARTITION_LABEL_WAYPOINTS) != 0)
			panic(1);
	}
#endif /* PIOS_INCLUDE_FLASH_JEDEC */

#if defined(ERASE_FLASH)
	PIOS_FLASHFS_Format(pios_uavo_settings_fs_id);
#endif

#endif	/* PIOS_INCLUDE_FLASH */

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();

	HwSparky2Initialize();
	ModuleSettingsInitialize();

#if defined(PIOS_INCLUDE_RTC)
	PIOS_RTC_Init(&pios_rtc_main_cfg);
#endif

#ifndef ERASE_FLASH
	/* Initialize watchdog as early as possible to catch faults during init
	 * but do it only if there is no debugger connected
	 */
	if ((CoreDebug->DHCSR & CoreDebug_DHCSR_C_DEBUGEN_Msk) == 0) {
		PIOS_WDG_Init();
	}
#endif

	/* Initialize the alarms library */
	AlarmsInitialize();

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Set up pulse timers */
	PIOS_TIM_InitClock(&tim_3_cfg);
	PIOS_TIM_InitClock(&tim_5_cfg);
	PIOS_TIM_InitClock(&tim_8_cfg);
	PIOS_TIM_InitClock(&tim_9_cfg);
	PIOS_TIM_InitClock(&tim_12_cfg);

	NVIC_InitTypeDef tim_8_up_irq = {
		.NVIC_IRQChannel                   = TIM8_UP_TIM13_IRQn,
		.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
		.NVIC_IRQChannelSubPriority        = 0,
		.NVIC_IRQChannelCmd                = ENABLE,
	};
	NVIC_Init(&tim_8_up_irq);

	/* IAP System Setup */
	PIOS_IAP_Init();
	uint16_t boot_count = PIOS_IAP_ReadBootCount();
	if (boot_count < 3) {
		PIOS_IAP_WriteBootCount(++boot_count);
		AlarmsClear(SYSTEMALARMS_ALARM_BOOTFAULT);
	} else {
		/* Too many failed boot attempts, force hw config to defaults */
		HwSparky2SetDefaults(HwSparky2Handle(), 0);
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
	PIOS_USB_Init(&pios_usb_id, PIOS_BOARD_HW_DEFS_GetUsbCfg(bdinfo->board_rev));

#if defined(PIOS_INCLUDE_USB_CDC)

	uint8_t hw_usb_vcpport;
	/* Configure the USB VCP port */
	HwSparky2USB_VCPPortGet(&hw_usb_vcpport);

	if (!usb_cdc_present) {
		/* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
		hw_usb_vcpport = HWSPARKY2_USB_VCPPORT_DISABLED;
	}

	uintptr_t pios_usb_cdc_id;
	if (PIOS_USB_CDC_Init(&pios_usb_cdc_id, &pios_usb_cdc_cfg, pios_usb_id)) {
		PIOS_Assert(0);
	}

	switch (hw_usb_vcpport) {
	case HWSPARKY2_USB_VCPPORT_DISABLED:
		break;
	case HWSPARKY2_USB_VCPPORT_USBTELEMETRY:
#if defined(PIOS_INCLUDE_COM)
		{
			uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_TELEM_USB_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_TELEM_USB_TX_BUF_LEN);
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
	case HWSPARKY2_USB_VCPPORT_COMBRIDGE:
#if defined(PIOS_INCLUDE_COM)
		{
			uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_BRIDGE_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_BRIDGE_TX_BUF_LEN);
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
	case HWSPARKY2_USB_VCPPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_COM)
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
		{
			uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN);
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
	case HWSPARKY2_USB_VCPPORT_PICOC:
#if defined(PIOS_INCLUDE_COM)
		{
			uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_PICOC_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_PICOC_TX_BUF_LEN);
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
	HwSparky2USB_HIDPortGet(&hw_usb_hidport);

	if (!usb_hid_present) {
		/* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
		hw_usb_hidport = HWSPARKY2_USB_HIDPORT_DISABLED;
	}

	uintptr_t pios_usb_hid_id;
	if (PIOS_USB_HID_Init(&pios_usb_hid_id, &pios_usb_hid_cfg, pios_usb_id)) {
		PIOS_Assert(0);
	}

	switch (hw_usb_hidport) {
	case HWSPARKY2_USB_HIDPORT_DISABLED:
		break;
	case HWSPARKY2_USB_HIDPORT_USBTELEMETRY:
#if defined(PIOS_INCLUDE_COM)
		{
			uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_TELEM_USB_RX_BUF_LEN);
			uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_TELEM_USB_TX_BUF_LEN);
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
	HwSparky2DSMxBindGet(&hw_DSMxBind);
	
	/* Configure main USART port */
	uint8_t hw_mainport;
	HwSparky2MainPortGet(&hw_mainport);
	switch (hw_mainport) {
		case HWSPARKY2_MAINPORT_DISABLED:
			break;
		case HWSPARKY2_MAINPORT_TELEMETRY:
			PIOS_Board_configure_com(&pios_usart_main_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
			break;
		case HWSPARKY2_MAINPORT_GPS:
			PIOS_Board_configure_com(&pios_usart_main_cfg, PIOS_COM_GPS_RX_BUF_LEN, PIOS_COM_GPS_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_gps_id);
			break;
		case HWSPARKY2_MAINPORT_DSM:
#if defined(PIOS_INCLUDE_DSM)
			{
				// Force binding to zero on the main port
				hw_DSMxBind = 0;

				//TODO: Define the various Channelgroup for dsm inputs and handle here
				PIOS_Board_configure_dsm(&pios_usart_dsm_hsum_main_cfg, &pios_dsm_main_cfg,
							&pios_usart_com_driver, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT,&hw_DSMxBind);
			}
#endif	/* PIOS_INCLUDE_DSM */
			break;
		case HWSPARKY2_MAINPORT_HOTTSUMD:
		case HWSPARKY2_MAINPORT_HOTTSUMH:
#if defined(PIOS_INCLUDE_HSUM)
			{
				enum pios_hsum_proto proto;
				switch (hw_mainport) {
				case HWSPARKY2_MAINPORT_HOTTSUMD:
					proto = PIOS_HSUM_PROTO_SUMD;
					break;
				case HWSPARKY2_MAINPORT_HOTTSUMH:
					proto = PIOS_HSUM_PROTO_SUMH;
					break;
				default:
					PIOS_Assert(0);
					break;
				}
				PIOS_Board_configure_hsum(&pios_usart_dsm_hsum_main_cfg, &pios_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_HOTTSUM);
			}
#endif	/* PIOS_INCLUDE_HSUM */
			break;
		case HWSPARKY2_MAINPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
			{
				PIOS_Board_configure_com(&pios_usart_main_cfg, 0, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_debug_id);
			}
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWSPARKY2_MAINPORT_COMBRIDGE:
			PIOS_Board_configure_com(&pios_usart_main_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
			break;
		case HWSPARKY2_MAINPORT_MAVLINKTX:
#if defined(PIOS_INCLUDE_MAVLINK)
		PIOS_Board_configure_com(&pios_usart_main_cfg, 0, PIOS_COM_MAVLINK_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_mavlink_id);
#endif		/* PIOS_INCLUDE_MAVLINK */
		break;
		case HWSPARKY2_MAINPORT_MAVLINKTX_GPS_RX:
#if defined(PIOS_INCLUDE_MAVLINK)
		PIOS_Board_configure_com(&pios_usart_main_cfg, PIOS_COM_GPS_RX_BUF_LEN, PIOS_COM_MAVLINK_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_gps_id);
		pios_com_mavlink_id = pios_com_gps_id;
 #endif	/* PIOS_INCLUDE_MAVLINK */   	
		break;
		case HWSPARKY2_MAINPORT_HOTTTELEMETRY:
#if defined(PIOS_INCLUDE_HOTT) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usart_main_cfg, PIOS_COM_HOTT_RX_BUF_LEN, PIOS_COM_HOTT_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_hott_id);
#endif /* PIOS_INCLUDE_HOTT */
			break;
        case HWSPARKY2_MAINPORT_FRSKYSENSORHUB:
#if defined(PIOS_INCLUDE_FRSKY_SENSOR_HUB) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
            PIOS_Board_configure_com(&pios_usart_main_cfg, 0, PIOS_COM_FRSKYSENSORHUB_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_frsky_sensor_hub_id);
#endif /* PIOS_INCLUDE_FRSKY_SENSOR_HUB */
		break;
		case HWSPARKY2_MAINPORT_LIGHTTELEMETRYTX:
#if defined(PIOS_INCLUDE_LIGHTTELEMETRY)
		PIOS_Board_configure_com(&pios_usart_main_cfg, 0, PIOS_COM_LIGHTTELEMETRY_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_lighttelemetry_id);
#endif
		break;
		case HWSPARKY2_MAINPORT_PICOC:
#if defined(PIOS_INCLUDE_PICOC) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usart_main_cfg, PIOS_COM_PICOC_RX_BUF_LEN, PIOS_COM_PICOC_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_picoc_id);
#endif /* PIOS_INCLUDE_PICOC */
		break;
	} /* 	hw_mainport */

	/* Configure FlexiPort */
	uint8_t hw_flexiport;
	HwSparky2FlexiPortGet(&hw_flexiport);
	switch (hw_flexiport) {
		case HWSPARKY2_FLEXIPORT_DISABLED:
			break;
                case HWSPARKY2_FLEXIPORT_TELEMETRY:
                        PIOS_Board_configure_com(&pios_usart_flexi_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
			break;
		case HWSPARKY2_FLEXIPORT_I2C:
#if defined(PIOS_INCLUDE_I2C)
			{
				if (PIOS_I2C_Init(&pios_i2c_flexiport_adapter_id, &pios_i2c_flexiport_adapter_cfg)) {
					PIOS_Assert(0);
				}
				if (PIOS_I2C_CheckClear(pios_i2c_flexiport_adapter_id) != 0)
					panic(6);
			}
#endif	/* PIOS_INCLUDE_I2C */
			break;
		case HWSPARKY2_FLEXIPORT_GPS:
			PIOS_Board_configure_com(&pios_usart_flexi_cfg, PIOS_COM_GPS_RX_BUF_LEN, PIOS_COM_GPS_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_gps_id);
			break;
		case HWSPARKY2_FLEXIPORT_DSM:
#if defined(PIOS_INCLUDE_DSM)
			//TODO: Define the various Channelgroup for dsm inputs and handle here
			PIOS_Board_configure_dsm(&pios_usart_dsm_hsum_flexi_cfg, &pios_dsm_flexi_cfg,
						&pios_usart_com_driver, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMFLEXIPORT,&hw_DSMxBind);
			break;
#endif	/* PIOS_INCLUDE_DSM */
			break;
		case HWSPARKY2_FLEXIPORT_HOTTSUMD:
		case HWSPARKY2_FLEXIPORT_HOTTSUMH:
#if defined(PIOS_INCLUDE_HSUM)
			{
				enum pios_hsum_proto proto;
				switch (hw_flexiport) {
				case HWSPARKY2_FLEXIPORT_HOTTSUMD:
					proto = PIOS_HSUM_PROTO_SUMD;
					break;
				case HWSPARKY2_FLEXIPORT_HOTTSUMH:
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
		case HWSPARKY2_FLEXIPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
			{
				PIOS_Board_configure_com(&pios_usart_main_cfg, 0, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_debug_id);
			}
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWSPARKY2_FLEXIPORT_COMBRIDGE:
			PIOS_Board_configure_com(&pios_usart_flexi_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
			break;
		case HWSPARKY2_FLEXIPORT_MAVLINKTX:
#if defined(PIOS_INCLUDE_MAVLINK)
		PIOS_Board_configure_com(&pios_usart_flexi_cfg, 0, PIOS_COM_MAVLINK_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_mavlink_id);
#endif		/* PIOS_INCLUDE_MAVLINK */
		break;
		case HWSPARKY2_FLEXIPORT_MAVLINKTX_GPS_RX:
#if defined(PIOS_INCLUDE_MAVLINK)
		PIOS_Board_configure_com(&pios_usart_flexi_cfg, PIOS_COM_GPS_RX_BUF_LEN, PIOS_COM_MAVLINK_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_gps_id);
		pios_com_mavlink_id = pios_com_gps_id;
#endif    	/* PIOS_INCLUDE_MAVLINK */
		break;
		case HWSPARKY2_FLEXIPORT_HOTTTELEMETRY:
#if defined(PIOS_INCLUDE_HOTT) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usart_flexi_cfg, PIOS_COM_HOTT_RX_BUF_LEN, PIOS_COM_HOTT_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_hott_id);
#endif /* PIOS_INCLUDE_HOTT */
			break;
        case HWSPARKY2_FLEXIPORT_FRSKYSENSORHUB:
#if defined(PIOS_INCLUDE_FRSKY_SENSOR_HUB) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
            PIOS_Board_configure_com(&pios_usart_flexi_cfg, 0, PIOS_COM_FRSKYSENSORHUB_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_frsky_sensor_hub_id);
#endif /* PIOS_INCLUDE_FRSKY_SENSOR_HUB */
		break;
		case HWSPARKY2_FLEXIPORT_LIGHTTELEMETRYTX:
#if defined(PIOS_INCLUDE_LIGHTTELEMETRY)
		PIOS_Board_configure_com(&pios_usart_flexi_cfg, 0, PIOS_COM_LIGHTTELEMETRY_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_lighttelemetry_id);
#endif  
		case HWSPARKY2_FLEXIPORT_PICOC:
#if defined(PIOS_INCLUDE_PICOC) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usart_flexi_cfg, PIOS_COM_PICOC_RX_BUF_LEN, PIOS_COM_PICOC_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_picoc_id);
#endif /* PIOS_INCLUDE_PICOC */
		break;
	} /* hwsettings_rv_flexiport */


#if defined(PIOS_INCLUDE_RFM22B)
    /* Initalize the RFM22B radio COM device. */
	RFM22BStatusInitialize();
	RFM22BStatusCreateInstance();

	RFM22BStatusData rfm22bstatus;
	RFM22BStatusGet(&rfm22bstatus);
	RFM22BStatusInstSet(1,&rfm22bstatus);


	HwSparky2Data hwSparky2;
	HwSparky2Get(&hwSparky2);

	// Initialize out status object.
	rfm22bstatus.BoardType     = bdinfo->board_type;
	rfm22bstatus.BoardRevision = bdinfo->board_rev;

	if (hwSparky2.Radio == HWSPARKY2_RADIO_OPENLRS) {
		uintptr_t openlrs_id;

		const struct pios_openlrs_cfg *openlrs_cfg = PIOS_BOARD_HW_DEFS_GetOpenLRSCfg(bdinfo->board_rev);
		PIOS_OpenLRS_Init(&openlrs_id, PIOS_RFM22_SPI_PORT, 0, openlrs_cfg);

#if defined(PIOS_INCLUDE_OPENLRS_RCVR)
		{
			uintptr_t pios_rfm22brcvr_id;
			PIOS_OpenLRS_Rcvr_Init(&pios_rfm22brcvr_id, openlrs_id);
			uintptr_t pios_rfm22brcvr_rcvr_id;
			if (PIOS_RCVR_Init(&pios_rfm22brcvr_rcvr_id, &pios_openlrs_rcvr_driver, pios_rfm22brcvr_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_OPENLRS] = pios_rfm22brcvr_rcvr_id;
		}
#endif /* PIOS_INCLUDE_OPENLRS_RCVR */
	} else if (hwSparky2.Radio == HWSPARKY2_RADIO_DISABLED || hwSparky2.MaxRfPower == HWSPARKY2_MAXRFPOWER_0) {

			// When radio disabled, it is ok for init to fail. This allows boards without populating
			// this component.
			const struct pios_rfm22b_cfg *rfm22b_cfg = PIOS_BOARD_HW_DEFS_GetRfm22Cfg(bdinfo->board_rev);
			if (PIOS_RFM22B_Init(&pios_rfm22b_id, PIOS_RFM22_SPI_PORT, rfm22b_cfg->slave_num, rfm22b_cfg) == 0) {
				PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_0);
				rfm22bstatus.DeviceID = PIOS_RFM22B_DeviceID(pios_rfm22b_id);
				rfm22bstatus.BoardRevision = PIOS_RFM22B_ModuleVersion(pios_rfm22b_id);
			} else {
				pios_rfm22b_id = 0;
			}
			rfm22bstatus.LinkState = RFM22BSTATUS_LINKSTATE_DISABLED;

	} else {

		// always allow receiving PPM when radio is on
		bool ppm_mode    = hwSparky2.Radio == HWSPARKY2_RADIO_TELEMPPM || hwSparky2.Radio == HWSPARKY2_RADIO_PPM;
		bool ppm_only    = hwSparky2.Radio == HWSPARKY2_RADIO_PPM;
		bool is_oneway   = hwSparky2.Radio == HWSPARKY2_RADIO_PPM; // Sparky2 can only receive PPM only

		/* Configure the RFM22B device. */
		const struct pios_rfm22b_cfg *rfm22b_cfg = PIOS_BOARD_HW_DEFS_GetRfm22Cfg(bdinfo->board_rev);
		if (PIOS_RFM22B_Init(&pios_rfm22b_id, PIOS_RFM22_SPI_PORT, rfm22b_cfg->slave_num, rfm22b_cfg)) {
			PIOS_Assert(0);
		}

		rfm22bstatus.DeviceID = PIOS_RFM22B_DeviceID(pios_rfm22b_id);
		rfm22bstatus.BoardRevision = PIOS_RFM22B_ModuleVersion(pios_rfm22b_id);

		/* Configure the radio com interface */
		uint8_t *rx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_RFM22B_RF_RX_BUF_LEN);
		uint8_t *tx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_RFM22B_RF_TX_BUF_LEN);
		PIOS_Assert(rx_buffer);
		PIOS_Assert(tx_buffer);
		if (PIOS_COM_Init(&pios_com_rf_id, &pios_rfm22b_com_driver, pios_rfm22b_id,
		                  rx_buffer, PIOS_COM_RFM22B_RF_RX_BUF_LEN,
		                  tx_buffer, PIOS_COM_RFM22B_RF_TX_BUF_LEN)) {
			PIOS_Assert(0);
		}

		/* Set Telemetry to use RFM22b if no other telemetry is configured (USB always overrides anyway) */
		if (!pios_com_telem_rf_id) {
			pios_com_telem_rf_id = pios_com_rf_id;
		}
		rfm22bstatus.LinkState = RFM22BSTATUS_LINKSTATE_ENABLED;

		enum rfm22b_datarate datarate = RFM22_datarate_64000;
		switch (hwSparky2.MaxRfSpeed) {
		case HWSPARKY2_MAXRFSPEED_9600:
			datarate = RFM22_datarate_9600;
			break;
		case HWSPARKY2_MAXRFSPEED_19200:
			datarate = RFM22_datarate_19200;
			break;
		case HWSPARKY2_MAXRFSPEED_32000:
			datarate = RFM22_datarate_32000;
			break;
		case HWSPARKY2_MAXRFSPEED_64000:
			datarate = RFM22_datarate_64000;
			break;
		case HWSPARKY2_MAXRFSPEED_100000:
			datarate = RFM22_datarate_100000;
			break;
		case HWSPARKY2_MAXRFSPEED_192000:
			datarate = RFM22_datarate_192000;
			break;
		}

		/* Set the radio configuration parameters. */
		PIOS_RFM22B_Config(pios_rfm22b_id, datarate, hwSparky2.MinChannel, hwSparky2.MaxChannel, hwSparky2.CoordID, is_oneway, ppm_mode, ppm_only);

		/* Set the modem Tx poer level */
		switch (hwSparky2.MaxRfPower) {
		case HWSPARKY2_MAXRFPOWER_125:
			PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_0);
			break;
		case HWSPARKY2_MAXRFPOWER_16:
			PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_1);
			break;
		case HWSPARKY2_MAXRFPOWER_316:
			PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_2);
			break;
		case HWSPARKY2_MAXRFPOWER_63:
			PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_3);
			break;
		case HWSPARKY2_MAXRFPOWER_126:
			PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_4);
			break;
		case HWSPARKY2_MAXRFPOWER_25:
			PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_5);
			break;
		case HWSPARKY2_MAXRFPOWER_50:
			PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_6);
			break;
		case HWSPARKY2_MAXRFPOWER_100:
			PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_7);
			break;
		default:
			// do nothing
			break;
		}

		/* Reinitialize the modem. */
		PIOS_RFM22B_Reinit(pios_rfm22b_id);

#if defined(PIOS_INCLUDE_RFM22B_RCVR)
		{
			uintptr_t pios_rfm22brcvr_id;
			PIOS_RFM22B_Rcvr_Init(&pios_rfm22brcvr_id, pios_rfm22b_id);
			uintptr_t pios_rfm22brcvr_rcvr_id;
			if (PIOS_RCVR_Init(&pios_rfm22brcvr_rcvr_id, &pios_rfm22b_rcvr_driver, pios_rfm22brcvr_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_RFM22B] = pios_rfm22brcvr_rcvr_id;
		}
	}

	RFM22BStatusInstSet(1,&rfm22bstatus);
#endif /* PIOS_INCLUDE_RFM22B_RCVR */

#endif /* PIOS_INCLUDE_RFM22B */

	/* Configure the receiver port*/
	uint8_t hw_rcvrport;
	HwSparky2RcvrPortGet(&hw_rcvrport);

	switch (hw_rcvrport){
		case HWSPARKY2_RCVRPORT_DISABLED:
			break;
		case HWSPARKY2_RCVRPORT_PPM:
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
			break;
	case HWSPARKY2_RCVRPORT_DSM:
#if defined(PIOS_INCLUDE_DSM)
		switch (bdinfo->board_rev) {
		case BRUSHEDSPARKY_V0_2:
			break;
		default:
			hw_DSMxBind = 0; // Do not attempt to bind pulse through XOR
			break;
		}
		PIOS_Board_configure_dsm(&pios_usart_dsm_hsum_rcvr_cfg, &pios_dsm_rcvr_cfg, &pios_usart_com_driver,
		MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMRCVRPORT, &hw_DSMxBind);
		break;
#endif	/* PIOS_INCLUDE_DSM */
		break;
	case HWSPARKY2_RCVRPORT_HOTTSUMD:
	case HWSPARKY2_RCVRPORT_HOTTSUMH:
#if defined(PIOS_INCLUDE_HSUM)
		{
			enum pios_hsum_proto proto;
			switch (hw_rcvrport) {
			case HWSPARKY2_RCVRPORT_HOTTSUMD:
				proto = PIOS_HSUM_PROTO_SUMD;
				break;
			case HWSPARKY2_RCVRPORT_HOTTSUMH:
				proto = PIOS_HSUM_PROTO_SUMH;
				break;
			default:
				PIOS_Assert(0);
				break;
			}
			PIOS_Board_configure_hsum(&pios_usart_dsm_hsum_rcvr_cfg, &pios_usart_com_driver,
				&proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_HOTTSUM);
		}
#endif	/* PIOS_INCLUDE_HSUM */
		break;
	case HWSPARKY2_RCVRPORT_SBUS:
#if defined(PIOS_INCLUDE_SBUS) && defined(PIOS_INCLUDE_USART)
		{
			uintptr_t pios_usart_sbus_id;
			if (PIOS_USART_Init(&pios_usart_sbus_id, get_sbus_rcvr_cfg(bdinfo->board_rev))) {
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
#endif	/* PIOS_INCLUDE_SBUS */
		break;
	default:
		break;
	}

	if (hw_rcvrport != HWSPARKY2_RCVRPORT_SBUS && get_sbus_toggle(bdinfo->board_rev)) {
		const struct pios_sbus_cfg * sbus_cfg = get_sbus_toggle(bdinfo->board_rev);
		GPIO_Init(sbus_cfg->inv.gpio, (GPIO_InitTypeDef*)&sbus_cfg->inv.init);
		GPIO_WriteBit(sbus_cfg->inv.gpio, sbus_cfg->inv.init.GPIO_Pin, sbus_cfg->gpio_inv_disable);
	}



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
	/* Set up the servo outputs */
	PIOS_Servo_Init(&pios_servo_cfg);
#else
	PIOS_DEBUG_Init(&pios_tim_servo_all_channels, NELEMENTS(pios_tim_servo_all_channels));
#endif
	
	if (PIOS_I2C_Init(&pios_i2c_mag_pressure_adapter_id, &pios_i2c_mag_pressure_adapter_cfg)) {
		PIOS_DEBUG_Assert(0);
	}

#if defined(PIOS_INCLUDE_CAN)
	if(get_use_can(bdinfo->board_rev)) {
		if (PIOS_CAN_Init(&pios_can_id, &pios_can_cfg) != 0)
			panic(6);

		uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_CAN_RX_BUF_LEN);
		uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_CAN_TX_BUF_LEN);
		PIOS_Assert(rx_buffer);
		PIOS_Assert(tx_buffer);
		if (PIOS_COM_Init(&pios_com_can_id, &pios_can_com_driver, pios_can_id,
		                  rx_buffer, PIOS_COM_CAN_RX_BUF_LEN,
		                  tx_buffer, PIOS_COM_CAN_TX_BUF_LEN))
			panic(6);

		pios_com_bridge_id = pios_com_can_id;
	}
#endif

	PIOS_DELAY_WaitmS(50);

	PIOS_SENSORS_Init();

#if defined(PIOS_INCLUDE_ADC)
	uint32_t internal_adc_id;
	PIOS_INTERNAL_ADC_Init(&internal_adc_id, &pios_adc_cfg);
	PIOS_ADC_Init(&pios_internal_adc_id, &pios_internal_adc_driver, internal_adc_id);
 
        // configure the pullup for PA8 (inhibit pullups from current/sonar shared pin)
        GPIO_Init(pios_current_sonar_pin.gpio, &pios_current_sonar_pin.init);
#endif

#if defined(PIOS_INCLUDE_MS5611)
	if (PIOS_MS5611_Init(&pios_ms5611_cfg, pios_i2c_mag_pressure_adapter_id) != 0)
		panic(4);
	if (PIOS_MS5611_Test() != 0)
		panic(4);
#endif

	uint8_t Magnetometer;
	HwSparky2MagnetometerGet(&Magnetometer);

	if (Magnetometer != HWSPARKY2_MAGNETOMETER_INTERNAL)
		pios_mpu9250_cfg.use_magnetometer = false;


#if defined(PIOS_INCLUDE_MPU9250_SPI)
	if (PIOS_MPU9250_SPI_Init(pios_spi_gyro_id, 0, &pios_mpu9250_cfg) != 0)
		panic(2);

	// To be safe map from UAVO enum to driver enum
	uint8_t hw_gyro_range;
	HwSparky2GyroRangeGet(&hw_gyro_range);
	switch(hw_gyro_range) {
		case HWSPARKY2_GYRORANGE_250:
			PIOS_MPU9250_SetGyroRange(PIOS_MPU60X0_SCALE_250_DEG);
			break;
		case HWSPARKY2_GYRORANGE_500:
			PIOS_MPU9250_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);
			break;
		case HWSPARKY2_GYRORANGE_1000:
			PIOS_MPU9250_SetGyroRange(PIOS_MPU60X0_SCALE_1000_DEG);
			break;
		case HWSPARKY2_GYRORANGE_2000:
			PIOS_MPU9250_SetGyroRange(PIOS_MPU60X0_SCALE_2000_DEG);
			break;
	}

	uint8_t hw_accel_range;
	HwSparky2AccelRangeGet(&hw_accel_range);
	switch(hw_accel_range) {
		case HWSPARKY2_ACCELRANGE_2G:
			PIOS_MPU9250_SetAccelRange(PIOS_MPU60X0_ACCEL_2G);
			break;
		case HWSPARKY2_ACCELRANGE_4G:
			PIOS_MPU9250_SetAccelRange(PIOS_MPU60X0_ACCEL_4G);
			break;
		case HWSPARKY2_ACCELRANGE_8G:
			PIOS_MPU9250_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
			break;
		case HWSPARKY2_ACCELRANGE_16G:
			PIOS_MPU9250_SetAccelRange(PIOS_MPU60X0_ACCEL_16G);
			break;
	}

	// the filter has to be set before rate else divisor calculation will fail
	uint8_t hw_mpu9250_dlpf;
	HwSparky2MPU9250GyroLPFGet(&hw_mpu9250_dlpf);
	enum pios_mpu9250_gyro_filter mpu9250_gyro_lpf = \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250GYROLPF_250) ? PIOS_MPU9250_GYRO_LOWPASS_250_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250GYROLPF_184) ? PIOS_MPU9250_GYRO_LOWPASS_184_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250GYROLPF_92) ? PIOS_MPU9250_GYRO_LOWPASS_92_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250GYROLPF_41) ? PIOS_MPU9250_GYRO_LOWPASS_41_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250GYROLPF_20) ? PIOS_MPU9250_GYRO_LOWPASS_20_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250GYROLPF_10) ? PIOS_MPU9250_GYRO_LOWPASS_10_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250GYROLPF_5) ? PIOS_MPU9250_GYRO_LOWPASS_5_HZ : \
	    pios_mpu9250_cfg.default_gyro_filter;
	PIOS_MPU9250_SetGyroLPF(mpu9250_gyro_lpf);

	HwSparky2MPU9250AccelLPFGet(&hw_mpu9250_dlpf);
	enum pios_mpu9250_accel_filter mpu9250_accel_lpf = \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250ACCELLPF_460) ? PIOS_MPU9250_ACCEL_LOWPASS_460_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250ACCELLPF_184) ? PIOS_MPU9250_ACCEL_LOWPASS_184_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250ACCELLPF_92) ? PIOS_MPU9250_ACCEL_LOWPASS_92_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250ACCELLPF_41) ? PIOS_MPU9250_ACCEL_LOWPASS_41_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250ACCELLPF_20) ? PIOS_MPU9250_ACCEL_LOWPASS_20_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250ACCELLPF_10) ? PIOS_MPU9250_ACCEL_LOWPASS_10_HZ : \
	    (hw_mpu9250_dlpf == HWSPARKY2_MPU9250ACCELLPF_5) ? PIOS_MPU9250_ACCEL_LOWPASS_5_HZ : \
	    pios_mpu9250_cfg.default_accel_filter;
	PIOS_MPU9250_SetGyroLPF(mpu9250_accel_lpf);

	uint8_t hw_mpu9250_samplerate;
	HwSparky2MPU9250RateGet(&hw_mpu9250_samplerate);
	uint16_t mpu9250_samplerate = \
	    (hw_mpu9250_samplerate == HWSPARKY2_MPU9250RATE_200) ? 200 : \
	    (hw_mpu9250_samplerate == HWSPARKY2_MPU9250RATE_250) ? 250 : \
	    (hw_mpu9250_samplerate == HWSPARKY2_MPU9250RATE_333) ? 333 : \
	    (hw_mpu9250_samplerate == HWSPARKY2_MPU9250RATE_500) ? 500 : \
	    (hw_mpu9250_samplerate == HWSPARKY2_MPU9250RATE_1000) ? 1000 : \
	    pios_mpu9250_cfg.default_samplerate;
	PIOS_MPU9250_SetSampleRate(mpu9250_samplerate);
#endif /* PIOS_INCLUDE_MPU9250_SPI */


	PIOS_WDG_Clear();

#if defined(PIOS_INCLUDE_HMC5883)
	{
		uint8_t Magnetometer;
		HwSparky2MagnetometerGet(&Magnetometer);

		if (Magnetometer == HWSPARKY2_MAGNETOMETER_EXTERNALI2CFLEXIPORT)
		{
			if (PIOS_HMC5883_Init(pios_i2c_flexiport_adapter_id, &pios_hmc5883_external_cfg) != 0)
				panic(6);
			if (PIOS_HMC5883_Test() != 0)
				panic(6);
		} else if (Magnetometer == HWSPARKY2_MAGNETOMETER_EXTERNALAUXI2C) {
			if (PIOS_HMC5883_Init(pios_i2c_mag_pressure_adapter_id, &pios_hmc5883_external_cfg) != 0)
				panic(6);
			if (PIOS_HMC5883_Test() != 0)
				panic(6);
		}

		if (Magnetometer != HWSPARKY2_MAGNETOMETER_INTERNAL) {
			// setup sensor orientation
			uint8_t ExtMagOrientation;
			HwSparky2ExtMagOrientationGet(&ExtMagOrientation);
			enum pios_hmc5883_orientation hmc5883_orientation = \
				(ExtMagOrientation == HWSPARKY2_EXTMAGORIENTATION_TOP0DEGCW)      ? PIOS_HMC5883_TOP_0DEG      : \
				(ExtMagOrientation == HWSPARKY2_EXTMAGORIENTATION_TOP90DEGCW)     ? PIOS_HMC5883_TOP_90DEG     : \
				(ExtMagOrientation == HWSPARKY2_EXTMAGORIENTATION_TOP180DEGCW)    ? PIOS_HMC5883_TOP_180DEG    : \
				(ExtMagOrientation == HWSPARKY2_EXTMAGORIENTATION_TOP270DEGCW)    ? PIOS_HMC5883_TOP_270DEG    : \
				(ExtMagOrientation == HWSPARKY2_EXTMAGORIENTATION_BOTTOM0DEGCW)   ? PIOS_HMC5883_BOTTOM_0DEG   : \
				(ExtMagOrientation == HWSPARKY2_EXTMAGORIENTATION_BOTTOM90DEGCW)  ? PIOS_HMC5883_BOTTOM_90DEG  : \
				(ExtMagOrientation == HWSPARKY2_EXTMAGORIENTATION_BOTTOM180DEGCW) ? PIOS_HMC5883_BOTTOM_180DEG : \
				(ExtMagOrientation == HWSPARKY2_EXTMAGORIENTATION_BOTTOM270DEGCW) ? PIOS_HMC5883_BOTTOM_270DEG : \
				pios_hmc5883_external_cfg.Default_Orientation;
			PIOS_HMC5883_SetOrientation(hmc5883_orientation);
		}
	}
#endif /* PIOS_INCLUDE_HMC5883 */

#if defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_FLASH_JEDEC)
	if (get_external_flash(bdinfo->board_rev)) {
		if ( PIOS_STREAMFS_Init(&streamfs_id, &streamfs_settings, FLASH_PARTITION_LABEL_LOG) != 0)
			panic(8);
			
		const uint32_t LOG_BUF_LEN = 256;
		uint8_t *log_rx_buffer = PIOS_malloc(LOG_BUF_LEN);
		uint8_t *log_tx_buffer = PIOS_malloc(LOG_BUF_LEN);
		if (PIOS_COM_Init(&pios_com_logging_id, &pios_streamfs_com_driver, streamfs_id,
			log_rx_buffer, LOG_BUF_LEN, log_tx_buffer, LOG_BUF_LEN) != 0)
			panic(9);
	}
#endif	/* PIOS_INCLUDE_FLASH */

	switch (bdinfo->board_rev) {
	case BRUSHEDSPARKY_V0_2:
		{
			HwSparky2VTX_ChOptions channel;
			HwSparky2VTX_ChGet(&channel);
			set_vtx_channel(channel);
		}
		break;
	}

}

/**
 * @}
 * @}
 */

