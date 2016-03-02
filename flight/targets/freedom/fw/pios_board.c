/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup Freedom Tau Labs Freedom support files
 * @{
 *
 * @file       pios_board.c
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
#include <pios_hal.h>
#include <openpilot.h>
#include <uavobjectsinit.h>
#include "hwfreedom.h"
#include "modulesettings.h"
#include "manualcontrolsettings.h"
#include "rfm22bstatus.h"

#include "pios_internal_adc_priv.h"
#include "pios_adc_priv.h"
#include "pios_rfm22b_priv.h"
#include "pios_rfm22b_rcvr_priv.h"

/**
 * Sensor configurations 
 */

#if defined(PIOS_INCLUDE_HMC5883)
#include "pios_hmc5883_priv.h"
static const struct pios_exti_cfg pios_exti_hmc5883_cfg __exti_config = {
	.vector = PIOS_HMC5883_IRQHandler,
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

static const struct pios_hmc5883_cfg pios_hmc5883_cfg = {
	.exti_cfg = &pios_exti_hmc5883_cfg,
	.M_ODR = PIOS_HMC5883_ODR_75,
	.Meas_Conf = PIOS_HMC5883_MEASCONF_NORMAL,
	.Gain = PIOS_HMC5883_GAIN_1_9,
	.Mode = PIOS_HMC5883_MODE_CONTINUOUS,
	.Default_Orientation = PIOS_HMC5883_TOP_90DEG,

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
 * Configuration for the MPU6000 chip
 */
#if defined(PIOS_INCLUDE_MPU6000)
#include "pios_mpu6000.h"
static const struct pios_exti_cfg pios_exti_mpu6000_cfg __exti_config = {
	.vector = PIOS_MPU6000_IRQHandler,
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

static const struct pios_mpu60x0_cfg pios_mpu6000_cfg = {
	.exti_cfg = &pios_exti_mpu6000_cfg,
	.default_samplerate = 666,
	.interrupt_cfg = PIOS_MPU60X0_INT_CLR_ANYRD,
	.interrupt_en = PIOS_MPU60X0_INTEN_DATA_RDY,
	.User_ctl = PIOS_MPU60X0_USERCTL_DIS_I2C,
	.Pwr_mgmt_clk = PIOS_MPU60X0_PWRMGMT_PLL_Z_CLK,
	.default_filter = PIOS_MPU60X0_LOWPASS_256_HZ,
	.orientation = PIOS_MPU60X0_TOP_180DEG
};
#endif /* PIOS_INCLUDE_MPU6000 */


uintptr_t pios_com_overo_id;
uintptr_t pios_internal_adc_id;
uintptr_t pios_uavo_settings_fs_id;
uintptr_t pios_waypoints_settings_fs_id;

/**
 * Indicate a target-specific error code when a component fails to initialize
 * 1 pulse - flash chip
 * 2 pulses - MPU6000
 * 3 pulses - HMC5883
 * 4 pulses - MS5611
 * 5 pulses - I2C bus locked
 */
void panic(int32_t code) {
	PIOS_HAL_Panic(PIOS_LED_ALARM, code);
}

/**
 * PIOS_Board_Init()
 * initializes all the core subsystems on this specific hardware
 * called from System/openpilot.c
 */

#include <pios_board_info.h>

void PIOS_Board_Init(void) {

	/* Delay system */
	PIOS_DELAY_Init();

	const struct pios_board_info * bdinfo = &pios_board_info_blob;

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
	if (PIOS_Flash_Jedec_Init(&pios_external_flash_id, pios_spi_telem_flash_id, 1, &flash_m25p_cfg) != 0)
		panic(1);
	if (PIOS_Flash_Internal_Init(&pios_internal_flash_id, &flash_internal_cfg) != 0)
		panic(1);

	/* Register the partition table */
	const struct pios_flash_partition * flash_partition_table;
	uint32_t num_partitions;
	flash_partition_table = PIOS_BOARD_HW_DEFS_GetPartitionTable(bdinfo->board_rev, &num_partitions);
	PIOS_FLASH_register_partition_table(flash_partition_table, num_partitions);

	/* Mount all filesystems */
	if (PIOS_FLASHFS_Logfs_Init(&pios_uavo_settings_fs_id, &flashfs_settings_cfg, FLASH_PARTITION_LABEL_SETTINGS) != 0)
		panic(1);
	if (PIOS_FLASHFS_Logfs_Init(&pios_waypoints_settings_fs_id, &flashfs_waypoints_cfg, FLASH_PARTITION_LABEL_WAYPOINTS) != 0)
		panic(1);

#if defined(ERASE_FLASH)
	PIOS_FLASHFS_Format(pios_uavo_settings_fs_id);
#endif

#endif	/* PIOS_INCLUDE_FLASH */

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();

	/* Initialize the alarms library */
	AlarmsInitialize();

	HwFreedomInitialize();
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

	// /* Set up pulse timers */
	PIOS_TIM_InitClock(&tim_1_cfg);
	PIOS_TIM_InitClock(&tim_2_cfg);
	PIOS_TIM_InitClock(&tim_3_cfg);

	/* IAP System Setup */
	PIOS_IAP_Init();
	uint16_t boot_count = PIOS_IAP_ReadBootCount();
	if (boot_count < 3) {
		PIOS_IAP_WriteBootCount(++boot_count);
		AlarmsClear(SYSTEMALARMS_ALARM_BOOTFAULT);
	} else {
		/* Too many failed boot attempts, force hw config to defaults */
		HwFreedomSetDefaults(HwFreedomHandle(), 0);
		ModuleSettingsSetDefaults(ModuleSettingsHandle(),0);
		AlarmsSet(SYSTEMALARMS_ALARM_BOOTFAULT, SYSTEMALARMS_ALARM_CRITICAL);
	}


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
	HwFreedomUSB_VCPPortGet(&hw_usb_vcpport);

	if (!usb_cdc_present) {
		/* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
		hw_usb_vcpport = HWFREEDOM_USB_VCPPORT_DISABLED;
	}

	PIOS_HAL_ConfigureCDC(hw_usb_vcpport, pios_usb_id, &pios_usb_cdc_cfg);

#endif	/* PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_USB_HID)
	/* Configure the usb HID port */
	uint8_t hw_usb_hidport;
	HwFreedomUSB_HIDPortGet(&hw_usb_hidport);

	if (!usb_hid_present) {
		/* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
		hw_usb_hidport = HWFREEDOM_USB_HIDPORT_DISABLED;
	}

	PIOS_HAL_ConfigureHID(hw_usb_hidport, pios_usb_id, &pios_usb_hid_cfg);

#endif	/* PIOS_INCLUDE_USB_HID */

	if (usb_hid_present || usb_cdc_present) {
		PIOS_USBHOOK_Activate();
	}
#endif	/* PIOS_INCLUDE_USB */

	/* Configure IO ports */
	HwFreedomDSMxModeOptions hw_DSMxMode;
	HwFreedomDSMxModeGet(&hw_DSMxMode);

	/* Configure FlexiPort */
	uint8_t hw_mainport;
	HwFreedomMainPortGet(&hw_mainport);

	PIOS_HAL_ConfigurePort(hw_mainport,          // port type protocol
			&pios_usart_main_cfg,                // usart_port_cfg
			&pios_usart_main_cfg,                // frsky usart_port_cfg
			&pios_usart_com_driver,              // com_driver 
			NULL,                                // i2c_id 
			NULL,                                // i2c_cfg 
			NULL,                                // i2c_cfg 
			NULL,                                // pwm_cfg
			PIOS_LED_ALARM,                      // led_id
			&pios_usart_dsm_hsum_main_cfg,       // usart_dsm_hsum_cfg 
			&pios_dsm_main_cfg,                  // dsm_cfg
			hw_DSMxMode,                         // dsm_mode 
			NULL,                                // sbus_rcvr_cfg 
			NULL,                                // sbus_cfg 
			false);                              // sbus_toggle

	/* Configure FlexiPort */
	uint8_t hw_flexiport;
	HwFreedomFlexiPortGet(&hw_flexiport);

	PIOS_HAL_ConfigurePort(hw_flexiport,         // port type protocol
			&pios_usart_flexi_cfg,               // usart_port_cfg
			&pios_usart_flexi_cfg,               // frsky usart_port_cfg
			&pios_usart_com_driver,              // com_driver
			&pios_i2c_flexiport_adapter_id,      // i2c_id
			&pios_i2c_flexiport_adapter_cfg,     // i2c_cfg 
			NULL,                                // i2c_cfg 
			NULL,                                // pwm_cfg
			PIOS_LED_ALARM,                      // led_id
			&pios_usart_dsm_hsum_flexi_cfg,      // usart_dsm_hsum_cfg
			&pios_dsm_flexi_cfg,                 // dsm_cfg
			hw_DSMxMode,                         // dsm_mode 
			NULL,                                // sbus_rcvr_cfg 
			NULL,                                // sbus_cfg 
			false);                              // sbus_toggle


    /* Initalize the RFM22B radio COM device. */
#if defined(PIOS_INCLUDE_RFM22B)
	HwFreedomData hwFreedom;
	HwFreedomGet(&hwFreedom);

	const struct pios_rfm22b_cfg *rfm22b_cfg = PIOS_BOARD_HW_DEFS_GetRfm22Cfg(bdinfo->board_rev);

	const struct pios_openlrs_cfg *openlrs_cfg = PIOS_BOARD_HW_DEFS_GetOpenLRSCfg(bdinfo->board_rev);

	PIOS_HAL_ConfigureRFM22B(hwFreedom.Radio,
			bdinfo->board_type, bdinfo->board_rev,
			hwFreedom.MaxRfPower, hwFreedom.MaxRfSpeed,
			hwFreedom.RfBand,
			openlrs_cfg, rfm22b_cfg,
			hwFreedom.MinChannel, hwFreedom.MaxChannel,
			hwFreedom.CoordID, 1);

#endif /* PIOS_INCLUDE_RFM22B */

	PIOS_WDG_Clear();
	
	/* Configure the receiver port*/
	uint8_t hw_rcvrport;
	HwFreedomRcvrPortGet(&hw_rcvrport);

	if (hw_DSMxMode >= HWFREEDOM_DSMXMODE_BIND3PULSES) {
		hw_DSMxMode = HWFREEDOM_DSMXMODE_AUTODETECT; /* Do not try to bind through XOR */
	}

	PIOS_HAL_ConfigurePort(hw_rcvrport,           // port type protocol
			NULL,                                 // usart_port_cfg
			NULL,                                 // frsky usart_port_cfg
			&pios_usart_com_driver,               // com_driver
			NULL,                                 // i2c_id
			NULL,                                 // i2c_cfg
			&pios_ppm_cfg,                        // ppm_cfg
			NULL,                                 // pwm_cfg
			PIOS_LED_ALARM,                       // led_id
			&pios_usart_dsm_hsum_rcvr_cfg,        // usart_dsm_hsum_cfg
			&pios_dsm_rcvr_cfg,                   // dsm_cfg
			hw_DSMxMode,                          // dsm_mode
			&pios_usart_sbus_rcvr_cfg,            // sbus_rcvr_cfg
			&pios_sbus_cfg,                       // sbus_cfg
			true);                                // sbus_toggle


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
			uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PACKET_SIZE);
			uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PACKET_SIZE);
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
	uint8_t hw_output_port;
	HwFreedomOutputGet(&hw_output_port);
	switch (hw_output_port) {
		case HWFREEDOM_OUTPUT_DISABLED:
			break;
		case HWFREEDOM_OUTPUT_PWM:
			/* Set up the servo outputs */
			PIOS_Servo_Init(&pios_servo_cfg);
			break;
		default:
			break;
	}
#else
	PIOS_DEBUG_Init(&pios_tim_servo_all_channels, NELEMENTS(pios_tim_servo_all_channels));
#endif

	PIOS_DELAY_WaitmS(200);

	PIOS_SENSORS_Init();

#if defined(PIOS_INCLUDE_MPU6000)
	if (PIOS_MPU6000_Init(pios_spi_gyro_id,0, &pios_mpu6000_cfg) != 0)
		panic(2);
	if (PIOS_MPU6000_Test() != 0)
		panic(2);
	
	// To be safe map from UAVO enum to driver enum
	uint8_t hw_gyro_range;
	HwFreedomGyroRangeGet(&hw_gyro_range);
	switch(hw_gyro_range) {
		case HWFREEDOM_GYRORANGE_250:
			PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_250_DEG);
			break;
		case HWFREEDOM_GYRORANGE_500:
			PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);
			break;
		case HWFREEDOM_GYRORANGE_1000:
			PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_1000_DEG);
			break;
		case HWFREEDOM_GYRORANGE_2000:
			PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_2000_DEG);
			break;
	}

	uint8_t hw_accel_range;
	HwFreedomAccelRangeGet(&hw_accel_range);
	switch(hw_accel_range) {
		case HWFREEDOM_ACCELRANGE_2G:
			PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_2G);
			break;
		case HWFREEDOM_ACCELRANGE_4G:
			PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_4G);
			break;
		case HWFREEDOM_ACCELRANGE_8G:
			PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
			break;
		case HWFREEDOM_ACCELRANGE_16G:
			PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_16G);
			break;
	}

	// the filter has to be set before rate else divisor calculation will fail
	uint8_t hw_mpu6000_dlpf;
	HwFreedomMPU6000DLPFGet(&hw_mpu6000_dlpf);
	enum pios_mpu60x0_filter mpu6000_dlpf = \
	    (hw_mpu6000_dlpf == HWFREEDOM_MPU6000DLPF_256) ? PIOS_MPU60X0_LOWPASS_256_HZ : \
	    (hw_mpu6000_dlpf == HWFREEDOM_MPU6000DLPF_188) ? PIOS_MPU60X0_LOWPASS_188_HZ : \
	    (hw_mpu6000_dlpf == HWFREEDOM_MPU6000DLPF_98) ? PIOS_MPU60X0_LOWPASS_98_HZ : \
	    (hw_mpu6000_dlpf == HWFREEDOM_MPU6000DLPF_42) ? PIOS_MPU60X0_LOWPASS_42_HZ : \
	    (hw_mpu6000_dlpf == HWFREEDOM_MPU6000DLPF_20) ? PIOS_MPU60X0_LOWPASS_20_HZ : \
	    (hw_mpu6000_dlpf == HWFREEDOM_MPU6000DLPF_10) ? PIOS_MPU60X0_LOWPASS_10_HZ : \
	    (hw_mpu6000_dlpf == HWFREEDOM_MPU6000DLPF_5) ? PIOS_MPU60X0_LOWPASS_5_HZ : \
	    pios_mpu6000_cfg.default_filter;
	PIOS_MPU6000_SetLPF(mpu6000_dlpf);

	uint8_t hw_mpu6000_samplerate;
	HwFreedomMPU6000RateGet(&hw_mpu6000_samplerate);
	uint16_t mpu6000_samplerate = \
	    (hw_mpu6000_samplerate == HWFREEDOM_MPU6000RATE_200) ? 200 : \
	    (hw_mpu6000_samplerate == HWFREEDOM_MPU6000RATE_333) ? 333 : \
	    (hw_mpu6000_samplerate == HWFREEDOM_MPU6000RATE_500) ? 500 : \
	    (hw_mpu6000_samplerate == HWFREEDOM_MPU6000RATE_666) ? 666 : \
	    (hw_mpu6000_samplerate == HWFREEDOM_MPU6000RATE_1000) ? 1000 : \
	    (hw_mpu6000_samplerate == HWFREEDOM_MPU6000RATE_2000) ? 2000 : \
	    (hw_mpu6000_samplerate == HWFREEDOM_MPU6000RATE_4000) ? 4000 : \
	    (hw_mpu6000_samplerate == HWFREEDOM_MPU6000RATE_8000) ? 8000 : \
	    pios_mpu6000_cfg.default_samplerate;
	PIOS_MPU6000_SetSampleRate(mpu6000_samplerate);
#endif
	
	if (PIOS_I2C_Init(&pios_i2c_mag_pressure_adapter_id, &pios_i2c_mag_pressure_adapter_cfg)) {
		PIOS_DEBUG_Assert(0);
	}
	if (PIOS_I2C_CheckClear(pios_i2c_mag_pressure_adapter_id) != 0)
		panic(5);
	
	PIOS_DELAY_WaitmS(50);

#if defined(PIOS_INCLUDE_ADC)
	uint32_t internal_adc_id;
	PIOS_INTERNAL_ADC_Init(&internal_adc_id, &pios_adc_cfg);
	PIOS_ADC_Init(&pios_internal_adc_id, &pios_internal_adc_driver, internal_adc_id);
#endif

	PIOS_LED_On(0);
#if defined(PIOS_INCLUDE_HMC5883)
	if (PIOS_HMC5883_Init(pios_i2c_mag_pressure_adapter_id, &pios_hmc5883_cfg) != 0)
		panic(3);
#endif
	PIOS_LED_On(1);

	
#if defined(PIOS_INCLUDE_MS5611)
	if (PIOS_MS5611_Init(&pios_ms5611_cfg, pios_i2c_mag_pressure_adapter_id) != 0)
		panic(4);
#endif
	PIOS_LED_On(2);
}

/**
 * @}
 * @}
 */

