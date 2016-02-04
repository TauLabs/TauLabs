/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup FlyingF3 FlyingF3 support files
 * @{
 *
 * @file       pios_board.c
 * @author     dRonin, http://dronin.org Copyright (C) 2015
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2016
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
#include "hwflyingf3.h"
#include "manualcontrolsettings.h"
#include "modulesettings.h"

/* This file defines the what and where regarding all hardware connected to the
 * FlyingF3 board. Please see hardware/Production/FlyingF3/pinout.txt for
 * an overview.
 */

/**
 * Configuration for L3GD20 chip
 */
#if defined(PIOS_INCLUDE_L3GD20)
#include "pios_l3gd20.h"
static const struct pios_exti_cfg pios_exti_l3gd20_cfg __exti_config = {
	.vector = PIOS_L3GD20_IRQHandler,
	.line = EXTI_Line1,
	.pin = {
		.gpio = GPIOE,
		.init = {
			.GPIO_Pin = GPIO_Pin_1,
			.GPIO_Speed = GPIO_Speed_50MHz,
			.GPIO_Mode = GPIO_Mode_IN,
			.GPIO_OType = GPIO_OType_OD,
			.GPIO_PuPd = GPIO_PuPd_NOPULL,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI1_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGH,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line1, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};

static const struct pios_l3gd20_cfg pios_l3gd20_cfg = {
	.exti_cfg = &pios_exti_l3gd20_cfg,
	.range = PIOS_L3GD20_SCALE_500_DEG,
	//.orientation = PIOS_L3GD20_TOP_0DEG, FIXME
};
#endif /* PIOS_INCLUDE_L3GD20 */


/**
 * Configuration for the LSM303 chip
 */
#if defined(PIOS_INCLUDE_LSM303)
#include "pios_lsm303.h"
static const struct pios_exti_cfg pios_exti_lsm303_cfg __exti_config = {
	.vector = PIOS_LSM303_IRQHandler,
	.line = EXTI_Line4,
	.pin = {
		.gpio = GPIOE,
		.init = {
			.GPIO_Pin = GPIO_Pin_4,
			.GPIO_Speed = GPIO_Speed_50MHz,
			.GPIO_Mode = GPIO_Mode_IN,
			.GPIO_OType = GPIO_OType_OD,
			.GPIO_PuPd = GPIO_PuPd_NOPULL,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI4_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
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

static const struct pios_lsm303_cfg pios_lsm303_cfg = {
	.exti_cfg = &pios_exti_lsm303_cfg,
	.devicetype = PIOS_LSM303DLHC_DEVICE,
	.orientation = PIOS_LSM303_TOP_180DEG,
};
#endif /* PIOS_INCLUDE_LSM303 */

/**
 * Configuration for the BMP085 chip
 */
#if defined(PIOS_INCLUDE_BMP085)
#include "pios_bmp085_priv.h"
static const struct pios_bmp085_cfg pios_bmp085_cfg = {
    .oversampling = BMP085_OSR_3,
    .temperature_interleaving = 1,
};
#endif /* PIOS_INCLUDE_BMP085 */

#define PIOS_COM_CAN_RX_BUF_LEN 256
#define PIOS_COM_CAN_TX_BUF_LEN 256

uintptr_t pios_com_openlog_logging_id;
uintptr_t pios_can_id;
uintptr_t pios_com_can_id;
uintptr_t pios_uavo_settings_fs_id;
uintptr_t pios_waypoints_settings_fs_id;
uintptr_t pios_internal_adc_id;

/**
 * Indicate a target-specific error code when a component fails to initialize
 * 1 pulse - L3GD20
 * 2 pulses - LSM303
 * 3 pulses - internal I2C bus locked
 * 4 pulses - external I2C bus locked
 * 6 pulses - CAN bus
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

#if defined(PIOS_INCLUDE_SPI)
	if (PIOS_SPI_Init(&pios_spi_internal_id, &pios_spi_internal_cfg)) {
		PIOS_DEBUG_Assert(0);
	}
#endif

#if defined(PIOS_INCLUDE_I2C)
	if (PIOS_I2C_Init(&pios_i2c_internal_id, &pios_i2c_internal_cfg)) {
		PIOS_DEBUG_Assert(0);
	}
	if (PIOS_I2C_CheckClear(pios_i2c_internal_id) != 0)
		panic(3);

	if (PIOS_I2C_Init(&pios_i2c_external_id, &pios_i2c_external_cfg)) {
		PIOS_DEBUG_Assert(0);
	}
	if (PIOS_I2C_CheckClear(pios_i2c_external_id) != 0)
		panic(4);
#endif

#if defined(PIOS_INCLUDE_CAN)
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
#endif

#if defined(PIOS_INCLUDE_FLASH)
	/* Inititialize all flash drivers */
	PIOS_Flash_Internal_Init(&pios_internal_flash_id, &flash_internal_cfg);

	/* Register the partition table */
	const struct pios_flash_partition * flash_partition_table;
	uint32_t num_partitions;
	flash_partition_table = PIOS_BOARD_HW_DEFS_GetPartitionTable(bdinfo->board_rev, &num_partitions);
	PIOS_FLASH_register_partition_table(flash_partition_table, num_partitions);

	/* Mount all filesystems */
	PIOS_FLASHFS_Logfs_Init(&pios_uavo_settings_fs_id, &flashfs_internal_settings_cfg, FLASH_PARTITION_LABEL_SETTINGS);
	PIOS_FLASHFS_Logfs_Init(&pios_waypoints_settings_fs_id, &flashfs_internal_waypoints_cfg, FLASH_PARTITION_LABEL_WAYPOINTS);

#if defined(ERASE_FLASH)
	PIOS_FLASHFS_Format(pios_uavo_settings_fs_id);
#endif

#endif

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();

	/* Initialize the alarms library. Reads RCC reset flags */
	AlarmsInitialize();
	PIOS_RESET_Clear(); // Clear the RCC reset flags after use.

	/* Initialize the hardware UAVOs */
	HwFlyingF3Initialize();
	ModuleSettingsInitialize();

#if defined(PIOS_INCLUDE_RTC)
	/* Initialize the real-time clock and its associated tick */
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

	/* Set up pulse timers */
	//inputs
	PIOS_TIM_InitClock(&tim_1_cfg);
	PIOS_TIM_InitClock(&tim_8_cfg);
	PIOS_TIM_InitClock(&tim_15_cfg);
	PIOS_TIM_InitClock(&tim_16_cfg);
	PIOS_TIM_InitClock(&tim_17_cfg);
	//outputs
	PIOS_TIM_InitClock(&tim_2_cfg);
	PIOS_TIM_InitClock(&tim_3_cfg);
	PIOS_TIM_InitClock(&tim_4_cfg);

	/* IAP System Setup */
	PIOS_IAP_Init();
	uint16_t boot_count = PIOS_IAP_ReadBootCount();
	if (boot_count < 3) {
		PIOS_IAP_WriteBootCount(++boot_count);
		AlarmsClear(SYSTEMALARMS_ALARM_BOOTFAULT);
	} else {
		/* Too many failed boot attempts, force hw config to defaults */
		HwFlyingF3SetDefaults(HwFlyingF3Handle(), 0);
		ModuleSettingsSetDefaults(ModuleSettingsHandle(),0);
		AlarmsSet(SYSTEMALARMS_ALARM_BOOTFAULT, SYSTEMALARMS_ALARM_CRITICAL);
	}

#if defined(PIOS_INCLUDE_USB)
	/* Initialize board specific USB data */
	PIOS_USB_BOARD_DATA_Init();

	/* Flags to determine if various USB interfaces are advertised */
	bool usb_hid_present = false;

#if defined(PIOS_INCLUDE_USB_CDC)
	bool usb_cdc_present = false;
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
	HwFlyingF3USB_VCPPortGet(&hw_usb_vcpport);

	if (!usb_cdc_present) {
		/* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
		hw_usb_vcpport = HWFLYINGF3_USB_VCPPORT_DISABLED;
	}

	PIOS_HAL_ConfigureCDC(hw_usb_vcpport, pios_usb_id, &pios_usb_cdc_cfg);
	
#endif	/* PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_USB_HID)
	/* Configure the usb HID port */
	uint8_t hw_usb_hidport;
	HwFlyingF3USB_HIDPortGet(&hw_usb_hidport);

	if (!usb_hid_present) {
		/* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
		hw_usb_hidport = HWFLYINGF3_USB_HIDPORT_DISABLED;
	}

	PIOS_HAL_ConfigureHID(hw_usb_hidport, pios_usb_id, &pios_usb_hid_cfg);
	
#endif	/* PIOS_INCLUDE_USB_HID */
#endif	/* PIOS_INCLUDE_USB */

	/* Configure the IO ports */
	HwFlyingF3DSMxModeOptions hw_DSMxMode;
	HwFlyingF3DSMxModeGet(&hw_DSMxMode);

	/* UART1 Port */
	uint8_t hw_uart1;
	HwFlyingF3Uart1Get(&hw_uart1);

	PIOS_HAL_ConfigurePort(hw_uart1,             // port type protocol
			&pios_usart1_cfg,                    // usart_port_cfg
			&pios_usart1_sport_cfg,              // frsky usart_port_cfg
			&pios_usart_com_driver,              // com_driver
			NULL,                                // i2c_id
			NULL,                                // i2c_cfg
			NULL,                                // ppm_cfg
			NULL,                                // pwm_cfg
			PIOS_LED_ALARM,                      // led_id
			&pios_usart1_dsm_hsum_cfg,           // usart_dsm_hsum_cfg
			&pios_usart1_dsm_aux_cfg,            // dsm_cfg
			hw_DSMxMode,                         // dsm_mode
			&pios_usart1_sbus_cfg,               // sbus_rcvr_cfg
			&pios_usart1_sbus_aux_cfg,           // sbus_cfg
			false);                              // sbus_toggle

	/* UART2 Port */
	uint8_t hw_uart2;
	HwFlyingF3Uart2Get(&hw_uart2);

	PIOS_HAL_ConfigurePort(hw_uart2,             // port type protocol
			&pios_usart2_cfg,                    // usart_port_cfg
			&pios_usart2_sport_cfg,              // frsky usart_port_cfg
			&pios_usart_com_driver,              // com_driver
			NULL,                                // i2c_id
			NULL,                                // i2c_cfg
			NULL,                                // ppm_cfg
			NULL,                                // pwm_cfg
			PIOS_LED_ALARM,                      // led_id
			&pios_usart2_dsm_hsum_cfg,           // usart_dsm_hsum_cfg
			&pios_usart2_dsm_aux_cfg,            // dsm_cfg
			hw_DSMxMode,                         // dsm_mode
			&pios_usart2_sbus_cfg,               // sbus_rcvr_cfg
			&pios_usart2_sbus_aux_cfg,           // sbus_cfg
			false);                              // sbus_toggle

	/* UART3 Port */
	uint8_t hw_uart3;
	HwFlyingF3Uart3Get(&hw_uart3);

	PIOS_HAL_ConfigurePort(hw_uart3,             // port type protocol
			&pios_usart3_cfg,                    // usart_port_cfg
			&pios_usart3_sport_cfg,              // frsky usart_port_cfg
			&pios_usart_com_driver,              // com_driver
			NULL,                                // i2c_id
			NULL,                                // i2c_cfg
			NULL,                                // ppm_cfg
			NULL,                                // pwm_cfg
			PIOS_LED_ALARM,                      // led_id
			&pios_usart3_dsm_hsum_cfg,           // usart_dsm_hsum_cfg
			&pios_usart3_dsm_aux_cfg,            // dsm_cfg
			hw_DSMxMode,                         // dsm_mode
			&pios_usart3_sbus_cfg,               // sbus_rcvr_cfg
			&pios_usart3_sbus_aux_cfg,           // sbus_cfg
			false);                              // sbus_toggle

	/* UART4 Port */
	uint8_t hw_uart4;
	HwFlyingF3Uart4Get(&hw_uart4);

	PIOS_HAL_ConfigurePort(hw_uart4,             // port type protocol
			&pios_usart4_cfg,                    // usart_port_cfg
			&pios_usart4_sport_cfg,              // frsky usart_port_cfg
			&pios_usart_com_driver,              // com_driver
			NULL,                                // i2c_id
			NULL,                                // i2c_cfg
			NULL,                                // ppm_cfg
			NULL,                                // pwm_cfg
			PIOS_LED_ALARM,                      // led_id
			&pios_usart4_dsm_hsum_cfg,           // usart_dsm_hsum_cfg
			&pios_usart4_dsm_aux_cfg,            // dsm_cfg
			hw_DSMxMode,                         // dsm_mode
			&pios_usart4_sbus_cfg,               // sbus_rcvr_cfg
			&pios_usart4_sbus_aux_cfg,           // sbus_cfg
			false);                              // sbus_toggle

	/* UART5 Port */
	uint8_t hw_uart5;
	HwFlyingF3Uart5Get(&hw_uart5);

	PIOS_HAL_ConfigurePort(hw_uart5,             // port type protocol
			&pios_usart5_cfg,                    // usart_port_cfg
			&pios_usart5_sport_cfg,              // frsky usart_port_cfg
			&pios_usart_com_driver,              // com_driver
			NULL,                                // i2c_id
			NULL,                                // i2c_cfg
			NULL,                                // ppm_cfg
			NULL,                                // pwm_cfg
			PIOS_LED_ALARM,                      // led_id
			&pios_usart5_dsm_hsum_cfg,           // usart_dsm_hsum_cfg
			&pios_usart5_dsm_aux_cfg,            // dsm_cfg
			hw_DSMxMode,                         // dsm_mode
			&pios_usart5_sbus_cfg,               // sbus_rcvr_cfg
			&pios_usart5_sbus_aux_cfg,           // sbus_cfg
			false);                              // sbus_toggle

	/* Configure the rcvr port */
	uint8_t hw_rcvrport;
	HwFlyingF3RcvrPortGet(&hw_rcvrport);

	switch (hw_rcvrport) {
	
	case HWFLYINGF3_RCVRPORT_DISABLED:
		break;
	
	case HWFLYINGF3_RCVRPORT_PWM:
		PIOS_HAL_ConfigurePort(HWSHARED_PORTTYPES_PWM,  // port type protocol
				NULL,                                   // usart_port_cfg
				NULL,                                   // frsky usart_port_cfg
				NULL,                                   // com_driver
				NULL,                                   // i2c_id
				NULL,                                   // i2c_cfg
				NULL,                                   // ppm_cfg
				&pios_pwm_cfg,                          // pwm_cfg
				PIOS_LED_ALARM,                         // led_id
				NULL,                                   // usart_dsm_hsum_cfg
				NULL,                                   // dsm_cfg
				0,                                      // dsm_mode
				NULL,                                   // sbus_rcvr_cfg
				NULL,                                   // sbus_cfg    
				false);                                 // sbus_toggle
		break;

	case HWFLYINGF3_RCVRPORT_PPM:
	case HWFLYINGF3_RCVRPORT_PPMOUTPUTS:
		PIOS_HAL_ConfigurePort(HWSHARED_PORTTYPES_PPM,  // port type protocol
				NULL,                                   // usart_port_cfg
				NULL,                                   // frsky usart_port_cfg
				NULL,                                   // com_driver
				NULL,                                   // i2c_id
				NULL,                                   // i2c_cfg
				&pios_ppm_cfg,                          // ppm_cfg
				NULL,                                   // pwm_cfg
				PIOS_LED_ALARM,                         // led_id
				NULL,                                   // usart_dsm_hsum_cfg
				NULL,                                   // dsm_cfg
				0,                                      // dsm_mode
				NULL,                                   // sbus_rcvr_cfg
				NULL,                                   // sbus_cfg    
				false);                                 // sbus_toggle
		break;

	case HWFLYINGF3_RCVRPORT_PPMPWM:
		PIOS_HAL_ConfigurePort(HWSHARED_PORTTYPES_PPM,  // port type protocol
				NULL,                                   // usart_port_cfg
				NULL,                                   // frsky usart_port_cfg
				NULL,                                   // com_driver
				NULL,                                   // i2c_id
				NULL,                                   // i2c_cfg
				&pios_ppm_cfg,                          // ppm_cfg
				NULL,                                   // pwm_cfg
				PIOS_LED_ALARM,                         // led_id
				NULL,                                   // usart_dsm_hsum_cfg
				NULL,                                   // dsm_cfg
				0,                                      // dsm_mode
				NULL,                                   // sbus_rcvr_cfg
				NULL,                                   // sbus_cfg    
				false);                                 // sbus_toggle

		PIOS_HAL_ConfigurePort(HWSHARED_PORTTYPES_PWM,  // port type protocol
				NULL,                                   // usart_port_cfg
				NULL,                                   // frsky usart_port_cfg
				NULL,                                   // com_driver
				NULL,                                   // i2c_id
				NULL,                                   // i2c_cfg
				NULL,                                   // ppm_cfg
				&pios_pwm_with_ppm_cfg,                 // pwm_cfg
				PIOS_LED_ALARM,                         // led_id
				NULL,                                   // usart_dsm_hsum_cfg
				NULL,                                   // dsm_cfg
				0,                                      // dsm_mode
				NULL,                                   // sbus_rcvr_cfg
				NULL,                                   // sbus_cfg    
				false);                                 // sbus_toggle
		break;
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
	switch (hw_rcvrport) {
		case HWFLYINGF3_RCVRPORT_DISABLED:
		case HWFLYINGF3_RCVRPORT_PWM:
		case HWFLYINGF3_RCVRPORT_PPM:
			/* Set up the servo outputs */
#ifdef PIOS_INCLUDE_SERVO
			PIOS_Servo_Init(&pios_servo_cfg);
#endif
			break;
		case HWFLYINGF3_RCVRPORT_PPMOUTPUTS:
		case HWFLYINGF3_RCVRPORT_OUTPUTS:
#ifdef PIOS_INCLUDE_SERVO
			PIOS_Servo_Init(&pios_servo_rcvr_cfg);
#endif
			break;
	}
#else
	PIOS_DEBUG_Init(&pios_tim_servo_all_channels, NELEMENTS(pios_tim_servo_all_channels));
#endif

	PIOS_WDG_Clear();
	PIOS_DELAY_WaitmS(200);
	PIOS_WDG_Clear();

	PIOS_SENSORS_Init();

#if defined(PIOS_INCLUDE_L3GD20) && defined(PIOS_INCLUDE_SPI)
	if (PIOS_L3GD20_Init(pios_spi_internal_id, 0, &pios_l3gd20_cfg) != 0)
		panic(1);
	if (PIOS_L3GD20_Test() != 0)
		panic(1);

	uint8_t hw_l3gd20_samplerate;
	HwFlyingF3L3GD20RateGet(&hw_l3gd20_samplerate);
	enum pios_l3gd20_rate l3gd20_samplerate = PIOS_L3GD20_RATE_380HZ_100HZ;
	switch(hw_l3gd20_samplerate) {
		case HWFLYINGF3_L3GD20RATE_380:
			l3gd20_samplerate = PIOS_L3GD20_RATE_380HZ_100HZ;
			break;
		case HWFLYINGF3_L3GD20RATE_760:
			l3gd20_samplerate = PIOS_L3GD20_RATE_760HZ_100HZ;
			break;
	}
	PIOS_Assert(PIOS_L3GD20_SetSampleRate(l3gd20_samplerate) == 0);

	// To be safe map from UAVO enum to driver enum
	/*
	 * FIXME: add support for this to l3gd20 driver
	uint8_t hw_gyro_range;
	HwFlyingF3GyroRangeGet(&hw_gyro_range);
	switch(hw_gyro_range) {
		case HWFLYINGF3_GYRORANGE_250:
			PIOS_L3GD20_SetRange(PIOS_L3GD20_SCALE_250_DEG);
			break;
		case HWFLYINGF3_GYRORANGE_500:
			PIOS_L3GD20_SetRange(PIOS_L3GD20_SCALE_500_DEG);
			break;
		case HWFLYINGF3_GYRORANGE_1000:
			//FIXME: how to behave in this case?
			PIOS_L3GD20_SetRange(PIOS_L3GD20_SCALE_2000_DEG);
			break;
		case HWFLYINGF3_GYRORANGE_2000:
			PIOS_L3GD20_SetRange(PIOS_L3GD20_SCALE_2000_DEG);
			break;
	}
	*/

	PIOS_WDG_Clear();
	PIOS_DELAY_WaitmS(50);
	PIOS_WDG_Clear();
#endif /* PIOS_INCLUDE_L3GD20 && PIOS_INCLUDE_I2C*/

#if defined(PIOS_INCLUDE_LSM303) && defined(PIOS_INCLUDE_I2C)
	if (PIOS_LSM303_Init(pios_i2c_internal_id, &pios_lsm303_cfg) != 0)
		panic(2);
	if (PIOS_LSM303_Accel_Test() != 0)
		panic(2);
	if (PIOS_LSM303_Mag_Test() != 0)
		panic(2);

	uint8_t hw_accel_range;
	HwFlyingF3AccelRangeGet(&hw_accel_range);
	switch(hw_accel_range) {
		case HWFLYINGF3_ACCELRANGE_2G:
			PIOS_LSM303_Accel_SetRange(PIOS_LSM303_ACCEL_2G);
			break;
		case HWFLYINGF3_ACCELRANGE_4G:
			PIOS_LSM303_Accel_SetRange(PIOS_LSM303_ACCEL_4G);
			break;
		case HWFLYINGF3_ACCELRANGE_8G:
			PIOS_LSM303_Accel_SetRange(PIOS_LSM303_ACCEL_8G);
			break;
		case HWFLYINGF3_ACCELRANGE_16G:
			PIOS_LSM303_Accel_SetRange(PIOS_LSM303_ACCEL_16G);
			break;
	}

	//there is no setting for the mag scale yet
	PIOS_LSM303_Mag_SetRange(PIOS_LSM303_MAG_1_9GA);

	PIOS_WDG_Clear();
	PIOS_DELAY_WaitmS(50);
	PIOS_WDG_Clear();
#endif /* PIOS_INCLUDE_LSM303 && PIOS_INCLUDE_I2C*/

#if defined(PIOS_INCLUDE_GPIO)
	PIOS_GPIO_Init();
#endif

	//FlyingF3 shield specific hw init
	uint8_t flyingf3_shield;
	uint32_t internal_adc_id;
	HwFlyingF3ShieldGet(&flyingf3_shield);
	switch (flyingf3_shield) {
	case HWFLYINGF3_SHIELD_RCFLYER:
#if defined(PIOS_INCLUDE_ADC)
		//Sanity check, this is to ensure that no one changes the adc_pins array without changing the defines
		PIOS_Assert(internal_adc_cfg_rcflyer_shield.adc_pins[PIOS_ADC_RCFLYER_SHIELD_BARO_PIN].pin == GPIO_Pin_3);
		PIOS_Assert(internal_adc_cfg_rcflyer_shield.adc_pins[PIOS_ADC_RCFLYER_SHIELD_BAT_VOLTAGE_PIN].pin == GPIO_Pin_4);
		if (PIOS_INTERNAL_ADC_Init(&internal_adc_id, &internal_adc_cfg_rcflyer_shield) < 0)
			PIOS_Assert(0);
		PIOS_ADC_Init(&pios_internal_adc_id, &pios_internal_adc_driver, internal_adc_id);
#endif
#if defined(PIOS_INCLUDE_SPI)
		if (PIOS_SPI_Init(&pios_spi_2_id, &pios_spi_2_rcflyer_internal_cfg)) {
			PIOS_DEBUG_Assert(0);
		}
		if (PIOS_SPI_Init(&pios_spi_3_id, &pios_spi_3_rcflyer_external_cfg)) {
			PIOS_DEBUG_Assert(0);
		}
#if defined(PIOS_INCLUDE_MS5611_SPI)
		if (PIOS_MS5611_SPI_Init(pios_spi_2_id, 1, &pios_ms5611_cfg) != 0) {
			PIOS_Assert(0);
		}
#endif	/* PIOS_INCLUDE_MS5611_SPI */
#endif	/* PIOS_INCLUDE_SPI */
		break;
	case HWFLYINGF3_SHIELD_CHEBUZZ:
#if defined(PIOS_INCLUDE_I2C) && defined(PIOS_INCLUDE_MS5611)
		if (PIOS_MS5611_Init(&pios_ms5611_cfg, pios_i2c_external_id) != 0) {
			PIOS_Assert(0);
		}
#endif	/* PIOS_INCLUDE_I2C && PIOS_INCLUDE_MS5611 */
#if defined(PIOS_INCLUDE_SPI)
		if (PIOS_SPI_Init(&pios_spi_2_id, &pios_spi_2_chebuzz_external_cfg)) {
			PIOS_DEBUG_Assert(0);
		}
		if (PIOS_SPI_Init(&pios_spi_3_id, &pios_spi_3_chebuzz_internal_cfg)) {
			PIOS_DEBUG_Assert(0);
		}
#endif	/* PIOS_INCLUDE_SPI */
		break;
	case HWFLYINGF3_SHIELD_BMP085:
#if defined(PIOS_INCLUDE_BMP085) && defined(PIOS_INCLUDE_I2C)
	if (PIOS_BMP085_Init(&pios_bmp085_cfg, pios_i2c_external_id) != 0)
		panic(5);
	if (PIOS_BMP085_Test() != 0)
		panic(5);
#endif /* PIOS_INCLUDE_BMP085 && PIOS_INCLUDE_I2C */
		break;
	case HWFLYINGF3_SHIELD_NONE:
		break;
	default:
		PIOS_Assert(0);
		break;
	}

	/* Make sure we have at least one telemetry link configured or else fail initialization */
	PIOS_Assert(pios_com_telem_serial_id || pios_com_telem_usb_id);
}

/**
 * @}
 */
