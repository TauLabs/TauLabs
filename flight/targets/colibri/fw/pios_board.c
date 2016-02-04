/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup Colibri Colibri support files
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
#include "hwcolibri.h"
#include "manualcontrolsettings.h"
#include "modulesettings.h"

/**
 * Sensor configurations
 */
#if defined(PIOS_INCLUDE_HMC5883)
#include "pios_hmc5883_priv.h"
static const struct pios_exti_cfg pios_exti_hmc5883_internal_cfg
    __exti_config = {
	.vector = PIOS_HMC5883_IRQHandler,
	.line = EXTI_Line1,
	.pin = {
		.gpio = GPIOC,
		.init = {
			 .GPIO_Pin = GPIO_Pin_1,
			 .GPIO_Speed = GPIO_Speed_100MHz,
			 .GPIO_Mode = GPIO_Mode_IN,
			 .GPIO_OType = GPIO_OType_OD,
			 .GPIO_PuPd = GPIO_PuPd_NOPULL,
			 },
		},
	.irq = {
		.init = {
			 .NVIC_IRQChannel = EXTI1_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_LOW,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.exti = {
		 .init = {
			  .EXTI_Line = EXTI_Line1,	// matches above GPIO pin
			  .EXTI_Mode = EXTI_Mode_Interrupt,
			  .EXTI_Trigger = EXTI_Trigger_Rising,
			  .EXTI_LineCmd = ENABLE,
			  },
		 },
};

static const struct pios_hmc5883_cfg pios_hmc5883_internal_cfg = {
	.exti_cfg = &pios_exti_hmc5883_internal_cfg,
	.M_ODR = PIOS_HMC5883_ODR_75,
	.Meas_Conf = PIOS_HMC5883_MEASCONF_NORMAL,
	.Gain = PIOS_HMC5883_GAIN_1_9,
	.Mode = PIOS_HMC5883_MODE_CONTINUOUS,
	.Default_Orientation = PIOS_HMC5883_BOTTOM_90DEG,
};

static const struct pios_hmc5883_cfg pios_hmc5883_external_cfg = {
	.M_ODR = PIOS_HMC5883_ODR_75,
	.Meas_Conf = PIOS_HMC5883_MEASCONF_NORMAL,
	.Gain = PIOS_HMC5883_GAIN_1_9,
	.Mode = PIOS_HMC5883_MODE_SINGLE,
	.Default_Orientation = PIOS_HMC5883_TOP_0DEG,
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
	.line = EXTI_Line0,
	.pin = {
		.gpio = GPIOC,
		.init = {
			 .GPIO_Pin = GPIO_Pin_0,
			 .GPIO_Speed = GPIO_Speed_100MHz,
			 .GPIO_Mode = GPIO_Mode_IN,
			 .GPIO_OType = GPIO_OType_OD,
			 .GPIO_PuPd = GPIO_PuPd_NOPULL,
			 },
		},
	.irq = {
		.init = {
			 .NVIC_IRQChannel = EXTI0_IRQn,
			 .NVIC_IRQChannelPreemptionPriority =
			 PIOS_IRQ_PRIO_HIGH,
			 .NVIC_IRQChannelSubPriority = 0,
			 .NVIC_IRQChannelCmd = ENABLE,
			 },
		},
	.exti = {
		 .init = {
			  .EXTI_Line = EXTI_Line0,	// matches above GPIO pin
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
	.orientation = PIOS_MPU60X0_BOTTOM_180DEG
};
#endif /* PIOS_INCLUDE_MPU6000 */

uintptr_t pios_com_spiflash_logging_id;
uintptr_t pios_com_openlog_logging_id;
uintptr_t pios_uavo_settings_fs_id;
uintptr_t pios_waypoints_settings_fs_id;
uintptr_t pios_internal_adc_id;
uintptr_t streamfs_id;

/**
 * Indicate a target-specific error code when a component fails to initialize
 * 1 pulse - flash chip
 * 2 pulses - MPU6000
 * 3 pulses - HMC5883
 * 4 pulses - MS5611
 * 5 pulses - internal I2C bus locked
 * 6 pulses - uart1 I2C bus locked
 * 7 pulses - uart3 I2C bus locked
 * 8 pulses - HMC5883 on uart1 I2C
 * 9 pulses - HMC5883 on uart3 I2C
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

	const struct pios_board_info *bdinfo = &pios_board_info_blob;

#if defined(PIOS_INCLUDE_LED)
	const struct pios_led_cfg *led_cfg =
	    PIOS_BOARD_HW_DEFS_GetLedCfg(bdinfo->board_rev);
	PIOS_Assert(led_cfg);
	PIOS_LED_Init(led_cfg);
#endif /* PIOS_INCLUDE_LED */

#if defined(PIOS_INCLUDE_I2C)
	if (PIOS_I2C_Init
	    (&pios_i2c_internal_adapter_id,
	     &pios_i2c_internal_adapter_cfg)) {
		PIOS_DEBUG_Assert(0);
	}
	if (PIOS_I2C_CheckClear(pios_i2c_internal_adapter_id) != 0)
		panic(5);
#endif

#if defined(PIOS_INCLUDE_SPI)
	if (PIOS_SPI_Init(&pios_spi_flash_id, &pios_spi_flash_cfg)) {
		PIOS_DEBUG_Assert(0);
	}
	if (PIOS_SPI_Init
	    (&pios_spi_gyro_accel_id, &pios_spi_gyro_accel_cfg)) {
		PIOS_Assert(0);
	}
#endif

#if defined(PIOS_INCLUDE_FLASH)
	/* Inititialize all flash drivers */
	if (PIOS_Flash_Jedec_Init
	    (&pios_external_flash_id, pios_spi_flash_id, 0,
	     &flash_mx25_cfg) != 0)
		panic(1);
	if (PIOS_Flash_Internal_Init
	    (&pios_internal_flash_id, &flash_internal_cfg) != 0)
		panic(1);

	/* Register the partition table */
	const struct pios_flash_partition *flash_partition_table;
	uint32_t num_partitions;
	flash_partition_table =
	    PIOS_BOARD_HW_DEFS_GetPartitionTable(bdinfo->board_rev,
						 &num_partitions);
	PIOS_FLASH_register_partition_table(flash_partition_table,
					    num_partitions);

	/* Mount all filesystems */
	if (PIOS_FLASHFS_Logfs_Init
	    (&pios_uavo_settings_fs_id, &flashfs_settings_cfg,
	     FLASH_PARTITION_LABEL_SETTINGS) != 0)
		panic(1);
	if (PIOS_FLASHFS_Logfs_Init
	    (&pios_waypoints_settings_fs_id, &flashfs_waypoints_cfg,
	     FLASH_PARTITION_LABEL_WAYPOINTS) != 0)
		panic(1);

#if defined(ERASE_FLASH)
	PIOS_FLASHFS_Format(pios_uavo_settings_fs_id);
#endif

#endif /* PIOS_INCLUDE_FLASH */

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();

	/* Initialize the alarms library. Reads RCC reset flags */
	AlarmsInitialize();
	PIOS_RESET_Clear(); // Clear the RCC reset flags after use.

	/* Initialize the hardware UAVOs */
	HwColibriInitialize();
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
	//Timers used for inputs (1, 2, 5, 8)
	PIOS_TIM_InitClock(&tim_1_cfg);
	PIOS_TIM_InitClock(&tim_2_cfg);
	PIOS_TIM_InitClock(&tim_5_cfg);
	PIOS_TIM_InitClock(&tim_8_cfg);
	// Timers used for outputs (3, 10, 11, 12)
	PIOS_TIM_InitClock(&tim_3_cfg);
	PIOS_TIM_InitClock(&tim_10_cfg);
	PIOS_TIM_InitClock(&tim_11_cfg);
	PIOS_TIM_InitClock(&tim_12_cfg);

	/* IAP System Setup */
	PIOS_IAP_Init();
	uint16_t boot_count = PIOS_IAP_ReadBootCount();
	if (boot_count < 3) {
		PIOS_IAP_WriteBootCount(++boot_count);
		AlarmsClear(SYSTEMALARMS_ALARM_BOOTFAULT);
	} else {
		/* Too many failed boot attempts, force hw config to defaults */
		HwColibriSetDefaults(HwColibriHandle(), 0);
		ModuleSettingsSetDefaults(ModuleSettingsHandle(), 0);
		AlarmsSet(SYSTEMALARMS_ALARM_BOOTFAULT,
			  SYSTEMALARMS_ALARM_CRITICAL);
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
	PIOS_USB_Init(&pios_usb_id,
		      PIOS_BOARD_HW_DEFS_GetUsbCfg(bdinfo->board_rev));

#if defined(PIOS_INCLUDE_USB_CDC)

	uint8_t hw_usb_vcpport;
	/* Configure the USB VCP port */
	HwColibriUSB_VCPPortGet(&hw_usb_vcpport);

	if (!usb_cdc_present) {
		/* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
		hw_usb_vcpport = HWCOLIBRI_USB_VCPPORT_DISABLED;
	}

	PIOS_HAL_ConfigureCDC(hw_usb_vcpport, pios_usb_id, &pios_usb_cdc_cfg);
	
#endif /* PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_USB_HID)
	/* Configure the usb HID port */
	uint8_t hw_usb_hidport;
	HwColibriUSB_HIDPortGet(&hw_usb_hidport);

	if (!usb_hid_present) {
		/* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
		hw_usb_hidport = HWCOLIBRI_USB_HIDPORT_DISABLED;
	}

	PIOS_HAL_ConfigureHID(hw_usb_hidport, pios_usb_id, &pios_usb_hid_cfg);
	
#endif /* PIOS_INCLUDE_USB_HID */

	if (usb_hid_present || usb_cdc_present) {
		PIOS_USBHOOK_Activate();
	}
#endif /* PIOS_INCLUDE_USB */

	/* Configure the IO ports */
	HwColibriDSMxModeOptions hw_DSMxMode;
	HwColibriDSMxModeGet(&hw_DSMxMode);

	/* UART1 Port */
	uint8_t hw_uart1;
	HwColibriUart1Get(&hw_uart1);

	PIOS_HAL_ConfigurePort(hw_uart1,             // port type protocol
			&pios_usart1_cfg,                    // usart_port_cfg
			&pios_usart1_cfg,                    // frsky usart_port_cfg
			&pios_usart_com_driver,              // com_driver
			&pios_i2c_usart1_adapter_id,         // i2c_id
			&pios_i2c_usart1_adapter_cfg,        // i2c_cfg
			NULL,                                // ppm_cfg
			NULL,                                // pwm_cfg
			PIOS_LED_ALARM,                      // led_id
			&pios_usart1_dsm_hsum_cfg,           // usart_dsm_hsum_cfg
			&pios_usart1_dsm_aux_cfg,            // dsm_cfg
			hw_DSMxMode,                         // dsm_mode
			NULL,                                // sbus_rcvr_cfg
			NULL,                                // sbus_cfg
			false);                              // sbus_toggle

	/* UART2 Port */
	uint8_t hw_uart2;
	HwColibriUart2Get(&hw_uart2);

	PIOS_HAL_ConfigurePort(hw_uart2,             // port type protocol
			&pios_usart2_cfg,                    // usart_port_cfg
			&pios_usart2_cfg,                    // frsky usart_port_cfg
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
			true);                               // sbus_toggle

	if (hw_uart2 != HWCOLIBRI_UART2_SBUS) {
		GPIO_Init(pios_usart2_sbus_aux_cfg.inv.gpio, (GPIO_InitTypeDef*)&pios_usart2_sbus_aux_cfg.inv.init);
		GPIO_WriteBit(pios_usart2_sbus_aux_cfg.inv.gpio, pios_usart2_sbus_aux_cfg.inv.init.GPIO_Pin, pios_usart2_sbus_aux_cfg.gpio_inv_disable);
	}

	/* UART3 Port */
	uint8_t hw_uart3;
	HwColibriUart3Get(&hw_uart3);

	PIOS_HAL_ConfigurePort(hw_uart3,             // port type protocol
			&pios_usart3_cfg,                    // usart_port_cfg
			&pios_usart3_cfg,                    // frsky usart_port_cfg
			&pios_usart_com_driver,              // com_driver
			&pios_i2c_usart3_adapter_id,         // i2c_id
			&pios_i2c_usart3_adapter_cfg,        // i2c_cfg
			NULL,                                // ppm_cfg
			NULL,                                // pwm_cfg
			PIOS_LED_ALARM,                      // led_id
			&pios_usart3_dsm_hsum_cfg,           // usart_dsm_hsum_cfg
			&pios_usart3_dsm_aux_cfg,            // dsm_cfg
			hw_DSMxMode,                         // dsm_mode
			NULL,                                // sbus_rcvr_cfg
			NULL,                                // sbus_cfg
			false);                              // sbus_toggle

	/* UART4 Port */
	uint8_t hw_uart4;
	HwColibriUart4Get(&hw_uart4);

	PIOS_HAL_ConfigurePort(hw_uart4,             // port type protocol
			&pios_usart4_cfg,                    // usart_port_cfg
			&pios_usart4_cfg,                    // frsky usart_port_cfg
			&pios_usart_com_driver,              // com_driver
			NULL,                                // i2c_id
			NULL,                                // i2c_cfg
			NULL,                                // ppm_cfg
			NULL,                                // pwm_cfg
			PIOS_LED_ALARM,                      // led_id
			&pios_usart4_dsm_hsum_cfg,           // usart_dsm_hsum_cfg
			&pios_usart4_dsm_aux_cfg,            // dsm_cfg
			hw_DSMxMode,                         // dsm_mode
			NULL,                                // sbus_rcvr_cfg
			NULL,                                // sbus_cfg
			false);                              // sbus_toggle

	/* Configure the rcvr port */
	uint8_t hw_rcvrport;
	HwColibriRcvrPortGet(&hw_rcvrport);

	switch (hw_rcvrport) {
	case HWCOLIBRI_RCVRPORT_DISABLED:
		break;
	
	case HWCOLIBRI_RCVRPORT_PWM:
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

	case HWCOLIBRI_RCVRPORT_PWMADC:
		PIOS_HAL_ConfigurePort(HWSHARED_PORTTYPES_PWM,  // port type protocol
				NULL,                                   // usart_port_cfg
				NULL,                                   // frsky usart_port_cfg
				NULL,                                   // com_driver
				NULL,                                   // i2c_id
				NULL,                                   // i2c_cfg
				NULL,                                   // ppm_cfg
				&pios_pwm_with_adc_cfg,                 // pwm_cfg
				PIOS_LED_ALARM,                         // led_id
				NULL,                                   // usart_dsm_hsum_cfg
				NULL,                                   // dsm_cfg
				0,                                      // dsm_mode
				NULL,                                   // sbus_rcvr_cfg
				NULL,                                   // sbus_cfg    
				false);                                 // sbus_toggle
		break;

	case HWCOLIBRI_RCVRPORT_PPM:
	case HWCOLIBRI_RCVRPORT_PPMADC:
	case HWCOLIBRI_RCVRPORT_PPMOUTPUTS:
	case HWCOLIBRI_RCVRPORT_PPMOUTPUTSADC:
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

	case HWCOLIBRI_RCVRPORT_PPMPWM:
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

	case HWCOLIBRI_RCVRPORT_PPMPWMADC:
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
				&pios_pwm_with_ppm_with_adc_cfg,        // pwm_cfg
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
	if (PIOS_RCVR_Init
	    (&pios_gcsrcvr_rcvr_id, &pios_gcsrcvr_rcvr_driver,
	     pios_gcsrcvr_id)) {
		PIOS_Assert(0);
	}
	pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_GCS] =
	    pios_gcsrcvr_rcvr_id;
#endif /* PIOS_INCLUDE_GCSRCVR */

#ifndef PIOS_DEBUG_ENABLE_DEBUG_PINS
	switch (hw_rcvrport) {
	case HWCOLIBRI_RCVRPORT_DISABLED:
	case HWCOLIBRI_RCVRPORT_PWM:
	case HWCOLIBRI_RCVRPORT_PWMADC:
	case HWCOLIBRI_RCVRPORT_PPM:
	case HWCOLIBRI_RCVRPORT_PPMADC:
	case HWCOLIBRI_RCVRPORT_PPMPWM:
	case HWCOLIBRI_RCVRPORT_PPMPWMADC:
		/* Set up the servo outputs */
#ifdef PIOS_INCLUDE_SERVO
		PIOS_Servo_Init(&pios_servo_cfg);
#endif
		break;
	case HWCOLIBRI_RCVRPORT_PPMOUTPUTS:
	case HWCOLIBRI_RCVRPORT_OUTPUTS:
#ifdef PIOS_INCLUDE_SERVO
		PIOS_Servo_Init(&pios_servo_with_rcvr_cfg);
#endif
		break;
	case HWCOLIBRI_RCVRPORT_PPMOUTPUTSADC:
	case HWCOLIBRI_RCVRPORT_OUTPUTSADC:
#ifdef PIOS_INCLUDE_SERVO
		PIOS_Servo_Init(&pios_servo_with_rcvr_with_adc_cfg);
#endif
		break;
	}
#else
	PIOS_DEBUG_Init(&pios_tim_servo_all_channels,
			NELEMENTS(pios_tim_servo_all_channels));
#endif

	/* init sensor queue registration */
	PIOS_SENSORS_Init();

	PIOS_WDG_Clear();
	PIOS_DELAY_WaitmS(200);
	PIOS_WDG_Clear();

#if defined(PIOS_INCLUDE_MPU6000)
	if (PIOS_MPU6000_Init(pios_spi_gyro_accel_id, 0, &pios_mpu6000_cfg)
	    != 0)
		panic(2);
	if (PIOS_MPU6000_Test() != 0)
		panic(2);

	// To be safe map from UAVO enum to driver enum
	uint8_t hw_gyro_range;
	HwColibriGyroRangeGet(&hw_gyro_range);
	switch (hw_gyro_range) {
	case HWCOLIBRI_GYRORANGE_250:
		PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_250_DEG);
		break;
	case HWCOLIBRI_GYRORANGE_500:
		PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);
		break;
	case HWCOLIBRI_GYRORANGE_1000:
		PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_1000_DEG);
		break;
	case HWCOLIBRI_GYRORANGE_2000:
		PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_2000_DEG);
		break;
	}

	uint8_t hw_accel_range;
	HwColibriAccelRangeGet(&hw_accel_range);
	switch (hw_accel_range) {
	case HWCOLIBRI_ACCELRANGE_2G:
		PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_2G);
		break;
	case HWCOLIBRI_ACCELRANGE_4G:
		PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_4G);
		break;
	case HWCOLIBRI_ACCELRANGE_8G:
		PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
		break;
	case HWCOLIBRI_ACCELRANGE_16G:
		PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_16G);
		break;
	}

	// the filter has to be set before rate else divisor calculation will fail
	uint8_t hw_mpu6000_dlpf;
	HwColibriMPU6000DLPFGet(&hw_mpu6000_dlpf);
	enum pios_mpu60x0_filter mpu6000_dlpf =
	    (hw_mpu6000_dlpf ==
	     HWCOLIBRI_MPU6000DLPF_256) ? PIOS_MPU60X0_LOWPASS_256_HZ
	    : (hw_mpu6000_dlpf ==
	       HWCOLIBRI_MPU6000DLPF_188) ? PIOS_MPU60X0_LOWPASS_188_HZ
	    : (hw_mpu6000_dlpf ==
	       HWCOLIBRI_MPU6000DLPF_98) ? PIOS_MPU60X0_LOWPASS_98_HZ
	    : (hw_mpu6000_dlpf ==
	       HWCOLIBRI_MPU6000DLPF_42) ? PIOS_MPU60X0_LOWPASS_42_HZ
	    : (hw_mpu6000_dlpf ==
	       HWCOLIBRI_MPU6000DLPF_20) ? PIOS_MPU60X0_LOWPASS_20_HZ
	    : (hw_mpu6000_dlpf ==
	       HWCOLIBRI_MPU6000DLPF_10) ? PIOS_MPU60X0_LOWPASS_10_HZ
	    : (hw_mpu6000_dlpf ==
	       HWCOLIBRI_MPU6000DLPF_5) ? PIOS_MPU60X0_LOWPASS_5_HZ :
	    pios_mpu6000_cfg.default_filter;
	PIOS_MPU6000_SetLPF(mpu6000_dlpf);

	uint8_t hw_mpu6000_samplerate;
	HwColibriMPU6000RateGet(&hw_mpu6000_samplerate);
	uint16_t mpu6000_samplerate =
	    (hw_mpu6000_samplerate == HWCOLIBRI_MPU6000RATE_200) ? 200 :
	    (hw_mpu6000_samplerate == HWCOLIBRI_MPU6000RATE_333) ? 333 :
	    (hw_mpu6000_samplerate == HWCOLIBRI_MPU6000RATE_500) ? 500 :
	    (hw_mpu6000_samplerate == HWCOLIBRI_MPU6000RATE_666) ? 666 :
	    (hw_mpu6000_samplerate == HWCOLIBRI_MPU6000RATE_1000) ? 1000 :
	    (hw_mpu6000_samplerate == HWCOLIBRI_MPU6000RATE_2000) ? 2000 :
	    (hw_mpu6000_samplerate == HWCOLIBRI_MPU6000RATE_4000) ? 4000 :
	    (hw_mpu6000_samplerate == HWCOLIBRI_MPU6000RATE_8000) ? 8000 :
	    pios_mpu6000_cfg.default_samplerate;
	PIOS_MPU6000_SetSampleRate(mpu6000_samplerate);
#endif

#if defined(PIOS_INCLUDE_I2C)
#if defined(PIOS_INCLUDE_HMC5883)
	{
		uint8_t Magnetometer;
		HwColibriMagnetometerGet(&Magnetometer);

		if (Magnetometer == HWCOLIBRI_MAGNETOMETER_INTERNAL) {
			if (PIOS_HMC5883_Init (pios_i2c_internal_adapter_id, &pios_hmc5883_internal_cfg) != 0)
				panic(3);
			if (PIOS_HMC5883_Test() != 0)
				panic(3);
		}

		if (Magnetometer == HWCOLIBRI_MAGNETOMETER_EXTERNALI2CUART1) {
			// init sensor
			if (PIOS_HMC5883_Init(pios_i2c_usart1_adapter_id, &pios_hmc5883_external_cfg) != 0)
				AlarmsSet(SYSTEMALARMS_ALARM_I2C, SYSTEMALARMS_ALARM_CRITICAL);
			if (PIOS_HMC5883_Test() != 0)
				AlarmsSet(SYSTEMALARMS_ALARM_I2C, SYSTEMALARMS_ALARM_CRITICAL);
		}	
		
		if (Magnetometer == HWCOLIBRI_MAGNETOMETER_EXTERNALI2CUART3) {
				// init sensor
				if (PIOS_HMC5883_Init(pios_i2c_usart3_adapter_id, &pios_hmc5883_external_cfg) != 0)
					AlarmsSet(SYSTEMALARMS_ALARM_I2C, SYSTEMALARMS_ALARM_CRITICAL);
				if (PIOS_HMC5883_Test() != 0)
					AlarmsSet(SYSTEMALARMS_ALARM_I2C, SYSTEMALARMS_ALARM_CRITICAL);
			}
		
		if (Magnetometer == HWCOLIBRI_MAGNETOMETER_EXTERNALI2CUART1 || 
		    Magnetometer == HWCOLIBRI_MAGNETOMETER_EXTERNALI2CUART3) 
		{
			// setup sensor orientation
			uint8_t ExtMagOrientation;
			HwColibriExtMagOrientationGet(&ExtMagOrientation);

			enum pios_hmc5883_orientation hmc5883_orientation =
			    (ExtMagOrientation ==
			     HWCOLIBRI_EXTMAGORIENTATION_TOP0DEGCW) ?
			    PIOS_HMC5883_TOP_0DEG : (ExtMagOrientation ==
						     HWCOLIBRI_EXTMAGORIENTATION_TOP90DEGCW)
			    ? PIOS_HMC5883_TOP_90DEG : (ExtMagOrientation
							==
							HWCOLIBRI_EXTMAGORIENTATION_TOP180DEGCW)
			    ? PIOS_HMC5883_TOP_180DEG : (ExtMagOrientation
							 ==
							 HWCOLIBRI_EXTMAGORIENTATION_TOP270DEGCW)
			    ? PIOS_HMC5883_TOP_270DEG : (ExtMagOrientation
							 ==
							 HWCOLIBRI_EXTMAGORIENTATION_BOTTOM0DEGCW)
			    ? PIOS_HMC5883_BOTTOM_0DEG : (ExtMagOrientation
							  ==
							  HWCOLIBRI_EXTMAGORIENTATION_BOTTOM90DEGCW)
			    ? PIOS_HMC5883_BOTTOM_90DEG
			    : (ExtMagOrientation ==
			       HWCOLIBRI_EXTMAGORIENTATION_BOTTOM180DEGCW)
			    ? PIOS_HMC5883_BOTTOM_180DEG
			    : (ExtMagOrientation ==
			       HWCOLIBRI_EXTMAGORIENTATION_BOTTOM270DEGCW)
			    ? PIOS_HMC5883_BOTTOM_270DEG :
			    pios_hmc5883_external_cfg.Default_Orientation;
			PIOS_HMC5883_SetOrientation(hmc5883_orientation);
		}
	}
#endif

	//I2C is slow, sensor init as well, reset watchdog to prevent reset here
	PIOS_WDG_Clear();

#if defined(PIOS_INCLUDE_MS5611)
	if (PIOS_MS5611_Init
	    (&pios_ms5611_cfg, pios_i2c_internal_adapter_id) != 0)
		panic(4);
	if (PIOS_MS5611_Test() != 0)
		panic(4);
#endif

	//I2C is slow, sensor init as well, reset watchdog to prevent reset here
	PIOS_WDG_Clear();

#endif /* PIOS_INCLUDE_I2C */

#if defined(PIOS_INCLUDE_GPIO)
	PIOS_GPIO_Init();
#endif

#if defined(PIOS_INCLUDE_ADC)
	if (hw_rcvrport == HWCOLIBRI_RCVRPORT_PWMADC ||
	    hw_rcvrport == HWCOLIBRI_RCVRPORT_PPMADC ||
	    hw_rcvrport == HWCOLIBRI_RCVRPORT_PPMPWMADC ||
	    hw_rcvrport == HWCOLIBRI_RCVRPORT_OUTPUTSADC ||
	    hw_rcvrport == HWCOLIBRI_RCVRPORT_PPMOUTPUTSADC) {
		uint32_t internal_adc_id;
		PIOS_INTERNAL_ADC_Init(&internal_adc_id, &pios_adc_cfg);
		PIOS_ADC_Init(&pios_internal_adc_id,
			      &pios_internal_adc_driver, internal_adc_id);
	}
#endif

#if defined(PIOS_INCLUDE_FLASH)
	if ( PIOS_STREAMFS_Init(&streamfs_id, &streamfs_settings, FLASH_PARTITION_LABEL_LOG) != 0)
		panic(8);

	const uint32_t LOG_BUF_LEN = 256;
	uint8_t *log_rx_buffer = PIOS_malloc(LOG_BUF_LEN);
	uint8_t *log_tx_buffer = PIOS_malloc(LOG_BUF_LEN);
	if (PIOS_COM_Init(&pios_com_spiflash_logging_id, &pios_streamfs_com_driver, streamfs_id,
	                  log_rx_buffer, LOG_BUF_LEN, log_tx_buffer, LOG_BUF_LEN) != 0)
		panic(9);
#endif /* PIOS_INCLUDE_FLASH */

	//Set battery input pin to floating as long as it is unused
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_15;
	GPIO_Init(GPIOC, &GPIO_InitStructure);
	GPIO_ResetBits(GPIOC, GPIO_Pin_15);

	//Set buzzer output to low as long as it is unused
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_4;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	GPIO_ResetBits(GPIOA, GPIO_Pin_4);

	/* Make sure we have at least one telemetry link configured or else fail initialization */
	PIOS_Assert(pios_com_telem_serial_id || pios_com_telem_usb_id);
}

/**
 * @}
 */
