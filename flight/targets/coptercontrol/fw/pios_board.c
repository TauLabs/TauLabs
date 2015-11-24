/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup CopterControl OpenPilot coptercontrol support files
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
#include <hwcoptercontrol.h>
#include <modulesettings.h>
#include <manualcontrolsettings.h>
#include <gcsreceiver.h>
#include <pios_hal.h>

uintptr_t pios_com_telem_rf_id;
uintptr_t pios_internal_adc_id;
uintptr_t pios_uavo_settings_fs_id;

/**
 * Configuration for MPU6000 chip
 */
#if defined(PIOS_INCLUDE_MPU6000)
#include "pios_mpu6000.h"
static const struct pios_exti_cfg pios_exti_mpu6000_cfg __exti_config = {
	.vector = PIOS_MPU6000_IRQHandler,
	.line = EXTI_Line3,
	.pin = {
		.gpio = GPIOA,
		.init = {
			.GPIO_Pin   = GPIO_Pin_3,
			.GPIO_Speed = GPIO_Speed_10MHz,
			.GPIO_Mode  = GPIO_Mode_IN_FLOATING,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI3_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGH,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line3, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};

static const struct pios_mpu60x0_cfg pios_mpu6000_cfg = {
	.exti_cfg = &pios_exti_mpu6000_cfg,
	.default_samplerate = 500,
	.interrupt_cfg = PIOS_MPU60X0_INT_CLR_ANYRD,
	.interrupt_en = PIOS_MPU60X0_INTEN_DATA_RDY,
	.User_ctl = PIOS_MPU60X0_USERCTL_DIS_I2C,
	.Pwr_mgmt_clk = PIOS_MPU60X0_PWRMGMT_PLL_Z_CLK,
	.default_filter = PIOS_MPU60X0_LOWPASS_256_HZ,
	.orientation = PIOS_MPU60X0_TOP_180DEG
};
#endif /* PIOS_INCLUDE_MPU6000 */

#include <pios_board_info.h>
/**
 * PIOS_Board_Init()
 * initializes all the core subsystems on this specific hardware
 * called from System/openpilot.c
 */
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
	/* Set up the SPI interface to the serial flash */

	switch(bdinfo->board_rev) {
		case BOARD_REVISION_CC:
			if (PIOS_SPI_Init(&pios_spi_flash_accel_id, &pios_spi_flash_accel_cfg_cc)) {
				PIOS_Assert(0);
			}
			break;
		case BOARD_REVISION_CC3D:
			if (PIOS_SPI_Init(&pios_spi_flash_accel_id, &pios_spi_flash_accel_cfg_cc3d)) {
				PIOS_Assert(0);
			}
			break;
		default:
			PIOS_Assert(0);
	}

#endif

	switch(bdinfo->board_rev) {
		case BOARD_REVISION_CC:
			PIOS_Flash_Jedec_Init(&pios_external_flash_id, pios_spi_flash_accel_id, 1, &flash_w25x_cfg);
			break;
		case BOARD_REVISION_CC3D:
			PIOS_Flash_Jedec_Init(&pios_external_flash_id, pios_spi_flash_accel_id, 0, &flash_m25p_cfg);
			break;
		default:
			PIOS_DEBUG_Assert(0);
	}

	PIOS_Flash_Internal_Init(&pios_internal_flash_id, &flash_internal_cfg);

	/* Register the partition table */
	const struct pios_flash_partition * flash_partition_table;
	uint32_t num_partitions;
	flash_partition_table = PIOS_BOARD_HW_DEFS_GetPartitionTable(bdinfo->board_rev, &num_partitions);
	PIOS_FLASH_register_partition_table(flash_partition_table, num_partitions);

	/* Mount all filesystems */
	PIOS_FLASHFS_Logfs_Init(&pios_uavo_settings_fs_id, &flashfs_settings_cfg, FLASH_PARTITION_LABEL_SETTINGS);

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Initialize UAVObject libraries */
	UAVObjInitialize();

	/* Initialize the alarms library. Reads RCC reset flags */
	AlarmsInitialize();
	PIOS_RESET_Clear(); // Clear the RCC reset flags after use.

	/* Initialize the hardware UAVOs */
	HwCopterControlInitialize();
	ModuleSettingsInitialize();

#if defined(PIOS_INCLUDE_RTC)
	/* Initialize the real-time clock and its associated tick */
	PIOS_RTC_Init(&pios_rtc_main_cfg);
#endif

#ifndef ERASE_FLASH
	/* Initialize watchdog as early as possible to catch faults during init */
	PIOS_WDG_Init();
#endif

	/* Check for repeated boot failures */
	PIOS_IAP_Init();
	uint16_t boot_count = PIOS_IAP_ReadBootCount();
	if (boot_count < 3) {
		PIOS_IAP_WriteBootCount(++boot_count);
		AlarmsClear(SYSTEMALARMS_ALARM_BOOTFAULT);
	} else {
		/* Too many failed boot attempts, force hw configuration to defaults */
		HwCopterControlSetDefaults(HwCopterControlHandle(), 0);
		ModuleSettingsSetDefaults(ModuleSettingsHandle(),0);
		AlarmsSet(SYSTEMALARMS_ALARM_BOOTFAULT, SYSTEMALARMS_ALARM_CRITICAL);
	}

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Set up pulse timers */
	PIOS_TIM_InitClock(&tim_1_cfg);
	PIOS_TIM_InitClock(&tim_2_cfg);
	PIOS_TIM_InitClock(&tim_3_cfg);
	PIOS_TIM_InitClock(&tim_4_cfg);

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
	
	switch(bdinfo->board_rev) {
		case BOARD_REVISION_CC:
			PIOS_USB_Init(&pios_usb_id, &pios_usb_main_cfg_cc);
			break;
		case BOARD_REVISION_CC3D:
			PIOS_USB_Init(&pios_usb_id, &pios_usb_main_cfg_cc3d);
			break;
		default:
			PIOS_Assert(0);
	}

#if defined(PIOS_INCLUDE_USB_CDC)

	uint8_t hw_usb_vcpport;
	/* Configure the USB VCP port */
	HwCopterControlUSB_VCPPortGet(&hw_usb_vcpport);

	if (!usb_cdc_present) {
		/* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
		hw_usb_vcpport = HWCOPTERCONTROL_USB_VCPPORT_DISABLED;
	}

	PIOS_HAL_ConfigureCDC(hw_usb_vcpport, pios_usb_id,
			&pios_usb_cdc_cfg);
#endif	/* PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_USB_HID)
	/* Configure the usb HID port */
	uint8_t hw_usb_hidport;
	HwCopterControlUSB_HIDPortGet(&hw_usb_hidport);

	if (!usb_hid_present) {
		/* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
		hw_usb_hidport = HWCOPTERCONTROL_USB_HIDPORT_DISABLED;
	}

	PIOS_HAL_ConfigureHID(hw_usb_hidport, pios_usb_id, &pios_usb_hid_cfg);
#endif /* PIOS_INCLUDE_USB_HID */

#endif /* PIOS_INCLUDE_USB */

	/* Configure the main IO port */
	HwCopterControlDSMxModeOptions hw_DSMxMode;
	HwCopterControlDSMxModeGet(&hw_DSMxMode);
	uint8_t hw_mainport;
	HwCopterControlMainPortGet(&hw_mainport);

	PIOS_HAL_ConfigurePort(hw_mainport, &pios_usart_generic_main_cfg,
			&pios_usart_com_driver, NULL, NULL, NULL, NULL,
			0,
			&pios_usart_dsm_hsum_main_cfg, &pios_dsm_main_cfg,
			hw_DSMxMode, &pios_usart_sbus_main_cfg, &pios_sbus_cfg,
			true);

	/* Configure the flexi port */
	uint8_t hw_flexiport;
	HwCopterControlFlexiPortGet(&hw_flexiport);

	PIOS_HAL_ConfigurePort(hw_flexiport, &pios_usart_generic_flexi_cfg,
			&pios_usart_com_driver,
#ifdef PIOS_INCLUDE_I2C
			&pios_i2c_flexi_adapter_id, &pios_i2c_flexi_adapter_cfg,
#else
			NULL, NULL,
#endif
			NULL, NULL, 0,
			&pios_usart_dsm_hsum_flexi_cfg, &pios_dsm_flexi_cfg,
			hw_DSMxMode, NULL, NULL, true);

	/* Configure the rcvr port */
	uint8_t hw_rcvrport;
	HwCopterControlRcvrPortGet(&hw_rcvrport);

	switch (hw_rcvrport) {
	case HWCOPTERCONTROL_RCVRPORT_DISABLED:
		break;
	case HWCOPTERCONTROL_RCVRPORT_PWM:
#if defined(PIOS_INCLUDE_PWM)
		{
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
	case HWCOPTERCONTROL_RCVRPORT_PPM:
	case HWCOPTERCONTROL_RCVRPORT_PPMONPIN8:
	case HWCOPTERCONTROL_RCVRPORT_PPMOUTPUTS:
#if defined(PIOS_INCLUDE_PPM)
		{
			uintptr_t pios_ppm_id;
			PIOS_PPM_Init(&pios_ppm_id,
					(hw_rcvrport == HWCOPTERCONTROL_RCVRPORT_PPMONPIN8) ? &pios_ppm_pin8_cfg : &pios_ppm_cfg);

			uintptr_t pios_ppm_rcvr_id;
			if (PIOS_RCVR_Init(&pios_ppm_rcvr_id, &pios_ppm_rcvr_driver, pios_ppm_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PPM] = pios_ppm_rcvr_id;
		}
#endif	/* PIOS_INCLUDE_PPM */
		break;
	case HWCOPTERCONTROL_RCVRPORT_PPMPWM:
		/* This is a combination of PPM and PWM inputs */
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
#if defined(PIOS_INCLUDE_PWM)
		{
			uintptr_t pios_pwm_id;
			PIOS_PWM_Init(&pios_pwm_id, &pios_pwm_with_ppm_cfg);

			uintptr_t pios_pwm_rcvr_id;
			if (PIOS_RCVR_Init(&pios_pwm_rcvr_id, &pios_pwm_rcvr_driver, pios_pwm_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PWM] = pios_pwm_rcvr_id;
		}
#endif	/* PIOS_INCLUDE_PWM */
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

	/* Remap AFIO pin for PB4 (Servo 5 Out)*/
	GPIO_PinRemapConfig( GPIO_Remap_SWJ_NoJTRST, ENABLE);

#ifndef PIOS_DEBUG_ENABLE_DEBUG_PINS
	switch (hw_rcvrport) {
		case HWCOPTERCONTROL_RCVRPORT_DISABLED:
		case HWCOPTERCONTROL_RCVRPORT_PWM:
		case HWCOPTERCONTROL_RCVRPORT_PPM:
		case HWCOPTERCONTROL_RCVRPORT_PPMONPIN8:
		case HWCOPTERCONTROL_RCVRPORT_PPMPWM:
			PIOS_Servo_Init(&pios_servo_cfg);
			break;
		case HWCOPTERCONTROL_RCVRPORT_PPMOUTPUTS:
		case HWCOPTERCONTROL_RCVRPORT_OUTPUTS:
			PIOS_Servo_Init(&pios_servo_rcvr_cfg);
			break;
	}
#else
	PIOS_DEBUG_Init(&pios_tim_servo_all_channels, NELEMENTS(pios_tim_servo_all_channels));
#endif	/* PIOS_DEBUG_ENABLE_DEBUG_PINS */

	PIOS_SENSORS_Init();

	switch(bdinfo->board_rev) {
		case BOARD_REVISION_CC:
			// Revision 1 with invensense gyros, start the ADC
#if defined(PIOS_INCLUDE_ADC)
		{
			uint32_t internal_adc_id;
			PIOS_INTERNAL_ADC_Init(&internal_adc_id, &internal_adc_cfg);
			PIOS_ADC_Init(&pios_internal_adc_id, &pios_internal_adc_driver, internal_adc_id);
		}
#endif
#if defined(PIOS_INCLUDE_ADXL345)
			PIOS_ADXL345_Init(pios_spi_flash_accel_id, 0);
#endif
			break;
		case BOARD_REVISION_CC3D:
			// Revision 2 with L3GD20 gyros, start a SPI interface and connect to it
			GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE);
#if defined(PIOS_INCLUDE_MPU6000)
			// Set up the SPI interface to the serial flash 
			if (PIOS_SPI_Init(&pios_spi_gyro_id, &pios_spi_gyro_cfg)) {
				PIOS_Assert(0);
			}
			PIOS_MPU6000_Init(pios_spi_gyro_id,0,&pios_mpu6000_cfg);
			PIOS_MPU6000_Test();

			uint8_t hw_gyro_range;
			HwCopterControlGyroRangeGet(&hw_gyro_range);
			switch(hw_gyro_range) {
				case HWCOPTERCONTROL_GYRORANGE_250:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_250_DEG);
					break;
				case HWCOPTERCONTROL_GYRORANGE_500:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);
					break;
				case HWCOPTERCONTROL_GYRORANGE_1000:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_1000_DEG);
					break;
				case HWCOPTERCONTROL_GYRORANGE_2000:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_2000_DEG);
					break;
			}

			uint8_t hw_accel_range;
			HwCopterControlAccelRangeGet(&hw_accel_range);
			switch(hw_accel_range) {
				case HWCOPTERCONTROL_ACCELRANGE_2G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_2G);
					break;
				case HWCOPTERCONTROL_ACCELRANGE_4G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_4G);
					break;
				case HWCOPTERCONTROL_ACCELRANGE_8G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
					break;
				case HWCOPTERCONTROL_ACCELRANGE_16G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_16G);
					break;
			}

			// the filter has to be set before rate else divisor calculation will fail
			uint8_t hw_mpu6000_dlpf;
			HwCopterControlMPU6000DLPFGet(&hw_mpu6000_dlpf);
			enum pios_mpu60x0_filter mpu6000_dlpf = \
			    (hw_mpu6000_dlpf == HWCOPTERCONTROL_MPU6000DLPF_256) ? PIOS_MPU60X0_LOWPASS_256_HZ : \
			    (hw_mpu6000_dlpf == HWCOPTERCONTROL_MPU6000DLPF_188) ? PIOS_MPU60X0_LOWPASS_188_HZ : \
			    (hw_mpu6000_dlpf == HWCOPTERCONTROL_MPU6000DLPF_98) ? PIOS_MPU60X0_LOWPASS_98_HZ : \
			    (hw_mpu6000_dlpf == HWCOPTERCONTROL_MPU6000DLPF_42) ? PIOS_MPU60X0_LOWPASS_42_HZ : \
			    (hw_mpu6000_dlpf == HWCOPTERCONTROL_MPU6000DLPF_20) ? PIOS_MPU60X0_LOWPASS_20_HZ : \
			    (hw_mpu6000_dlpf == HWCOPTERCONTROL_MPU6000DLPF_10) ? PIOS_MPU60X0_LOWPASS_10_HZ : \
			    (hw_mpu6000_dlpf == HWCOPTERCONTROL_MPU6000DLPF_5) ? PIOS_MPU60X0_LOWPASS_5_HZ : \
			    pios_mpu6000_cfg.default_filter;
			PIOS_MPU6000_SetLPF(mpu6000_dlpf);

			uint8_t hw_mpu6000_samplerate;
			HwCopterControlMPU6000RateGet(&hw_mpu6000_samplerate);
			uint16_t mpu6000_samplerate = \
			    (hw_mpu6000_samplerate == HWCOPTERCONTROL_MPU6000RATE_200) ? 200 : \
			    (hw_mpu6000_samplerate == HWCOPTERCONTROL_MPU6000RATE_333) ? 333 : \
			    (hw_mpu6000_samplerate == HWCOPTERCONTROL_MPU6000RATE_500) ? 500 : \
			    (hw_mpu6000_samplerate == HWCOPTERCONTROL_MPU6000RATE_666) ? 666 : \
			    (hw_mpu6000_samplerate == HWCOPTERCONTROL_MPU6000RATE_1000) ? 1000 : \
			    (hw_mpu6000_samplerate == HWCOPTERCONTROL_MPU6000RATE_2000) ? 2000 : \
			    (hw_mpu6000_samplerate == HWCOPTERCONTROL_MPU6000RATE_4000) ? 4000 : \
			    (hw_mpu6000_samplerate == HWCOPTERCONTROL_MPU6000RATE_8000) ? 8000 : \
			    pios_mpu6000_cfg.default_samplerate;
			PIOS_MPU6000_SetSampleRate(mpu6000_samplerate);

#endif /* PIOS_INCLUDE_MPU6000 */

			break;
		default:
			PIOS_Assert(0);
	}

	PIOS_GPIO_Init();

	/* Make sure we have at least one telemetry link configured or else fail initialization */
	PIOS_Assert(pios_com_telem_rf_id || pios_com_telem_usb_id);
}

/**
 * @}
 */
