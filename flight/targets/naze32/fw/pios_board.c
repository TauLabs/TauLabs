/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup CopterControl OpenPilot coptercontrol support files
 * @{
 *
 * @file       pios_board.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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
#include "hwcoptercontrol.h"
#include "manualcontrolsettings.h"
#include "modulesettings.h"

/**
 * Configuration for the HMC5883L chip
 */
#if defined(PIOS_INCLUDE_HMC5883)
#include "pios_hmc5883_priv.h"
static struct pios_exti_cfg pios_exti_hmc5883_cfg __exti_config = {
	// MAG_DRDY output on rev4 hardware (PB12)
	.vector = PIOS_HMC5883_IRQHandler,
	.line = EXTI_Line12,
	.pin = {
		.gpio = GPIOB,
		.init = {
			.GPIO_Pin = GPIO_Pin_12,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_IN_FLOATING,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI15_10_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_LOW,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line12, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};

static struct pios_exti_cfg pios_exti_hmc5883_cfg_v5 __exti_config = {
        // MAG_DRDY output on rev5 hardware PC14
	.vector = PIOS_HMC5883_IRQHandler,
	.line = EXTI_Line14,
	.pin = {
		.gpio = GPIOC,
		.init = {
			.GPIO_Pin = GPIO_Pin_14,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode = GPIO_Mode_IN_FLOATING,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI15_10_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_LOW,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line14, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};

static /*const*/ struct pios_hmc5883_cfg pios_hmc5883_cfg = {
	.exti_cfg = &pios_exti_hmc5883_cfg,
	.M_ODR = PIOS_HMC5883_ODR_75,
	.Meas_Conf = PIOS_HMC5883_MEASCONF_NORMAL,
	.Gain = PIOS_HMC5883_GAIN_1_9,
	.Mode = PIOS_HMC5883_MODE_CONTINUOUS,
	.Default_Orientation = PIOS_HMC5883_TOP_90DEG,  // TODO: check & fix orientation
};

#endif /* PIOS_INCLUDE_HMC5883 */

/**
 * Configuration for the MS5611 chip
 */
#if defined(PIOS_INCLUDE_MS5611)
#include "pios_ms5611_priv.h"
static const struct pios_ms5611_cfg pios_ms5611_cfg = {
	.oversampling = MS5611_OSR_512,
	.temperature_interleaving = 1,
};
#endif /* PIOS_INCLUDE_MS5611 */

/**
 * Configuration for the MPU6050 chip
 */
#if defined(PIOS_INCLUDE_MPU6050)
#include "pios_mpu6050.h"
//#define PIOS_MPU6050_I2C_ADDR PIOS_MPU6050_I2C_ADD_A0_HIGH
#define PIOS_MPU6050_I2C_ADDR PIOS_MPU6050_I2C_ADD_A0_LOW
static const struct pios_exti_cfg pios_exti_mpu6050_cfg __exti_config = {
	// MPU_INT output on rev4 hardware (PB13)
	.vector = PIOS_MPU6050_IRQHandler,
	.line = EXTI_Line13,
	.pin = {
		.gpio = GPIOB,
		.init = {
			.GPIO_Pin   = GPIO_Pin_13,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_IN_FLOATING,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI15_10_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGH,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line13, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};

static const struct pios_exti_cfg pios_exti_mpu6050_cfg_v5 __exti_config = {
	// MPU_INT output on rev5 hardware (PC13)
	.vector = PIOS_MPU6050_IRQHandler,
	.line = EXTI_Line13,
	.pin = {
		.gpio = GPIOC,
		.init = {
			.GPIO_Pin   = GPIO_Pin_13,
			.GPIO_Speed = GPIO_Speed_2MHz,
			.GPIO_Mode  = GPIO_Mode_IN_FLOATING,
		},
	},
	.irq = {
		.init = {
			.NVIC_IRQChannel = EXTI15_10_IRQn,
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGH,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line13, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};

static /*const*/ struct pios_mpu60x0_cfg pios_mpu6050_cfg = {
	.exti_cfg = &pios_exti_mpu6050_cfg,
	.default_samplerate = 400,
	.interrupt_cfg = PIOS_MPU60X0_INT_CLR_ANYRD | PIOS_MPU60X0_INT_I2C_BYPASS_EN,
	.interrupt_en = PIOS_MPU60X0_INTEN_DATA_RDY,
	.User_ctl = 0,
	.Pwr_mgmt_clk = PIOS_MPU60X0_PWRMGMT_PLL_Z_CLK,
	.default_filter = PIOS_MPU60X0_LOWPASS_256_HZ,
	.orientation = PIOS_MPU60X0_TOP_90DEG
};
#endif /* PIOS_INCLUDE_MPU6050 */

/**
 * One slot per selectable receiver group.
 *  eg. PWM, PPM, GCS, DSMMAINPORT, DSMFLEXIPORT, SBUS
 * NOTE: No slot in this map for NONE.
 */
uintptr_t pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE];

#define PIOS_COM_TELEM_RF_RX_BUF_LEN 32
#define PIOS_COM_TELEM_RF_TX_BUF_LEN 12

#define PIOS_COM_GPS_RX_BUF_LEN 32
#define PIOS_COM_GPS_TX_BUF_LEN 16

//#define PIOS_COM_BRIDGE_RX_BUF_LEN 65
//#define PIOS_COM_BRIDGE_TX_BUF_LEN 12

//#define PIOS_COM_MAVLINK_TX_BUF_LEN 32
//#define PIOS_COM_LIGHTTELEMETRY_TX_BUF_LEN 19



//#define PIOS_COM_FRSKYSENSORHUB_TX_BUF_LEN 128

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
#define PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN 40
uintptr_t pios_com_debug_id;
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */


uintptr_t pios_com_gps_id;
uintptr_t pios_com_telem_rf_id;
uintptr_t pios_com_bridge_id;
uintptr_t pios_com_mavlink_id;
uintptr_t pios_com_frsky_sensor_hub_id;
uintptr_t pios_com_lighttelemetry_id;

uintptr_t pios_uavo_settings_fs_id;

uintptr_t pios_internal_adc_id;


/*
 * Setup a com port based on the passed cfg, driver and buffer sizes. tx size of -1 make the port rx only
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


/**
 * Indicate a target-specific error code when a component fails to initialize
 * 1 pulse - MPU6050 - no irq
 * 2 pulses - MPU6050 - failed configuration or task starting
 * 3 pulses - internal I2C bus locked
 * 4 pulses - ms5611
 * 5 pulses - flash
 * 6 pulses - hmc5883l
 */
void panic(int32_t code) {
	while(1){
		for (int32_t i = 0; i < code; i++) {
			PIOS_WDG_Clear();
			PIOS_LED_Toggle(PIOS_LED_ALARM);
			PIOS_DELAY_WaitmS(200);
			PIOS_WDG_Clear();
			PIOS_LED_Toggle(PIOS_LED_ALARM);
			PIOS_DELAY_WaitmS(200);
		}
		PIOS_WDG_Clear();
		PIOS_DELAY_WaitmS(200);
		PIOS_WDG_Clear();
		PIOS_DELAY_WaitmS(200);
		PIOS_WDG_Clear();
		PIOS_DELAY_WaitmS(100);
	}
}

/**
 * PIOS_Board_Init()
 * initializes all the core subsystems on this specific hardware
 * called from System/openpilot.c
 */

//#include <pios_board_info.h>
//from flight/targets/naze32/board-info/system_stm32f10x.c
extern uint32_t hse_value;

void PIOS_Board_Init(void) {

	/* Delay system */
	PIOS_DELAY_Init();

	bool board_v5;
	if (hse_value == 12000000)
 		board_v5 = true;
	else
		board_v5 = false;

	//TODO: Buzzer
	//rev5 needs inverted beeper. 

	//const struct pios_board_info * bdinfo = &pios_board_info_blob;

#if defined(PIOS_INCLUDE_LED)
	const struct pios_led_cfg * led_cfg = PIOS_BOARD_HW_DEFS_GetLedCfg(1);
	PIOS_Assert(led_cfg);
	PIOS_LED_Init(led_cfg);
#endif	/* PIOS_INCLUDE_LED */

#if defined(PIOS_INCLUDE_SPI)
	/* Set up the SPI interface to the serial flash */

	if (PIOS_SPI_Init(&pios_spi_generic_id, &pios_spi_generic_cfg)) {
		PIOS_Assert(0);
	}
#endif

#if defined(PIOS_INCLUDE_I2C)
	if (PIOS_I2C_Init(&pios_i2c_internal_id, &pios_i2c_internal_cfg)) {
		PIOS_DEBUG_Assert(0);
	}
	if (PIOS_I2C_CheckClear(pios_i2c_internal_id) != 0)
		panic(3);
#endif

#if defined(PIOS_INCLUDE_FLASH)
	/* Inititialize all flash drivers */
	if (PIOS_Flash_Internal_Init(&pios_internal_flash_id, &flash_internal_cfg) != 0)
		panic(5);

	/* Register the partition table */
	const struct pios_flash_partition * flash_partition_table;
	uint32_t num_partitions;
	flash_partition_table = PIOS_BOARD_HW_DEFS_GetPartitionTable(1, &num_partitions);
	PIOS_FLASH_register_partition_table(flash_partition_table, num_partitions);

	/* Mount all filesystems */
	if (PIOS_FLASHFS_Logfs_Init(&pios_uavo_settings_fs_id, &flashfs_internal_settings_cfg, FLASH_PARTITION_LABEL_SETTINGS) != 0)
		panic(5);

#endif	/* PIOS_INCLUDE_FLASH */

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();

	HwCopterControlInitialize();
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

	/* Initialize the alarms library */
	AlarmsInitialize();

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Set up pulse timers */
	//inputs

	//outputs
	PIOS_TIM_InitClock(&tim_1_cfg);
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
		HwCopterControlSetDefaults(HwCopterControlHandle(), 0);
		ModuleSettingsSetDefaults(ModuleSettingsHandle(),0);
		AlarmsSet(SYSTEMALARMS_ALARM_BOOTFAULT, SYSTEMALARMS_ALARM_CRITICAL);
	}


	/* UART1 Port */
#if defined(PIOS_INCLUDE_TELEMETRY_RF) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
        PIOS_Board_configure_com(&pios_main_usart_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
#endif /* PIOS_INCLUDE_TELEMETRY_RF */

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
	case HWCOPTERCONTROL_RCVRPORT_PPMOUTPUTS:
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
#ifdef PIOS_INCLUDE_SERVO
	switch (hw_rcvrport) {
		case HWCOPTERCONTROL_RCVRPORT_DISABLED:
		case HWCOPTERCONTROL_RCVRPORT_PWM:
		case HWCOPTERCONTROL_RCVRPORT_PPM:
		case HWCOPTERCONTROL_RCVRPORT_PPMPWM:
			PIOS_Servo_Init(&pios_servo_cfg);
			break;
		case HWCOPTERCONTROL_RCVRPORT_PPMOUTPUTS:
		case HWCOPTERCONTROL_RCVRPORT_OUTPUTS:
			PIOS_Servo_Init(&pios_servo_rcvr_cfg);
			break;
	}
#endif
#else
	PIOS_DEBUG_Init(&pios_tim_servo_all_channels, NELEMENTS(pios_tim_servo_all_channels));
#endif

#if defined(PIOS_INCLUDE_ADC)
#error "Not yet implemented"
#endif /* PIOS_INCLUDE_ADC */
	PIOS_WDG_Clear();
	PIOS_DELAY_WaitmS(200);
	PIOS_WDG_Clear();

	PIOS_SENSORS_Init();

	GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE);

#if defined(PIOS_INCLUDE_MPU6050)
	if(board_v5) { 
		// rev5 hardware use PC13 instead of PB13 for MPU_INT
		pios_mpu6050_cfg.exti_cfg = &pios_exti_mpu6050_cfg_v5;
	}

	if (PIOS_MPU6050_Init(pios_i2c_internal_id, PIOS_MPU6050_I2C_ADD_A0_LOW, &pios_mpu6050_cfg) != 0)
		panic(2);
	if (PIOS_MPU6050_Test() != 0)
		panic(2);

	uint8_t hw_gyro_range;
	HwCopterControlGyroRangeGet(&hw_gyro_range);
	switch(hw_gyro_range) {
		case HWCOPTERCONTROL_GYRORANGE_250:
			PIOS_MPU6050_SetGyroRange(PIOS_MPU60X0_SCALE_250_DEG);
			break;
		case HWCOPTERCONTROL_GYRORANGE_500:
			PIOS_MPU6050_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);
			break;
		case HWCOPTERCONTROL_GYRORANGE_1000:
			PIOS_MPU6050_SetGyroRange(PIOS_MPU60X0_SCALE_1000_DEG);
			break;
		case HWCOPTERCONTROL_GYRORANGE_2000:
			PIOS_MPU6050_SetGyroRange(PIOS_MPU60X0_SCALE_2000_DEG);
			break;
	}

	uint8_t hw_accel_range;
	HwCopterControlAccelRangeGet(&hw_accel_range);
	switch(hw_accel_range) {
		case HWCOPTERCONTROL_ACCELRANGE_2G:
			PIOS_MPU6050_SetAccelRange(PIOS_MPU60X0_ACCEL_2G);
			break;
		case HWCOPTERCONTROL_ACCELRANGE_4G:
			PIOS_MPU6050_SetAccelRange(PIOS_MPU60X0_ACCEL_4G);
			break;
		case HWCOPTERCONTROL_ACCELRANGE_8G:
			PIOS_MPU6050_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
			break;
		case HWCOPTERCONTROL_ACCELRANGE_16G:
			PIOS_MPU6050_SetAccelRange(PIOS_MPU60X0_ACCEL_16G);
			break;
	}

#endif /* PIOS_INCLUDE_MPU6050 */

	//I2C is slow, sensor init as well, reset watchdog to prevent reset here
	PIOS_WDG_Clear();

#if defined(PIOS_INCLUDE_MS5611)
	if (PIOS_MS5611_Init(&pios_ms5611_cfg, pios_i2c_internal_id) != 0)
		panic(4);
	if (PIOS_MS5611_Test() != 0)
		panic(4);
#endif

#if defined(PIOS_INCLUDE_HMC5883)
	//TODO: if(board_v5) { /* use PC14 instead of PB12 for MAG_DRDY (exti) */ }
	if (PIOS_HMC5883_Init(pios_i2c_internal_id, &pios_hmc5883_cfg) != 0)
		panic(6);
        if (PIOS_HMC5883_Test() != 0)
		panic(6);
#endif

#if defined(PIOS_INCLUDE_GPIO)
	PIOS_GPIO_Init();
#endif

	/* Make sure we have at least one telemetry link configured or else fail initialization */
	PIOS_Assert(pios_com_telem_rf_id);
}

/**
 * @}
 */
