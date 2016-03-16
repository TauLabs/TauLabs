/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup Naze family support files
 * @{
 *
 * @file       pios_board.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
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
#include "hwnaze.h"
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
#if defined(PIOS_INCLUDE_MS5XXX)
#include "pios_ms5xxx_priv.h"
static const struct pios_ms5xxx_cfg pios_ms5xxx_cfg = {
	.oversampling = MS5XXX_OSR_512,
	.temperature_interleaving = 1,
	.pios_ms5xxx_model = PIOS_MS5M_MS5611,
};
#endif /* PIOS_INCLUDE_MS5XXX */

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

#define PIOS_COM_TELEM_RF_RX_BUF_LEN 32
#define PIOS_COM_TELEM_RF_TX_BUF_LEN 12

uintptr_t pios_com_lighttelemetry_id;

uintptr_t pios_uavo_settings_fs_id;

uintptr_t pios_internal_adc_id;

#if defined(PIOS_INCLUDE_MSP_BRIDGE)
extern uintptr_t pios_com_msp_id;
#endif


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
	PIOS_HAL_Panic(PIOS_LED_ALARM, code);
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

	/* Initialize the hardware UAVOs */
	HwNazeInitialize();
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

	/* Initialize the alarms library. Reads RCC reset flags */
	AlarmsInitialize();
	PIOS_RESET_Clear(); // Clear the RCC reset flags after use.

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
		HwNazeSetDefaults(HwNazeHandle(), 0);
		ModuleSettingsSetDefaults(ModuleSettingsHandle(),0);
		AlarmsSet(SYSTEMALARMS_ALARM_BOOTFAULT, SYSTEMALARMS_ALARM_CRITICAL);
	}


	/* UART1 Port */
#if (defined(PIOS_INCLUDE_TELEMETRY) || defined(PIOS_INCLUDE_MSP_BRIDGE)) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
	uint8_t hw_mainport;
	HwNazeMainPortGet(&hw_mainport);

	switch (hw_mainport) {
	case HWNAZE_MAINPORT_TELEMETRY:
#if defined(PIOS_INCLUDE_TELEMETRY)
		PIOS_HAL_ConfigureCom(&pios_usart_main_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_serial_id);
#endif /* PIOS_INCLUDE_TELEMETRY */
		break;
	case HWNAZE_MAINPORT_MSP:
#if defined(PIOS_INCLUDE_MSP_BRIDGE)
		PIOS_HAL_ConfigureCom(&pios_usart_main_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_msp_id);
#endif /* PIOS_INCLUDE_MSP_BRIDGE */
		break;
	}
#endif /* PIOS_INCLUDE_TELEMETRY || PIOS_INCLUDE_MSP_BRIDGE */

	/* Configure the rcvr port */
	uint8_t hw_rcvrport;
	HwNazeRcvrPortGet(&hw_rcvrport);

	switch (hw_rcvrport) {
	case HWNAZE_RCVRPORT_DISABLED:
		break;
	case HWNAZE_RCVRPORT_PWM:
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
	case HWNAZE_RCVRPORT_PPMSERIAL:
	case HWNAZE_RCVRPORT_PPM:
	case HWNAZE_RCVRPORT_PPMOUTPUTS:
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
	case HWNAZE_RCVRPORT_PPMPWM:
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

#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
	switch (hw_rcvrport) {
	case HWNAZE_RCVRPORT_PPMSERIAL:
	case HWNAZE_RCVRPORT_SERIAL:
		{
			uint8_t hw_rcvrserial;
			HwNazeRcvrSerialGet(&hw_rcvrserial);
			
			HwNazeDSMxModeOptions hw_DSMxMode;
			HwNazeDSMxModeGet(&hw_DSMxMode);
			
			PIOS_HAL_ConfigurePort(hw_rcvrserial,        // port type protocol
					&pios_usart_rcvrserial_cfg,          // usart_port_cfg
					&pios_usart_rcvrserial_cfg,          // frsky usart_port_cfg
					&pios_usart_com_driver,              // com_driver
					NULL,                                // i2c_id
					NULL,                                // i2c_cfg
					NULL,                                // ppm_cfg
					NULL,                                // pwm_cfg
					PIOS_LED_ALARM,                      // led_id
					&pios_usart_dsm_hsum_rcvrserial_cfg, // usart_dsm_hsum_cfg
					&pios_dsm_rcvrserial_cfg,            // dsm_cfg
					hw_DSMxMode,                         // dsm_mode
					NULL,                                // sbus_rcvr_cfg
					NULL,                                // sbus_cfg
					false);                              // sbus_toggle
		}
		break;
	default:
		break;
	}
#endif	/* PIOS_INCLUDE_USART && PIOS_INCLUDE_COM */

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
		case HWNAZE_RCVRPORT_DISABLED:
		case HWNAZE_RCVRPORT_PWM:
		case HWNAZE_RCVRPORT_PPM:
		case HWNAZE_RCVRPORT_PPMPWM:
		case HWNAZE_RCVRPORT_PPMSERIAL:
		case HWNAZE_RCVRPORT_SERIAL:
			PIOS_Servo_Init(&pios_servo_cfg);
			break;
		case HWNAZE_RCVRPORT_PPMOUTPUTS:
		case HWNAZE_RCVRPORT_OUTPUTS:
			PIOS_Servo_Init(&pios_servo_rcvr_cfg);
			break;
	}
#endif
#else
	PIOS_DEBUG_Init(&pios_tim_servo_all_channels, NELEMENTS(pios_tim_servo_all_channels));
#endif

#if defined(PIOS_INCLUDE_ADC)
	{
		uint16_t number_of_adc_pins = 2; // first two pins are always available
		switch(hw_rcvrport) {
		case HWNAZE_RCVRPORT_PPM:
		case HWNAZE_RCVRPORT_PPMSERIAL:
		case HWNAZE_RCVRPORT_SERIAL:
			number_of_adc_pins += 2; // rcvr port pins also available
			break;
		default:
			break;
		}
		uint32_t internal_adc_id;
		PIOS_INTERNAL_ADC_LIGHT_Init(&internal_adc_id, &internal_adc_cfg, number_of_adc_pins);
		PIOS_ADC_Init(&pios_internal_adc_id, &pios_internal_adc_driver, internal_adc_id);
	}
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
	HwNazeGyroRangeGet(&hw_gyro_range);
	switch(hw_gyro_range) {
		case HWNAZE_GYRORANGE_250:
			PIOS_MPU6050_SetGyroRange(PIOS_MPU60X0_SCALE_250_DEG);
			break;
		case HWNAZE_GYRORANGE_500:
			PIOS_MPU6050_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);
			break;
		case HWNAZE_GYRORANGE_1000:
			PIOS_MPU6050_SetGyroRange(PIOS_MPU60X0_SCALE_1000_DEG);
			break;
		case HWNAZE_GYRORANGE_2000:
			PIOS_MPU6050_SetGyroRange(PIOS_MPU60X0_SCALE_2000_DEG);
			break;
	}

	uint8_t hw_accel_range;
	HwNazeAccelRangeGet(&hw_accel_range);
	switch(hw_accel_range) {
		case HWNAZE_ACCELRANGE_2G:
			PIOS_MPU6050_SetAccelRange(PIOS_MPU60X0_ACCEL_2G);
			break;
		case HWNAZE_ACCELRANGE_4G:
			PIOS_MPU6050_SetAccelRange(PIOS_MPU60X0_ACCEL_4G);
			break;
		case HWNAZE_ACCELRANGE_8G:
			PIOS_MPU6050_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
			break;
		case HWNAZE_ACCELRANGE_16G:
			PIOS_MPU6050_SetAccelRange(PIOS_MPU60X0_ACCEL_16G);
			break;
	}

	// the filter has to be set before rate else divisor calculation will fail
	uint8_t hw_mpu_dlpf;
	HwNazeMPU6050DLPFGet(&hw_mpu_dlpf);
	enum pios_mpu60x0_filter mpu_dlpf = \
			    (hw_mpu_dlpf == HWNAZE_MPU6050DLPF_256) ? PIOS_MPU60X0_LOWPASS_256_HZ : \
			    (hw_mpu_dlpf == HWNAZE_MPU6050DLPF_188) ? PIOS_MPU60X0_LOWPASS_188_HZ : \
			    (hw_mpu_dlpf == HWNAZE_MPU6050DLPF_98) ? PIOS_MPU60X0_LOWPASS_98_HZ : \
			    (hw_mpu_dlpf == HWNAZE_MPU6050DLPF_42) ? PIOS_MPU60X0_LOWPASS_42_HZ : \
			    (hw_mpu_dlpf == HWNAZE_MPU6050DLPF_20) ? PIOS_MPU60X0_LOWPASS_20_HZ : \
			    (hw_mpu_dlpf == HWNAZE_MPU6050DLPF_10) ? PIOS_MPU60X0_LOWPASS_10_HZ : \
			    (hw_mpu_dlpf == HWNAZE_MPU6050DLPF_5) ? PIOS_MPU60X0_LOWPASS_5_HZ : \
			    pios_mpu6050_cfg.default_filter;
	PIOS_MPU6050_SetLPF(mpu_dlpf);

	uint8_t hw_mpu_samplerate;
	HwNazeMPU6050RateGet(&hw_mpu_samplerate);
	uint16_t mpu_samplerate = \
			    (hw_mpu_samplerate == HWNAZE_MPU6050RATE_200) ? 200 : \
			    (hw_mpu_samplerate == HWNAZE_MPU6050RATE_333) ? 333 : \
			    (hw_mpu_samplerate == HWNAZE_MPU6050RATE_500) ? 500 : \
			    pios_mpu6050_cfg.default_samplerate;
	PIOS_MPU6050_SetSampleRate(mpu_samplerate);

#endif /* PIOS_INCLUDE_MPU6050 */

	//I2C is slow, sensor init as well, reset watchdog to prevent reset here
	PIOS_WDG_Clear();

#if defined(PIOS_INCLUDE_MS5XXX)
	if (PIOS_MS5XXX_I2C_Init(pios_i2c_internal_id, MS5XXX_I2C_ADDR_0x77, &pios_ms5xxx_cfg) != 0)
		panic(4);
	if (PIOS_MS5XXX_Test() != 0)
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
}

/**
 * @}
 */
