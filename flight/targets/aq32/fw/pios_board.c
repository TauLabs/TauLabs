/**
 ******************************************************************************
 * @addtogroup AeroQuadTargets AeroQuad Targets
 * @{
 * @addtogroup AQ32 AQ32 support files
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
#include "hwaq32.h"
#include "loggingsettings.h"
#include "manualcontrolsettings.h"
#include "modulesettings.h"

/**
 * Configuration for the MPU6000 chip
 */
#if defined(PIOS_INCLUDE_MPU6000)
#include "pios_mpu6000.h"
static const struct pios_exti_cfg pios_exti_mpu6000_cfg __exti_config = {
    .vector = PIOS_MPU6000_IRQHandler,
    .line = EXTI_Line4,
    .pin = {
        .gpio = GPIOE,
        .init = {
            .GPIO_Pin   = GPIO_Pin_4,
            .GPIO_Speed = GPIO_Speed_100MHz,
            .GPIO_Mode  = GPIO_Mode_IN,
            .GPIO_OType = GPIO_OType_OD,
            .GPIO_PuPd  = GPIO_PuPd_NOPULL,
        },
    },
    .irq = {
        .init = {
            .NVIC_IRQChannel                   = EXTI4_IRQn,
            .NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGH,
            .NVIC_IRQChannelSubPriority        = 0,
            .NVIC_IRQChannelCmd                = ENABLE,
        },
    },
    .exti = {
        .init = {
            .EXTI_Line    = EXTI_Line4, // matches above GPIO pin
            .EXTI_Mode    = EXTI_Mode_Interrupt,
            .EXTI_Trigger = EXTI_Trigger_Rising,
            .EXTI_LineCmd = ENABLE,
        },
    },
};

static const struct pios_mpu60x0_cfg pios_mpu6000_cfg = {
    .exti_cfg            = &pios_exti_mpu6000_cfg,
    .default_samplerate  = 666,
    .interrupt_cfg       = PIOS_MPU60X0_INT_CLR_ANYRD,
    .interrupt_en        = PIOS_MPU60X0_INTEN_DATA_RDY,
    .User_ctl            = PIOS_MPU60X0_USERCTL_DIS_I2C,
    .Pwr_mgmt_clk        = PIOS_MPU60X0_PWRMGMT_PLL_Z_CLK,
    .default_filter      = PIOS_MPU60X0_LOWPASS_256_HZ,
    .orientation         = PIOS_MPU60X0_TOP_90DEG
};
#endif /* PIOS_INCLUDE_MPU6000 */

/**
 * Configuration for the HMC5883 chip
 */
#if defined(PIOS_INCLUDE_HMC5883)
#include "pios_hmc5883_priv.h"
static const struct pios_exti_cfg pios_exti_hmc5883_internal_cfg __exti_config = {
    .vector = PIOS_HMC5883_IRQHandler,
    .line = EXTI_Line2,
    .pin = {
        .gpio = GPIOE,
        .init = {
            .GPIO_Pin   = GPIO_Pin_2,
            .GPIO_Speed = GPIO_Speed_100MHz,
            .GPIO_Mode  = GPIO_Mode_IN,
            .GPIO_OType = GPIO_OType_OD,
            .GPIO_PuPd  = GPIO_PuPd_NOPULL,
        },
    },
    .irq = {
        .init = {
            .NVIC_IRQChannel                   = EXTI2_IRQn,
            .NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_LOW,
            .NVIC_IRQChannelSubPriority        = 0,
            .NVIC_IRQChannelCmd                = ENABLE,
        },
    },
    .exti = {
        .init = {
            .EXTI_Line    = EXTI_Line2, // matches above GPIO pin
            .EXTI_Mode    = EXTI_Mode_Interrupt,
            .EXTI_Trigger = EXTI_Trigger_Rising,
            .EXTI_LineCmd = ENABLE,
        },
    },
};

static const struct pios_hmc5883_cfg pios_hmc5883_internal_cfg = {
    .exti_cfg            = &pios_exti_hmc5883_internal_cfg,
    .M_ODR               = PIOS_HMC5883_ODR_75,
    .Meas_Conf           = PIOS_HMC5883_MEASCONF_NORMAL,
    .Gain                = PIOS_HMC5883_GAIN_1_9,
    .Mode                = PIOS_HMC5883_MODE_CONTINUOUS,
    .Default_Orientation = PIOS_HMC5883_TOP_270DEG,
};

static const struct pios_hmc5883_cfg pios_hmc5883_external_cfg = {
    .M_ODR               = PIOS_HMC5883_ODR_75,
    .Meas_Conf           = PIOS_HMC5883_MEASCONF_NORMAL,
    .Gain                = PIOS_HMC5883_GAIN_1_9,
    .Mode                = PIOS_HMC5883_MODE_SINGLE,
    .Default_Orientation = PIOS_HMC5883_TOP_270DEG,
};
#endif /* PIOS_INCLUDE_HMC5883 */

/**
 * Configuration for the MS5611 chip
 */
#if defined(PIOS_INCLUDE_MS5611)
#include "pios_ms5611_priv.h"
static const struct pios_ms5611_cfg pios_ms5611_cfg = {
    .oversampling             = MS5611_OSR_4096,
    .temperature_interleaving = 1,
    .use_0x76_address         = true,
};
#endif /* PIOS_INCLUDE_MS5611 */

bool external_mag_fail;

uintptr_t pios_com_logging_id;
uintptr_t pios_uavo_settings_fs_id;
uintptr_t pios_waypoints_settings_fs_id;
uintptr_t pios_internal_adc_id;

/**
 * Indicate a target-specific error code when a component fails to initialize
 *  1 pulse:  Flash   - PIOS_Flash_Internal_Init failed
 *  2 pulses: Flash   - PIOS_FLASHFS_Logfs_Init failed (settings)
 *  3 pulses: Flash   - PIOS_FLASHFS_Logfs_Init failed (waypoints)
 *  4 pulse:  MPU6000 - PIOS_MPU6000_Init failed
 *  5 pulses: MPU6000 - PIOS_MPU6000_Test failed
 *  6 pulses: HMC5883 - PIOS_HMC5883_Init failed (internal)
 *  7 pulses: HMC5883 - PIOS_HMC5883_Test failed (internal)
 *  8 pulses: I2C     - Internal I2C bus locked
 *  9 Not Used
 * 10 Not Used
 * 11 Not Used
 * 12 pulses: MS5611  - PIOS_MS5611_Init failed
 * 13 pulses: MS5611  - PIOS_MS5611_Test failed
 * 14 pulses: ADC     - PIOS_INTERNAL_ADC_Init failed
 * 15 pulses: ADC     - PIOS_ADC_Init failed
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
#endif    /* PIOS_INCLUDE_LED */

#if defined(PIOS_INCLUDE_I2C)
    if (PIOS_I2C_Init(&pios_i2c_internal_id, &pios_i2c_internal_cfg)) {
        PIOS_DEBUG_Assert(0);
    }
    if (PIOS_I2C_CheckClear(pios_i2c_internal_id) != 0)
        panic(8);
#endif

#if defined(PIOS_INCLUDE_SPI)
    if (PIOS_SPI_Init(&pios_spi_internal_id, &pios_spi_internal_cfg)) {
        PIOS_Assert(0);
    }
    #endif

#if defined(PIOS_INCLUDE_FLASH)
    /* Initialize all flash drivers */
    if (PIOS_Flash_Internal_Init(&pios_internal_flash_id, &flash_internal_cfg) != 0)
        panic(1);

    /* Register the partition table */
    const struct pios_flash_partition * flash_partition_table;
    uint32_t num_partitions;
    flash_partition_table = PIOS_BOARD_HW_DEFS_GetPartitionTable(bdinfo->board_rev, &num_partitions);
    PIOS_FLASH_register_partition_table(flash_partition_table, num_partitions);

    /* Mount all filesystems */
    if (PIOS_FLASHFS_Logfs_Init(&pios_uavo_settings_fs_id, &flashfs_settings_cfg, FLASH_PARTITION_LABEL_SETTINGS) != 0)
        panic(2);
    if (PIOS_FLASHFS_Logfs_Init(&pios_waypoints_settings_fs_id, &flashfs_waypoints_cfg, FLASH_PARTITION_LABEL_WAYPOINTS) != 0)
        panic(3);
#endif    /* PIOS_INCLUDE_FLASH */

    /* Initialize the task monitor library */
    TaskMonitorInitialize();

    /* Initialize UAVObject libraries */
    EventDispatcherInitialize();
    UAVObjInitialize();

    /* Initialize the alarms library */
    AlarmsInitialize();

    HwAQ32Initialize();
    ModuleSettingsInitialize();
    LoggingSettingsInitialize();

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
    
    // Timers used for inputs (4)
    PIOS_TIM_InitClock(&tim_4_cfg);
    
    // Timers used for outputs (2, 3, 8)
    PIOS_TIM_InitClock(&tim_2_cfg);
    PIOS_TIM_InitClock(&tim_3_cfg);
    PIOS_TIM_InitClock(&tim_8_cfg);
    
    // Timers used for inputs or outputs (1)
    // Configure TIM_Period (ARR) accordingly
    
    PIOS_TIM_InitClock(&tim_1_cfg);
    
    uint8_t hw_rcvrport;
    HwAQ32RcvrPortGet(&hw_rcvrport);

    switch (hw_rcvrport) {
    case HWAQ32_RCVRPORT_DISABLED:
    case HWAQ32_RCVRPORT_PPM:
        TIM1->ARR = ((1000000 / PIOS_SERVO_UPDATE_HZ) - 1);  // Timer 1 configured for PWM outputs
        break;
    case HWAQ32_RCVRPORT_PWM:
        TIM1->ARR = 0xFFFF;  // Timer 1 configured for PWM inputs
        break;
    }    

    /* IAP System Setup */
    PIOS_IAP_Init();
    uint16_t boot_count = PIOS_IAP_ReadBootCount();
    if (boot_count < 3) {
        PIOS_IAP_WriteBootCount(++boot_count);
        AlarmsClear(SYSTEMALARMS_ALARM_BOOTFAULT);
    } else {
        /* Too many failed boot attempts, force hw config to defaults */
        HwAQ32SetDefaults(HwAQ32Handle(), 0);
        ModuleSettingsSetDefaults(ModuleSettingsHandle(),0);
        AlarmsSet(SYSTEMALARMS_ALARM_BOOTFAULT, SYSTEMALARMS_ALARM_CRITICAL);
    }

#if defined(PIOS_INCLUDE_USB)
    /* Initialize USB disconnect GPIO */
    GPIO_Init(pios_usb_main_cfg.disconnect.gpio, (GPIO_InitTypeDef*)&pios_usb_main_cfg.disconnect.init);
    GPIO_SetBits(pios_usb_main_cfg.disconnect.gpio, pios_usb_main_cfg.disconnect.init.GPIO_Pin);

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
    HwAQ32USB_VCPPortGet(&hw_usb_vcpport);

    if (!usb_cdc_present) {
        /* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
        hw_usb_vcpport = HWAQ32_USB_VCPPORT_DISABLED;
    }

    PIOS_HAL_ConfigureCDC(hw_usb_vcpport, pios_usb_id, &pios_usb_cdc_cfg);

#endif    /* PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_USB_HID)
    /* Configure the usb HID port */
    uint8_t hw_usb_hidport;
    HwAQ32USB_HIDPortGet(&hw_usb_hidport);

    if (!usb_hid_present) {
        /* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
        hw_usb_hidport = HWAQ32_USB_HIDPORT_DISABLED;
    }

    PIOS_HAL_ConfigureHID(hw_usb_hidport, pios_usb_id, &pios_usb_hid_cfg);
    
#endif    /* PIOS_INCLUDE_USB_HID */

    if (usb_hid_present || usb_cdc_present) {
        PIOS_USBHOOK_Activate();
    }

    /* Issue USB Disconnect Pulse */
    PIOS_WDG_Clear();
    
    GPIO_ResetBits(pios_usb_main_cfg.disconnect.gpio, pios_usb_main_cfg.disconnect.init.GPIO_Pin);
        
    PIOS_DELAY_WaitmS(200);
        
    GPIO_SetBits(pios_usb_main_cfg.disconnect.gpio, pios_usb_main_cfg.disconnect.init.GPIO_Pin);
    
    PIOS_WDG_Clear();
#endif    /* PIOS_INCLUDE_USB */

    /* Configure the IO ports */
    HwAQ32DSMxModeOptions hw_DSMxMode;
    HwAQ32DSMxModeGet(&hw_DSMxMode);

    LoggingSettingsLogDestinationOptions log_destination;
    LoggingSettingsLogDestinationGet(&log_destination);
    
    /* UART1 Port */
    uint8_t hw_uart1;
    HwAQ32Uart1Get(&hw_uart1);

    PIOS_HAL_ConfigurePort(hw_uart1,             // port type protocol
            &pios_usart1_cfg,                    // usart_port_cfg
            &pios_usart_com_driver,              // com_driver
            NULL,                                // i2c_id
            NULL,                                // i2c_cfg
            NULL,                                // ppm_cfg
            NULL,                                // pwm_cfg
            PIOS_LED_ALARM,                      // led_id
            NULL,                                // usart_dsm_hsum_cfg
            NULL,                                // dsm_cfg
            0,                                   // dsm_mode
            NULL,                                // sbus_rcvr_cfg
            NULL,                                // sbus_cfg    
            false,                               // sbus_toggle
            log_destination);                     // log_dest

    /* UART2 Port */
    uint8_t hw_uart2;
    HwAQ32Uart2Get(&hw_uart2);

    PIOS_HAL_ConfigurePort(hw_uart2,             // port type protocol
            &pios_usart2_cfg,                    // usart_port_cfg
            &pios_usart_com_driver,              // com_driver
            NULL,                                // i2c_id
            NULL,                                // i2c_cfg
            NULL,                                // ppm_cfg
            NULL,                                // pwm_cfg
            PIOS_LED_ALARM,                      // led_id
            NULL,                                // usart_dsm_hsum_cfg
            NULL,                                // dsm_cfg
            0,                                   // dsm_mode
            NULL,                                // sbus_rcvr_cfg
            NULL,                                // sbus_cfg    
            false,                               // sbus_toggle
            log_destination);                     // log_dest

    /* UART3 Port */
    uint8_t hw_uart3;
    HwAQ32Uart3Get(&hw_uart3);

    PIOS_HAL_ConfigurePort(hw_uart3,             // port type protocol
            &pios_usart3_cfg,                    // usart_port_cfg
            &pios_usart_com_driver,              // com_driver
            NULL,                                // i2c_id 
            NULL,                                // i2c_cfg
            NULL,                                // ppm_cfg
            NULL,                                // pwm_cfg
            PIOS_LED_ALARM,                      // led_id
            &pios_usart3_dsm_hsum_cfg,           // usart_dsm_hsum_cfg
            NULL,                                // dsm_cfg
            0,                                   // dsm_mode
            &pios_usart3_sbus_cfg,               // sbus_rcvr_cfg
            &pios_usart3_sbus_aux_cfg,           // sbus_cfg                
            true,                                // sbus_toggle
            log_destination);                     // log_dest
            
    if (hw_uart3 == HWAQ32_UART3_FRSKYSENSORHUB)
    {
        GPIO_Init(pios_usart3_sbus_aux_cfg.inv.gpio, (GPIO_InitTypeDef*)&pios_usart3_sbus_aux_cfg.inv.init);
        GPIO_WriteBit(pios_usart3_sbus_aux_cfg.inv.gpio, pios_usart3_sbus_aux_cfg.inv.init.GPIO_Pin, pios_usart3_sbus_aux_cfg.gpio_inv_enable);
    }
           
    /* UART4 Port */
    uint8_t hw_uart4;
    HwAQ32Uart4Get(&hw_uart4);

    PIOS_HAL_ConfigurePort(hw_uart4,             // port type protocol
            &pios_usart4_cfg,                    // usart_port_cfg
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
            false,                               // sbus_toggle
            log_destination);                     // log_dest

    /* UART6 Port */
    uint8_t hw_uart6;
    HwAQ32Uart6Get(&hw_uart6);

    PIOS_HAL_ConfigurePort(hw_uart6,             // port type protocol
            &pios_usart6_cfg,                    // usart_port_cfg
            &pios_usart_com_driver,              // com_driver
            NULL,                                // i2c_id
            NULL,                                // i2c_cfg
            NULL,                                // ppm_cfg
            NULL,                                // pwm_cfg
            PIOS_LED_ALARM,                      // led_id
            &pios_usart6_dsm_hsum_cfg,           // usart_dsm_hsum_cfg
            &pios_usart6_dsm_aux_cfg,            // dsm_cfg
            hw_DSMxMode,                         // dsm_mode
            NULL,                                // sbus_rcvr_cfg
            NULL,                                // sbus_cfg    
            false,                               // sbus_toggle
            log_destination);                     // log_dest

    /* Configure the rcvr port */
    PIOS_HAL_ConfigurePort(hw_rcvrport,          // port type protocol
            NULL,                                // usart_port_cfg
            NULL,                                // com_driver
            NULL,                                // i2c_id
            NULL,                                // i2c_cfg
            &pios_ppm_cfg,                       // ppm_cfg
            &pios_pwm_cfg,                       // pwm_cfg
            PIOS_LED_ALARM,                      // led_id
            NULL,                                // usart_dsm_hsum_cfg
            NULL,                                // dsm_cfg
            0,                                   // dsm_mode
            NULL,                                // sbus_rcvr_cfg
            NULL,                                // sbus_cfg    
            false,                               // sbus_toggle
            log_destination);                     // log_dest

#if defined(PIOS_INCLUDE_GCSRCVR)
    GCSReceiverInitialize();
    uintptr_t pios_gcsrcvr_id;
    PIOS_GCSRCVR_Init(&pios_gcsrcvr_id);
    uintptr_t pios_gcsrcvr_rcvr_id;
    if (PIOS_RCVR_Init(&pios_gcsrcvr_rcvr_id, &pios_gcsrcvr_rcvr_driver, pios_gcsrcvr_id)) {
        PIOS_Assert(0);
    }
    pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_GCS] = pios_gcsrcvr_rcvr_id;
#endif    /* PIOS_INCLUDE_GCSRCVR */

#if defined(PIOS_INCLUDE_TIM) && defined(PIOS_INCLUDE_SERVO)
#ifndef PIOS_DEBUG_ENABLE_DEBUG_PINS
    switch (hw_rcvrport) {
    case HWAQ32_RCVRPORT_DISABLED:
    case HWAQ32_RCVRPORT_PPM:
        PIOS_Servo_Init(&pios_servo_cfg_ppm_rx);
        break;
    case HWAQ32_RCVRPORT_PWM:
        PIOS_Servo_Init(&pios_servo_cfg_pwm_rx);
        break;
    }
#else
    PIOS_DEBUG_Init(&pios_tim_servo_all_channels, NELEMENTS(pios_tim_servo_all_channels));
#endif
#endif

/* init sensor queue registration */
    PIOS_SENSORS_Init();

    PIOS_WDG_Clear();
    PIOS_DELAY_WaitmS(200);
    PIOS_WDG_Clear();

#if defined(PIOS_INCLUDE_MPU6000)
    if (PIOS_MPU6000_Init(pios_spi_internal_id, 0, &pios_mpu6000_cfg) != 0)
        panic(4);
    if (PIOS_MPU6000_Test() != 0)
        panic(5);

    // To be safe map from UAVO enum to driver enum
    uint8_t hw_gyro_range;
    HwAQ32GyroRangeGet(&hw_gyro_range);
    switch(hw_gyro_range) {
        case HWAQ32_GYRORANGE_250:
            PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_250_DEG);
            break;
        case HWAQ32_GYRORANGE_500:
            PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);
            break;
        case HWAQ32_GYRORANGE_1000:
            PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_1000_DEG);
            break;
        case HWAQ32_GYRORANGE_2000:
            PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_2000_DEG);
            break;
    }

    uint8_t hw_accel_range;
    HwAQ32AccelRangeGet(&hw_accel_range);
    switch(hw_accel_range) {
        case HWAQ32_ACCELRANGE_2G:
            PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_2G);
            break;
        case HWAQ32_ACCELRANGE_4G:
            PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_4G);
            break;
        case HWAQ32_ACCELRANGE_8G:
            PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
            break;
        case HWAQ32_ACCELRANGE_16G:
            PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_16G);
            break;
    }

    // the filter has to be set before rate else divisor calculation will fail
    uint8_t hw_mpu6000_dlpf;
    HwAQ32MPU6000DLPFGet(&hw_mpu6000_dlpf);
    enum pios_mpu60x0_filter mpu6000_dlpf = \
        (hw_mpu6000_dlpf == HWAQ32_MPU6000DLPF_256) ? PIOS_MPU60X0_LOWPASS_256_HZ : \
        (hw_mpu6000_dlpf == HWAQ32_MPU6000DLPF_188) ? PIOS_MPU60X0_LOWPASS_188_HZ : \
        (hw_mpu6000_dlpf == HWAQ32_MPU6000DLPF_98)  ? PIOS_MPU60X0_LOWPASS_98_HZ  : \
        (hw_mpu6000_dlpf == HWAQ32_MPU6000DLPF_42)  ? PIOS_MPU60X0_LOWPASS_42_HZ  : \
        (hw_mpu6000_dlpf == HWAQ32_MPU6000DLPF_20)  ? PIOS_MPU60X0_LOWPASS_20_HZ  : \
        (hw_mpu6000_dlpf == HWAQ32_MPU6000DLPF_10)  ? PIOS_MPU60X0_LOWPASS_10_HZ  : \
        (hw_mpu6000_dlpf == HWAQ32_MPU6000DLPF_5)   ? PIOS_MPU60X0_LOWPASS_5_HZ   : \
        pios_mpu6000_cfg.default_filter;
    PIOS_MPU6000_SetLPF(mpu6000_dlpf);

    uint8_t hw_mpu6000_samplerate;
    HwAQ32MPU6000RateGet(&hw_mpu6000_samplerate);
    uint16_t mpu6000_samplerate = \
        (hw_mpu6000_samplerate == HWAQ32_MPU6000RATE_200)  ?  200 : \
        (hw_mpu6000_samplerate == HWAQ32_MPU6000RATE_333)  ?  333 : \
        (hw_mpu6000_samplerate == HWAQ32_MPU6000RATE_500)  ?  500 : \
        (hw_mpu6000_samplerate == HWAQ32_MPU6000RATE_666)  ?  666 : \
        (hw_mpu6000_samplerate == HWAQ32_MPU6000RATE_1000) ? 1000 : \
        (hw_mpu6000_samplerate == HWAQ32_MPU6000RATE_2000) ? 2000 : \
        (hw_mpu6000_samplerate == HWAQ32_MPU6000RATE_4000) ? 4000 : \
        (hw_mpu6000_samplerate == HWAQ32_MPU6000RATE_8000) ? 8000 : \
        pios_mpu6000_cfg.default_samplerate;
    PIOS_MPU6000_SetSampleRate(mpu6000_samplerate);

#endif

#if defined(PIOS_INCLUDE_I2C)
#if defined(PIOS_INCLUDE_HMC5883)
    PIOS_WDG_Clear();

    uint8_t Magnetometer;
    HwAQ32MagnetometerGet(&Magnetometer);
    
    external_mag_fail = false;

    if (Magnetometer == HWAQ32_MAGNETOMETER_EXTERNAL)
    {
        AlarmsSet(SYSTEMALARMS_ALARM_I2C, SYSTEMALARMS_ALARM_OK);
            
        if (PIOS_I2C_Init(&pios_i2c_external_id, &pios_i2c_external_cfg)) {
            PIOS_DEBUG_Assert(0);
            AlarmsSet(SYSTEMALARMS_ALARM_I2C, SYSTEMALARMS_ALARM_CRITICAL);
        }
        
        if (PIOS_I2C_CheckClear(pios_i2c_external_id) != 0)
            AlarmsSet(SYSTEMALARMS_ALARM_I2C, SYSTEMALARMS_ALARM_CRITICAL);;
    
        if (PIOS_HMC5883_Init(pios_i2c_external_id, &pios_hmc5883_external_cfg) == 0) {
            if (PIOS_HMC5883_Test() == 0) {
                // External mag configuration was successful
                
                // setup sensor orientation
                uint8_t ExtMagOrientation;
                HwAQ32ExtMagOrientationGet(&ExtMagOrientation);

                enum pios_hmc5883_orientation hmc5883_externalOrientation = \
                    (ExtMagOrientation == HWAQ32_EXTMAGORIENTATION_TOP0DEGCW)      ? PIOS_HMC5883_TOP_0DEG      : \
                    (ExtMagOrientation == HWAQ32_EXTMAGORIENTATION_TOP90DEGCW)     ? PIOS_HMC5883_TOP_90DEG     : \
                    (ExtMagOrientation == HWAQ32_EXTMAGORIENTATION_TOP180DEGCW)    ? PIOS_HMC5883_TOP_180DEG    : \
                    (ExtMagOrientation == HWAQ32_EXTMAGORIENTATION_TOP270DEGCW)    ? PIOS_HMC5883_TOP_270DEG    : \
                    (ExtMagOrientation == HWAQ32_EXTMAGORIENTATION_BOTTOM0DEGCW)   ? PIOS_HMC5883_BOTTOM_0DEG   : \
                    (ExtMagOrientation == HWAQ32_EXTMAGORIENTATION_BOTTOM90DEGCW)  ? PIOS_HMC5883_BOTTOM_90DEG  : \
                    (ExtMagOrientation == HWAQ32_EXTMAGORIENTATION_BOTTOM180DEGCW) ? PIOS_HMC5883_BOTTOM_180DEG : \
                    (ExtMagOrientation == HWAQ32_EXTMAGORIENTATION_BOTTOM270DEGCW) ? PIOS_HMC5883_BOTTOM_270DEG : \
                    pios_hmc5883_external_cfg.Default_Orientation;
                PIOS_HMC5883_SetOrientation(hmc5883_externalOrientation);
            }
            else
                external_mag_fail = true;  // External HMC5883 Test Failed
        }
        else
            external_mag_fail = true;  // External HMC5883 Init Failed
    }

    if (Magnetometer == HWAQ32_MAGNETOMETER_INTERNAL)
    {
        if (PIOS_HMC5883_Init(pios_i2c_internal_id, &pios_hmc5883_internal_cfg) != 0)
            panic(6);
        if (PIOS_HMC5883_Test() != 0)
            panic(7);
    }

#endif

    //I2C is slow, sensor init as well, reset watchdog to prevent reset here
    PIOS_WDG_Clear();

#if defined(PIOS_INCLUDE_MS5611)
    if (PIOS_MS5611_Init(&pios_ms5611_cfg, pios_i2c_internal_id) != 0)
        panic(12);
    if (PIOS_MS5611_Test() != 0)
        panic(13);
#endif

    //I2C is slow, sensor init as well, reset watchdog to prevent reset here
    PIOS_WDG_Clear();

#endif    /* PIOS_INCLUDE_I2C */

#if defined(PIOS_INCLUDE_GPIO)
    PIOS_GPIO_Init();
#endif

#if defined(PIOS_INCLUDE_ADC)
    /* Configure the adc port(s) */
    uint8_t hw_adcport;

    HwAQ32ADCInputsGet(&hw_adcport);

    switch (hw_adcport)    {
    case HWAQ32_ADCINPUTS_DISABLED:
        break;
    case HWAQ32_ADCINPUTS_ENABLED:
        {
            uint32_t internal_adc_id;

            if (PIOS_INTERNAL_ADC_Init(&internal_adc_id, &pios_adc_cfg) < 0) {
                PIOS_Assert(0);
                    panic(14);
            }

            if (PIOS_ADC_Init(&pios_internal_adc_id, &pios_internal_adc_driver, internal_adc_id) < 0)
                panic(15);
        }
        break;
    }
#endif

    /* Make sure we have at least one telemetry link configured or else fail initialization */
    PIOS_Assert(pios_com_telem_serial_id || pios_com_telem_usb_id);
}

/**
 * @}
 */
