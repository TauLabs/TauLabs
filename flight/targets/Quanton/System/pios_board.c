/**
 *****************************************************************************
 * @file       pios_board.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://www.taulabs.org, Copyright (C) 2012-2013
 * @addtogroup OpenPilotSystem OpenPilot System
 * @{
 * @addtogroup OpenPilotCore OpenPilot Core
 * @{
 * @brief Defines board specific static initializers for hardware for the Quanton board.
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
#include "hwquanton.h"
#include "manualcontrolsettings.h"
#include "modulesettings.h"


#if defined(PIOS_INCLUDE_HMC5883)
#include "pios_hmc5883.h"
static const struct pios_exti_cfg pios_exti_hmc5883_cfg __exti_config = {
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
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_LOW,
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

static const struct pios_hmc5883_cfg pios_hmc5883_cfg = {
	.exti_cfg = &pios_exti_hmc5883_cfg,
	.M_ODR = PIOS_HMC5883_ODR_75,
	.Meas_Conf = PIOS_HMC5883_MEASCONF_NORMAL,
	.Gain = PIOS_HMC5883_GAIN_1_9,
	.Mode = PIOS_HMC5883_MODE_CONTINUOUS,
	.orientation = PIOS_HMC5883_TOP_90DEG,

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
			.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_MID,
			.NVIC_IRQChannelSubPriority = 0,
			.NVIC_IRQChannelCmd = ENABLE,
		},
	},
	.exti = {
		.init = {
			.EXTI_Line = EXTI_Line0, // matches above GPIO pin
			.EXTI_Mode = EXTI_Mode_Interrupt,
			.EXTI_Trigger = EXTI_Trigger_Rising,
			.EXTI_LineCmd = ENABLE,
		},
	},
};

static const struct pios_mpu60x0_cfg pios_mpu6000_cfg = {
	.exti_cfg = &pios_exti_mpu6000_cfg,
	.Fifo_store = PIOS_MPU60X0_FIFO_TEMP_OUT | PIOS_MPU60X0_FIFO_GYRO_X_OUT | PIOS_MPU60X0_FIFO_GYRO_Y_OUT | PIOS_MPU60X0_FIFO_GYRO_Z_OUT,
	// Clock at 8 khz, downsampled by 8 for 1khz
	.Smpl_rate_div = 11,
	.interrupt_cfg = PIOS_MPU60X0_INT_CLR_ANYRD,
	.interrupt_en = PIOS_MPU60X0_INTEN_DATA_RDY,
	.User_ctl = PIOS_MPU60X0_USERCTL_FIFO_EN | PIOS_MPU60X0_USERCTL_DIS_I2C,
	.Pwr_mgmt_clk = PIOS_MPU60X0_PWRMGMT_PLL_X_CLK,
	.filter = PIOS_MPU60X0_LOWPASS_256_HZ,
	.orientation = PIOS_MPU60X0_TOP_180DEG
};
#endif /* PIOS_INCLUDE_MPU6000 */

/* One slot per selectable receiver group.
 *  eg. PWM, PPM, GCS, SPEKTRUM1, SPEKTRUM2, SBUS
 * NOTE: No slot in this map for NONE.
 */
uint32_t pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE];

#define PIOS_COM_TELEM_RF_RX_BUF_LEN 512
#define PIOS_COM_TELEM_RF_TX_BUF_LEN 512

#define PIOS_COM_GPS_RX_BUF_LEN 32

#define PIOS_COM_TELEM_USB_RX_BUF_LEN 65
#define PIOS_COM_TELEM_USB_TX_BUF_LEN 65

#define PIOS_COM_BRIDGE_RX_BUF_LEN 65
#define PIOS_COM_BRIDGE_TX_BUF_LEN 12

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
#define PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN 40
uintptr_t pios_com_debug_id;
#endif /* PIOS_INCLUDE_DEBUG_CONSOLE */

uintptr_t pios_com_aux_id = 0;
uintptr_t pios_com_gps_id = 0;
uintptr_t pios_com_telem_usb_id = 0;
uintptr_t pios_com_telem_rf_id = 0;
uintptr_t pios_com_bridge_id = 0;
uintptr_t pios_com_overo_id = 0;

/*
 * Setup a com port based on the passed cfg, driver and buffer sizes. tx size of -1 make the port rx only
 */
#ifdef PIOS_INCLUDE_USART
static void PIOS_Board_configure_com(const struct pios_usart_cfg *usart_port_cfg, size_t rx_buf_len, size_t tx_buf_len,
		const struct pios_com_driver *com_driver, uintptr_t *pios_com_id)
{
	uint32_t pios_usart_id;
	if (PIOS_USART_Init(&pios_usart_id, usart_port_cfg)) {
		PIOS_Assert(0);
	}

	uint8_t * rx_buffer = (uint8_t *) pvPortMalloc(rx_buf_len);
	PIOS_Assert(rx_buffer);
	if(tx_buf_len!= -1){ // this is the case for rx/tx ports
		uint8_t * tx_buffer = (uint8_t *) pvPortMalloc(tx_buf_len);
		PIOS_Assert(tx_buffer);

		if (PIOS_COM_Init(pios_com_id, com_driver, pios_usart_id,
				rx_buffer, rx_buf_len,
				tx_buffer, tx_buf_len)) {
			PIOS_Assert(0);
		}
	}
	else{ //rx only port
		if (PIOS_COM_Init(pios_com_id, com_driver, pios_usart_id,
				rx_buffer, rx_buf_len,
				NULL, 0)) {
			PIOS_Assert(0);
		}
	}
}
#endif

#ifdef PIOS_INCLUDE_DSM
static void PIOS_Board_configure_dsm(const struct pios_usart_cfg *pios_usart_dsm_cfg, const struct pios_dsm_cfg *pios_dsm_cfg,
		const struct pios_com_driver *pios_usart_com_driver,enum pios_dsm_proto *proto,
		ManualControlSettingsChannelGroupsOptions channelgroup,uint8_t *bind)
{
	uint32_t pios_usart_dsm_id;
	if (PIOS_USART_Init(&pios_usart_dsm_id, pios_usart_dsm_cfg)) {
		PIOS_Assert(0);
	}

	uint32_t pios_dsm_id;
	if (PIOS_DSM_Init(&pios_dsm_id, pios_dsm_cfg, pios_usart_com_driver,
			pios_usart_dsm_id, *proto, *bind)) {
		PIOS_Assert(0);
	}

	uint32_t pios_dsm_rcvr_id;
	if (PIOS_RCVR_Init(&pios_dsm_rcvr_id, &pios_dsm_rcvr_driver, pios_dsm_id)) {
		PIOS_Assert(0);
	}
	pios_rcvr_group_map[channelgroup] = pios_dsm_rcvr_id;
}
#endif

void panic() {
	while(1){
		PIOS_WDG_Clear();
		PIOS_LED_Toggle(PIOS_LED_ALARM);
		PIOS_DELAY_WaitmS(200);
	}
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

#if defined(PIOS_INCLUDE_FLASH)
	if (PIOS_SPI_Init(&pios_spi_flash_id, &pios_spi_flash_cfg)) {
		PIOS_DEBUG_Assert(0);
	}

	uintptr_t flash_id;
	if (PIOS_Flash_Jedec_Init(&flash_id, pios_spi_flash_id, 0, &flash_mx25_cfg) != 0)
		panic();
	uintptr_t fs_id;
	if (PIOS_FLASHFS_Logfs_Init(&fs_id, &flashfs_mx25_cfg, &pios_jedec_flash_driver, flash_id) != 0)
		panic();
#endif
	
	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();

	HwQuantonInitialize();
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
		HwQuantonSetDefaults(HwQuantonHandle(), 0);
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

	uint32_t pios_usb_id;
	PIOS_USB_Init(&pios_usb_id, PIOS_BOARD_HW_DEFS_GetUsbCfg(bdinfo->board_rev));

#if defined(PIOS_INCLUDE_USB_CDC)

	uint8_t hw_usb_vcpport;
	/* Configure the USB VCP port */
	HwQuantonUSB_VCPPortGet(&hw_usb_vcpport);

	if (!usb_cdc_present) {
		/* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
		hw_usb_vcpport = HWQUANTON_USB_VCPPORT_DISABLED;
	}

	switch (hw_usb_vcpport) {
	case HWQUANTON_USB_VCPPORT_DISABLED:
		break;
	case HWQUANTON_USB_VCPPORT_USBTELEMETRY:
#if defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usb_cdc_cfg, PIOS_COM_TELEM_USB_RX_BUF_LEN, PIOS_COM_TELEM_USB_TX_BUF_LEN, &pios_usb_cdc_com_driver, &pios_com_telem_usb_id);
#endif	/* PIOS_INCLUDE_COM */
		break;
	case HWQUANTON_USB_VCPPORT_COMBRIDGE:
#if defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usb_cdc_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usb_cdc_com_driver, &pios_com_vcp_id);
#endif	/* PIOS_INCLUDE_COM */
		break;
	case HWQUANTON_USB_VCPPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_COM)
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
		{
			uint32_t pios_usb_cdc_id;
			if (PIOS_USB_CDC_Init(&pios_usb_cdc_id, &pios_usb_cdc_cfg, pios_usb_id)) {
				PIOS_Assert(0);
			}
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
	}
#endif	/* PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_USB_HID)
	/* Configure the usb HID port */
	uint8_t hw_usb_hidport;
	HwQuantonUSB_HIDPortGet(&hw_usb_hidport);

	if (!usb_hid_present) {
		/* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
		hw_usb_hidport = HWQUANTON_USB_HIDPORT_DISABLED;
	}

	switch (hw_usb_hidport) {
	case HWQUANTON_USB_HIDPORT_DISABLED:
		break;
	case HWQUANTON_USB_HIDPORT_USBTELEMETRY:
#if defined(PIOS_INCLUDE_COM)
		{
			uint32_t pios_usb_hid_id;
			if (PIOS_USB_HID_Init(&pios_usb_hid_id, &pios_usb_hid_cfg, pios_usb_id)) {
				PIOS_Assert(0);
			}
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

	/* Configure the IO ports */
	uint8_t hw_DSMxBind;
	HwQuantonDSMxBindGet(&hw_DSMxBind);

	/* UART1 Port */
	uint8_t hw_uart1;
	HwQuantonUart1Get(&hw_uart1);
	switch (hw_uart1) {
	case HWQUANTON_UART1_DISABLED:
		break;
	case HWQUANTON_UART1_TELEMETRY:
#if defined(PIOS_INCLUDE_TELEMETRY_RF) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart1_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
#endif /* PIOS_INCLUDE_TELEMETRY_RF */
		break;
	case HWQUANTON_UART1_GPS:
#if defined(PIOS_INCLUDE_GPS) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart1_cfg, PIOS_COM_GPS_RX_BUF_LEN, -1, &pios_usart_com_driver, &pios_com_gps_id);
#endif
		break;
	case HWQUANTON_UART1_I2C:
#if defined(PIOS_INCLUDE_I2C)
		if (PIOS_I2C_Init(&pios_i2c_usart1_adapter_id, &pios_i2c_usart1_adapter_cfg)) {
			PIOS_Assert(0);
		}
#endif	/* PIOS_INCLUDE_I2C */
		break;
	case HWQUANTON_UART1_DSM2:
	case HWQUANTON_UART1_DSMX10BIT:
	case HWQUANTON_UART1_DSMX11BIT:
#if defined(PIOS_INCLUDE_DSM)
		{
			enum pios_dsm_proto proto;
			switch (hw_uart1) {
			case HWQUANTON_UART1_DSM2:
				proto = PIOS_DSM_PROTO_DSM2;
				break;
			case HWQUANTON_UART1_DSMX10BIT:
				proto = PIOS_DSM_PROTO_DSMX10BIT;
				break;
			case HWQUANTON_UART1_DSMX11BIT:
				proto = PIOS_DSM_PROTO_DSMX11BIT;
				break;
			default:
				PIOS_Assert(0);
				break;
			}
			PIOS_Board_configure_dsm(&pios_usart1_dsm_cfg, &pios_usart1_dsm_aux_cfg, &pios_usart_com_driver,
				&proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT, &hw_DSMxBind);
		}
#endif	/* PIOS_INCLUDE_DSM */
		break;
	case HWQUANTON_UART1_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart_1_cfg, 0, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_aux_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
		break;
	case HWQUANTON_UART1_COMBRIDGE:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart1_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
#endif
		break;
	}

	/* UART2 Port */
	uint8_t hw_uart2;
	HwQuantonUart2Get(&hw_uart2);
	switch (hw_uart2) {
	case HWQUANTON_UART2_DISABLED:
		break;
	case HWQUANTON_UART2_TELEMETRY:
#if defined(PIOS_INCLUDE_TELEMETRY_RF) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart2_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
#endif /* PIOS_INCLUDE_TELEMETRY_RF */
		break;
	case HWQUANTON_UART2_GPS:
#if defined(PIOS_INCLUDE_GPS) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart2_cfg, PIOS_COM_GPS_RX_BUF_LEN, -1, &pios_usart_com_driver, &pios_com_gps_id);
#endif
		break;
	case HWQUANTON_UART2_SBUS:
		//hardware signal inverter required
#if defined(PIOS_INCLUDE_SBUS) && defined(PIOS_INCLUDE_USART)
		{
			uint32_t pios_usart_sbus_id;
			if (PIOS_USART_Init(&pios_usart_sbus_id, &pios_usart2_sbus_cfg)) {
				PIOS_Assert(0);
			}
			uint32_t pios_sbus_id;
			if (PIOS_SBus_Init(&pios_sbus_id, &pios_usart2_sbus_aux_cfg, &pios_usart_com_driver, pios_usart_sbus_id)) {
				PIOS_Assert(0);
			}
			uint32_t pios_sbus_rcvr_id;
			if (PIOS_RCVR_Init(&pios_sbus_rcvr_id, &pios_sbus_rcvr_driver, pios_sbus_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_SBUS] = pios_sbus_rcvr_id;
		}
#endif	/* PIOS_INCLUDE_SBUS */
		break;
	case HWQUANTON_UART2_DSM2:
	case HWQUANTON_UART2_DSMX10BIT:
	case HWQUANTON_UART2_DSMX11BIT:
#if defined(PIOS_INCLUDE_DSM)
		{
			enum pios_dsm_proto proto;
			switch (hw_uart2) {
			case HWQUANTON_UART2_DSM2:
				proto = PIOS_DSM_PROTO_DSM2;
				break;
			case HWQUANTON_UART2_DSMX10BIT:
				proto = PIOS_DSM_PROTO_DSMX10BIT;
				break;
			case HWQUANTON_UART2_DSMX11BIT:
				proto = PIOS_DSM_PROTO_DSMX11BIT;
				break;
			default:
				PIOS_Assert(0);
				break;
			}
			PIOS_Board_configure_dsm(&pios_usart2_dsm_cfg, &pios_usart2_dsm_aux_cfg, &pios_usart_com_driver,
				&proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT, &hw_DSMxBind);
		}
#endif	/* PIOS_INCLUDE_DSM */
		break;
	case HWQUANTON_UART2_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart_2_cfg, 0, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_aux_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
		break;
	case HWQUANTON_UART2_COMBRIDGE:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart2_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
#endif
		break;
	}

	/* UART3 Port */
	uint8_t hw_uart3;
	HwQuantonUart3Get(&hw_uart3);
	switch (hw_uart3) {
	case HWQUANTON_UART3_DISABLED:
		break;
	case HWQUANTON_UART3_TELEMETRY:
#if defined(PIOS_INCLUDE_TELEMETRY_RF) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart3_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
#endif /* PIOS_INCLUDE_TELEMETRY_RF */
		break;
	case HWQUANTON_UART3_GPS:
#if defined(PIOS_INCLUDE_GPS) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart3_cfg, PIOS_COM_GPS_RX_BUF_LEN, -1, &pios_usart_com_driver, &pios_com_gps_id);
#endif
		break;
	case HWQUANTON_UART3_I2C:
#if defined(PIOS_INCLUDE_I2C)
		if (PIOS_I2C_Init(&pios_i2c_usart3_adapter_id, &pios_i2c_usart3_adapter_cfg)) {
			PIOS_Assert(0);
		}
#endif	/* PIOS_INCLUDE_I2C */
		break;
	case HWQUANTON_UART3_DSM2:
	case HWQUANTON_UART3_DSMX10BIT:
	case HWQUANTON_UART3_DSMX11BIT:
#if defined(PIOS_INCLUDE_DSM)
		{
			enum pios_dsm_proto proto;
			switch (hw_uart3) {
			case HWQUANTON_UART3_DSM2:
				proto = PIOS_DSM_PROTO_DSM2;
				break;
			case HWQUANTON_UART3_DSMX10BIT:
				proto = PIOS_DSM_PROTO_DSMX10BIT;
				break;
			case HWQUANTON_UART3_DSMX11BIT:
				proto = PIOS_DSM_PROTO_DSMX11BIT;
				break;
			default:
				PIOS_Assert(0);
				break;
			}
			PIOS_Board_configure_dsm(&pios_usart3_dsm_cfg, &pios_usart3_dsm_aux_cfg, &pios_usart_com_driver,
				&proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT, &hw_DSMxBind);
		}
#endif	/* PIOS_INCLUDE_DSM */
		break;
	case HWQUANTON_UART3_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart_3_cfg, 0, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_aux_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
		break;
	case HWQUANTON_UART3_COMBRIDGE:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart3_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
#endif
		break;
	}

	/* UART4 Port */
	uint8_t hw_uart4;
	HwQuantonUart4Get(&hw_uart4);
	switch (hw_uart4) {
	case HWQUANTON_UART4_DISABLED:
		break;
	case HWQUANTON_UART4_TELEMETRY:
#if defined(PIOS_INCLUDE_TELEMETRY_RF) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart4_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
#endif /* PIOS_INCLUDE_TELEMETRY_RF */
		break;
	case HWQUANTON_UART4_GPS:
#if defined(PIOS_INCLUDE_GPS) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart4_cfg, PIOS_COM_GPS_RX_BUF_LEN, -1, &pios_usart_com_driver, &pios_com_gps_id);
#endif
		break;
	case HWQUANTON_UART4_DSM2:
	case HWQUANTON_UART4_DSMX10BIT:
	case HWQUANTON_UART4_DSMX11BIT:
#if defined(PIOS_INCLUDE_DSM)
		{
			enum pios_dsm_proto proto;
			switch (hw_uart4) {
			case HWQUANTON_UART4_DSM2:
				proto = PIOS_DSM_PROTO_DSM2;
				break;
			case HWQUANTON_UART4_DSMX10BIT:
				proto = PIOS_DSM_PROTO_DSMX10BIT;
				break;
			case HWQUANTON_UART4_DSMX11BIT:
				proto = PIOS_DSM_PROTO_DSMX11BIT;
				break;
			default:
				PIOS_Assert(0);
				break;
			}
			PIOS_Board_configure_dsm(&pios_usart4_dsm_cfg, &pios_usart4_dsm_aux_cfg, &pios_usart_com_driver,
				&proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT, &hw_DSMxBind);
		}
#endif	/* PIOS_INCLUDE_DSM */
		break;
	case HWQUANTON_UART4_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart_4_cfg, 0, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_aux_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
		break;
	case HWQUANTON_UART4_COMBRIDGE:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart4_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
#endif
		break;
	}

	/* UART5 Port */
	uint8_t hw_uart5;
	HwQuantonUart5Get(&hw_uart5);
	switch (hw_uart5) {
	case HWQUANTON_UART5_DISABLED:
		break;
	case HWQUANTON_UART5_TELEMETRY:
#if defined(PIOS_INCLUDE_TELEMETRY_RF) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart5_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
#endif /* PIOS_INCLUDE_TELEMETRY_RF */
		break;
	case HWQUANTON_UART5_GPS:
#if defined(PIOS_INCLUDE_GPS) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart5_cfg, PIOS_COM_GPS_RX_BUF_LEN, -1, &pios_usart_com_driver, &pios_com_gps_id);
#endif
		break;
	case HWQUANTON_UART5_DSM2:
	case HWQUANTON_UART5_DSMX10BIT:
	case HWQUANTON_UART5_DSMX11BIT:
#if defined(PIOS_INCLUDE_DSM)
		{
			enum pios_dsm_proto proto;
			switch (hw_uart5) {
			case HWQUANTON_UART5_DSM2:
				proto = PIOS_DSM_PROTO_DSM2;
				break;
			case HWQUANTON_UART5_DSMX10BIT:
				proto = PIOS_DSM_PROTO_DSMX10BIT;
				break;
			case HWQUANTON_UART5_DSMX11BIT:
				proto = PIOS_DSM_PROTO_DSMX11BIT;
				break;
			default:
				PIOS_Assert(0);
				break;
			}
			PIOS_Board_configure_dsm(&pios_usart5_dsm_cfg, &pios_usart5_dsm_aux_cfg, &pios_usart_com_driver,
				&proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT, &hw_DSMxBind);
		}
#endif	/* PIOS_INCLUDE_DSM */
		break;
	case HWQUANTON_UART5_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart_5_cfg, 0, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_aux_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
		break;
	case HWQUANTON_UART5_COMBRIDGE:
#if defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart5_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
#endif
		break;
	}

	/* Configure the rcvr port */
	uint8_t hw_rcvrport;
	HwQuantonRcvrPortGet(&hw_rcvrport);

	switch (hw_rcvrport) {
	case HWQUANTON_RCVRPORT_DISABLED:
		break;
	case HWQUANTON_RCVRPORT_PWM:
#if defined(PIOS_INCLUDE_PWM)
		{
			uint32_t pios_pwm_id;
			PIOS_PWM_Init(&pios_pwm_id, &pios_pwm_cfg);

			uint32_t pios_pwm_rcvr_id;
			if (PIOS_RCVR_Init(&pios_pwm_rcvr_id, &pios_pwm_rcvr_driver, pios_pwm_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PWM] = pios_pwm_rcvr_id;
		}
#endif	/* PIOS_INCLUDE_PWM */
		break;
	case HWQUANTON_RCVRPORT_PPM:
	case HWQUANTON_RCVRPORT_PPMOUTPUTS:
#if defined(PIOS_INCLUDE_PPM)
		{
			uint32_t pios_ppm_id;
			PIOS_PPM_Init(&pios_ppm_id, &pios_ppm_cfg);

			uint32_t pios_ppm_rcvr_id;
			if (PIOS_RCVR_Init(&pios_ppm_rcvr_id, &pios_ppm_rcvr_driver, pios_ppm_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PPM] = pios_ppm_rcvr_id;
		}
#endif	/* PIOS_INCLUDE_PPM */
		break;
	case HWQUANTON_RCVRPORT_PPMPWM:
		/* This is a combination of PPM and PWM inputs */
#if defined(PIOS_INCLUDE_PPM)
		{
			uint32_t pios_ppm_id;
			PIOS_PPM_Init(&pios_ppm_id, &pios_ppm_cfg);

			uint32_t pios_ppm_rcvr_id;
			if (PIOS_RCVR_Init(&pios_ppm_rcvr_id, &pios_ppm_rcvr_driver, pios_ppm_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PPM] = pios_ppm_rcvr_id;
		}
#endif	/* PIOS_INCLUDE_PPM */
#if defined(PIOS_INCLUDE_PWM)
		{
			uint32_t pios_pwm_id;
			PIOS_PWM_Init(&pios_pwm_id, &pios_pwm_with_ppm_cfg);

			uint32_t pios_pwm_rcvr_id;
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
	uint32_t pios_gcsrcvr_id;
	PIOS_GCSRCVR_Init(&pios_gcsrcvr_id);
	uint32_t pios_gcsrcvr_rcvr_id;
	if (PIOS_RCVR_Init(&pios_gcsrcvr_rcvr_id, &pios_gcsrcvr_rcvr_driver, pios_gcsrcvr_id)) {
		PIOS_Assert(0);
	}
	pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_GCS] = pios_gcsrcvr_rcvr_id;
#endif	/* PIOS_INCLUDE_GCSRCVR */

#ifndef PIOS_DEBUG_ENABLE_DEBUG_PINS
	switch (hw_rcvrport) {
		case HWQUANTON_RCVRPORT_DISABLED:
		case HWQUANTON_RCVRPORT_PWM:
		case HWQUANTON_RCVRPORT_PPM:
			/* Set up the servo outputs */
#ifdef PIOS_INCLUDE_SERVO
			PIOS_Servo_Init(&pios_servo_cfg);
#endif
			break;
		case HWQUANTON_RCVRPORT_PPMOUTPUTS:
		case HWQUANTON_RCVRPORT_OUTPUTS:
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

#if defined(PIOS_INCLUDE_MPU6000)
	if (PIOS_SPI_Init(&pios_spi_gyro_accel_id, &pios_spi_gyro_accel_cfg)) {
		PIOS_Assert(0);
	}

	if (PIOS_MPU6000_Init(pios_spi_gyro_accel_id, 0, &pios_mpu6000_cfg) != 0)
		panic();
	if (PIOS_MPU6000_Test() != 0)
		panic();

	// To be safe map from UAVO enum to driver enum
	uint8_t hw_gyro_range;
	HwQuantonGyroRangeGet(&hw_gyro_range);
	switch(hw_gyro_range) {
		case HWQUANTON_GYRORANGE_250:
			PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_250_DEG);
			break;
		case HWQUANTON_GYRORANGE_500:
			PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);
			break;
		case HWQUANTON_GYRORANGE_1000:
			PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_1000_DEG);
			break;
		case HWQUANTON_GYRORANGE_2000:
			PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_2000_DEG);
			break;
	}

	uint8_t hw_accel_range;
	HwQuantonAccelRangeGet(&hw_accel_range);
	switch(hw_accel_range) {
		case HWQUANTON_ACCELRANGE_2G:
			PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_2G);
			break;
		case HWQUANTON_ACCELRANGE_4G:
			PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_4G);
			break;
		case HWQUANTON_ACCELRANGE_8G:
			PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
			break;
		case HWQUANTON_ACCELRANGE_16G:
			PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_16G);
			break;
	}
#endif

#if defined(PIOS_INCLUDE_I2C)
	if (PIOS_I2C_Init(&pios_i2c_internal_adapter_id, &pios_i2c_internal_adapter_cfg)) {
		PIOS_DEBUG_Assert(0);
	}

	PIOS_WDG_Clear();
	PIOS_DELAY_WaitmS(50);
	PIOS_WDG_Clear();

#if defined(PIOS_INCLUDE_HMC5883)
	PIOS_HMC5883_Init(PIOS_I2C_MAIN_ADAPTER, &pios_hmc5883_cfg);
	if (PIOS_HMC5883_Test() != 0)
		panic();
#endif

	//I2C is slow, sensor init as well, reset watchdog to prevent reset here
	PIOS_WDG_Clear();

#if defined(PIOS_INCLUDE_MS5611)
	PIOS_MS5611_Init(&pios_ms5611_cfg, pios_i2c_internal_adapter_id);
	if (PIOS_MS5611_Test() != 0)
		panic();
#endif

	//I2C is slow, sensor init as well, reset watchdog to prevent reset here
	PIOS_WDG_Clear();

#endif	/* PIOS_INCLUDE_I2C */


#if defined(PIOS_INCLUDE_GPIO)
	PIOS_GPIO_Init();
#endif

#if defined(PIOS_INCLUDE_ADC)
	PIOS_ADC_Init(&pios_adc_cfg);
#endif

	//Set adc input to floating as long as it is unused
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
	PIOS_Assert(pios_com_telem_rf_id || pios_com_telem_usb_id);
}

/**
 * @}
 */
