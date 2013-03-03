/**
 *****************************************************************************
 * @file       pios_board.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://www.taulabs.org, Copyright (C) 2012-2013
 * @addtogroup OpenPilotSystem OpenPilot System
 * @{
 * @addtogroup OpenPilotCore OpenPilot Core
 * @{
 * @brief Defines board specific static initializers for hardware for the revolution board.
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
#include "hwvbrain.h"
#include "modulesettings.h"
#include "manualcontrolsettings.h"

/**
 * Sensor configurations 
 */

#if defined(PIOS_INCLUDE_ADC)
#include "pios_adc_priv.h"
void PIOS_ADC_DMC_irq_handler(void);
void DMA2_Stream4_IRQHandler(void) __attribute__((alias("PIOS_ADC_DMC_irq_handler")));
struct pios_adc_cfg pios_adc_cfg = {
	.adc_dev = ADC1,
	.dma = {
		.irq = {
			.flags   = (DMA_FLAG_TCIF4 | DMA_FLAG_TEIF4 | DMA_FLAG_HTIF4),
			.init    = {
				.NVIC_IRQChannel                   = DMA2_Stream4_IRQn,
				.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_LOW,
				.NVIC_IRQChannelSubPriority        = 0,
				.NVIC_IRQChannelCmd                = ENABLE,
			},
		},
		.rx = {
			.channel = DMA2_Stream4,
			.init    = {
				.DMA_Channel                    = DMA_Channel_0,
				.DMA_PeripheralBaseAddr = (uint32_t) & ADC1->DR
			},
		}
	},
	.half_flag = DMA_IT_HTIF4,
	.full_flag = DMA_IT_TCIF4,

};
void PIOS_ADC_DMC_irq_handler(void)
{
	/* Call into the generic code to handle the IRQ for this specific device */
	PIOS_ADC_DMA_Handler();
}

#endif

#if defined(PIOS_INCLUDE_HMC5883)
#include "pios_hmc5883.h"
static const struct pios_exti_cfg pios_exti_hmc5883_cfg __exti_config = {
		.vector = PIOS_HMC5883_IRQHandler,
		.line = EXTI_Line12,
		.pin = {
			.gpio = GPIOB,
			.init = {
				.GPIO_Pin = GPIO_Pin_12,
				.GPIO_Speed = GPIO_Speed_100MHz,
				.GPIO_Mode = GPIO_Mode_IN,
				.GPIO_OType = GPIO_OType_OD,
				.GPIO_PuPd = GPIO_PuPd_NOPULL,
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
				.EXTI_Line = EXTI_Line12,
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
	.orientation = PIOS_HMC5883_TOP_270DEG,
};
#endif /* PIOS_INCLUDE_HMC5883 */

/**
 * Configuration for the MS5611 chip
 */

#if defined(PIOS_INCLUDE_MS5611_SPI)
#include "pios_ms5611_SPI_priv.h"
static const struct pios_ms5611_cfg pios_ms5611_cfg = {
	.oversampling = MS5611_OSR_512,
	.temperature_interleaving = 1,
};
#endif /* PIOS_INCLUDE_MS5611_SPI */

/**
 * Configuration for the MPU6000 chip
 */
#if defined(PIOS_INCLUDE_MPU6000)
#include "pios_mpu6000.h"
static const struct pios_exti_cfg pios_exti_mpu6000_cfg __exti_config = {
		.vector = PIOS_MPU6000_IRQHandler,
		.line = EXTI_Line10,
		.pin = {
			.gpio = GPIOD,
			.init = {
				.GPIO_Pin = GPIO_Pin_10,
				.GPIO_Speed = GPIO_Speed_100MHz,
				.GPIO_Mode = GPIO_Mode_IN,
				.GPIO_OType = GPIO_OType_OD,
				.GPIO_PuPd = GPIO_PuPd_NOPULL,
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
				.EXTI_Line = EXTI_Line10,
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

#define PIOS_COM_AUX_RX_BUF_LEN 512
#define PIOS_COM_AUX_TX_BUF_LEN 512

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
#define PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN 40
uintptr_t pios_com_debug_id; //cbac
#endif /* PIOS_INCLUDE_DEBUG_CONSOLE */

uintptr_t pios_com_aux_id = 0;
uintptr_t pios_com_gps_id = 0;
uintptr_t pios_com_telem_usb_id = 0;
uintptr_t pios_com_telem_rf_id = 0;
uintptr_t pios_com_bridge_id = 0;
uintptr_t pios_com_overo_id = 0;

static const struct flashfs_cfg flashfs_at45dbx_cfg = {
	.table_magic = 0x99abceef,
	.obj_magic = 0x3015A371,
	.obj_table_start = 0x00000001,
	.obj_table_end = 0x00000200,
	.sector_size = 0x0000001,
	.chip_size = 0x0001000,
};

static const struct pios_flash_jedec_cfg flash_at45dbx_cfg = {
	.sector_erase = 0x81,
	.chip_erase = 0x50
};

/* 
 * Setup a com port based on the passed cfg, driver and buffer sizes. tx size of -1 make the port rx only
 */
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

/**
 * PIOS_Board_Init()
 * initializes all the core subsystems on this specific hardware
 * called from System/openpilot.c
 */

#include <pios_board_info.h>

void PIOS_Board_Init(void) {

//	const struct pios_board_info * bdinfo = &pios_board_info_blob;
	
	/* Delay system */
	PIOS_DELAY_Init();

	PIOS_LED_Init(&pios_led_cfg);

	/* Set up the SPI interface to the baro_flash*/
	if (PIOS_SPI_Init(&pios_spi_baro_flash_id, &pios_spi_baro_flash_cfg)) {
		PIOS_DEBUG_Assert(0);
	}
	
	/* Set up the SPI interface to the gyro */
	if (PIOS_SPI_Init(&pios_spi_gyro_id, &pios_spi_gyro_cfg)) {
		PIOS_DEBUG_Assert(0);
	}

	PIOS_Flash_Jedec_Init(pios_spi_baro_flash_id, 1, &flash_at45dbx_cfg);
	PIOS_FLASHFS_Init(&flashfs_at45dbx_cfg);

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();
	
	HwVbrainInitialize();
	ModuleSettingsInitialize();
	
#if defined(PIOS_INCLUDE_RTC)
	PIOS_RTC_Init(&pios_rtc_main_cfg);
#endif

	/* Initialize the alarms library */
	AlarmsInitialize();

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Set up pulse timers */
	PIOS_TIM_InitClock(&tim_1_cfg); // RC_in ch 1,2,3,4
	PIOS_TIM_InitClock(&tim_2_cfg); // RC_out ch 1,2,3
	PIOS_TIM_InitClock(&tim_3_cfg); // RC_out ch 4,5,6
	PIOS_TIM_InitClock(&tim_8_cfg); // RC_in ch 5,6,7,8
	PIOS_TIM_InitClock(&tim_9_cfg); // RC_out 9
	PIOS_TIM_InitClock(&tim_10_cfg); // RC_out 7
	PIOS_TIM_InitClock(&tim_11_cfg); // RC_out 8

	/* IAP System Setup */
	PIOS_IAP_Init();
	uint16_t boot_count = PIOS_IAP_ReadBootCount();
	if (boot_count < 3) {
		PIOS_IAP_WriteBootCount(++boot_count);
		AlarmsClear(SYSTEMALARMS_ALARM_BOOTFAULT);
	} else {
		/* Too many failed boot attempts, force hw config to defaults */
		HwVbrainSetDefaults(HwVbrainHandle(), 0);
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

	uint32_t pios_usb_id;
	PIOS_USB_Init(&pios_usb_id, &pios_usb_main_cfg);

#if defined(PIOS_INCLUDE_USB_CDC)

	uint8_t hw_usb_vcpport;
	/* Configure the USB VCP port */
	HwVbrainUSB_VCPPortGet(&hw_usb_vcpport);

	if (!usb_cdc_present) {
		/* Force VCP port function to disabled if we haven't advertised VCP in our USB descriptor */
		hw_usb_vcpport = HWVBRAIN_USB_VCPPORT_DISABLED;
	}

	switch (hw_usb_vcpport) {
	case HWVBRAIN_USB_VCPPORT_DISABLED:
		break;
	case HWVBRAIN_USB_VCPPORT_USBTELEMETRY:
#if defined(PIOS_INCLUDE_COM)
			PIOS_Board_configure_com(&pios_usb_cdc_cfg, PIOS_COM_TELEM_USB_RX_BUF_LEN, PIOS_COM_TELEM_USB_TX_BUF_LEN, &pios_usb_cdc_com_driver, &pios_com_telem_usb_id);
#endif	/* PIOS_INCLUDE_COM */
		break;
	case HWVBRAIN_USB_VCPPORT_COMBRIDGE:
#if defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usb_cdc_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usb_cdc_com_driver, &pios_com_vcp_id);
#endif	/* PIOS_INCLUDE_COM */
		break;
	}
#endif	/* PIOS_INCLUDE_USB_CDC */

#if defined(PIOS_INCLUDE_USB_HID)
	/* Configure the usb HID port */
	uint8_t hw_usb_hidport;
	HwVbrainUSB_HIDPortGet(&hw_usb_hidport);

	if (!usb_hid_present) {
		/* Force HID port function to disabled if we haven't advertised HID in our USB descriptor */
		hw_usb_hidport = HWVBRAIN_USB_HIDPORT_DISABLED;
	}

	switch (hw_usb_hidport) {
	case HWVBRAIN_USB_HIDPORT_DISABLED:
		break;
	case HWVBRAIN_USB_HIDPORT_USBTELEMETRY:
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

	/* Configure IO ports */
	uint8_t hw_DSMxBind;
	HwVbrainDSMxBindGet(&hw_DSMxBind);
	
	/* Configure Telemetry port */
	uint8_t hw_telemetryport;
	HwVbrainTelemetryPortGet(&hw_telemetryport);

	switch (hw_telemetryport){
		case HWVBRAIN_TELEMETRYPORT_DISABLED:
			break;
		case HWVBRAIN_TELEMETRYPORT_TELEMETRY:
			PIOS_Board_configure_com(&pios_usart_telem_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
			break;
		case HWVBRAIN_TELEMETRYPORT_COMAUX:
			PIOS_Board_configure_com(&pios_usart_telem_cfg, PIOS_COM_AUX_RX_BUF_LEN, PIOS_COM_AUX_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_aux_id);
			break;
		case HWVBRAIN_TELEMETRYPORT_COMBRIDGE:
			PIOS_Board_configure_com(&pios_usart_telem_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
			break;
			
	} /* 	hw_telemetryport */

	/* Configure GPS port */
	uint8_t hw_gpsport;
	HwVbrainGPSPortGet(&hw_gpsport);
	switch (hw_gpsport){
		case HWVBRAIN_GPSPORT_DISABLED:
			break;
			
		case HWVBRAIN_GPSPORT_TELEMETRY:
			PIOS_Board_configure_com(&pios_usart_gps_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
			break;
			
		case HWVBRAIN_GPSPORT_GPS:
			PIOS_Board_configure_com(&pios_usart_gps_cfg, PIOS_COM_GPS_RX_BUF_LEN, -1,  &pios_usart_com_driver, &pios_com_gps_id);
			break;
		
		case HWVBRAIN_GPSPORT_COMAUX:
			PIOS_Board_configure_com(&pios_usart_gps_cfg, PIOS_COM_AUX_RX_BUF_LEN, PIOS_COM_AUX_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_aux_id);
			break;
			
		case HWVBRAIN_GPSPORT_COMBRIDGE:
			PIOS_Board_configure_com(&pios_usart_gps_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
			break;
	}/* hw_gpsport */

	/* Configure AUXPort */
	uint8_t hw_auxport;
	HwVbrainAuxPortGet(&hw_auxport);

	switch (hw_auxport) {
		case HWVBRAIN_AUXPORT_DISABLED:
			break;
		case HWVBRAIN_AUXPORT_SBUS:
#ifdef PIOS_INCLUDE_SBUS
		{
			uint32_t pios_usart_sbus_id;
			if (PIOS_USART_Init(&pios_usart_sbus_id, &pios_usart1_sbus_cfg)) {
				PIOS_Assert(0);
			}

			uint32_t pios_sbus_id;
			if (PIOS_SBus_Init(&pios_sbus_id, &pios_sbus_cfg, &pios_usart_com_driver, pios_usart_sbus_id)) {
				PIOS_Assert(0);
			}

			uint32_t pios_sbus_rcvr_id;
			if (PIOS_RCVR_Init(&pios_sbus_rcvr_id, &pios_sbus_rcvr_driver, pios_sbus_id)) {
				PIOS_Assert(0);
			}
			pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_SBUS] = pios_sbus_rcvr_id;
			
		}
#endif /* PIOS_INCLUDE_SBUS */
			break;
//		case HWVBRAIN_AUXPORT_TELEMETRY:
//			PIOS_Board_configure_com(&pios_usart_aux_cfg, PIOS_COM_TELEM_RF_RX_BUF_LEN, PIOS_COM_TELEM_RF_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_rf_id);
//			break;
			
		case HWVBRAIN_AUXPORT_DSM2:
		case HWVBRAIN_AUXPORT_DSMX10BIT:
		case HWVBRAIN_AUXPORT_DSMX11BIT:
		{
			enum pios_dsm_proto proto;
			switch (hw_auxport) {
				case HWVBRAIN_AUXPORT_DSM2:
					proto = PIOS_DSM_PROTO_DSM2;
					break;
				case HWVBRAIN_AUXPORT_DSMX10BIT:
					proto = PIOS_DSM_PROTO_DSMX10BIT;
					break;
				case HWVBRAIN_AUXPORT_DSMX11BIT:
					proto = PIOS_DSM_PROTO_DSMX11BIT;
					break;
				default:
					PIOS_Assert(0);
					break;
			}
			//TODO: Define the various Channelgroup for Revo dsm inputs and handle here
			PIOS_Board_configure_dsm(&pios_usart1_dsm_cfg, &pios_dsm_aux_cfg,
											 &pios_usart_com_driver, &proto, MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT,&hw_DSMxBind);
		}
			break;
		case HWVBRAIN_AUXPORT_COMAUX:
		//	PIOS_Board_configure_com(&pios_usart_aux_cfg, PIOS_COM_AUX_RX_BUF_LEN, PIOS_COM_AUX_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_aux_id);
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE) && defined(PIOS_INCLUDE_USART) && defined(PIOS_INCLUDE_COM)
		PIOS_Board_configure_com(&pios_usart_aux_cfg, 0, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_aux_id);
#endif	/* PIOS_INCLUDE_DEBUG_CONSOLE */
			break;
		case HWVBRAIN_AUXPORT_COMBRIDGE:
			PIOS_Board_configure_com(&pios_usart_aux_cfg, PIOS_COM_BRIDGE_RX_BUF_LEN, PIOS_COM_BRIDGE_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_bridge_id);
			break;
	} /* hw_auxport */
	
	/* Configure the receiver port*/
	uint8_t hw_rcvrport;
	HwVbrainRcvrPortGet(&hw_rcvrport);
	//   
	switch (hw_rcvrport){
		case HWVBRAIN_RCVRPORT_DISABLED:
			break;
		case HWVBRAIN_RCVRPORT_PWM:
#if defined(PIOS_INCLUDE_PWM)
		{
			/* Set up the receiver port.  Later this should be optional */
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
		case HWVBRAIN_RCVRPORT_PPM:
		case HWVBRAIN_RCVRPORT_PPMOUTPUTS:
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
		case HWVBRAIN_RCVRPORT_OUTPUTS:
		
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
		case HWVBRAIN_RCVRPORT_DISABLED:
		case HWVBRAIN_RCVRPORT_PWM:
		case HWVBRAIN_RCVRPORT_PPM:
			/* Set up the servo outputs */
			PIOS_Servo_Init(&pios_servo_cfg);
			break;
		case HWVBRAIN_RCVRPORT_PPMOUTPUTS:
		case HWVBRAIN_RCVRPORT_OUTPUTS:
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
	
	PIOS_WDG_Clear();
	PIOS_DELAY_WaitmS(200);
	PIOS_WDG_Clear();

#if defined(PIOS_INCLUDE_HMC5883)
	PIOS_HMC5883_Init(PIOS_I2C_MAIN_ADAPTER, &pios_hmc5883_cfg);
	//I2C is slow, sensor init as well, reset watchdog to prevent reset here
	PIOS_WDG_Clear();
#endif

#if defined(PIOS_INCLUDE_ADC)
	PIOS_ADC_Init(&pios_adc_cfg);
#endif

#if defined(PIOS_INCLUDE_MS5611_SPI)
	PIOS_MS5611_Init(pios_spi_baro_flash_id, 0, &pios_ms5611_cfg);
#endif

#if defined(PIOS_INCLUDE_MPU6000)
			PIOS_MPU6000_Init(pios_spi_gyro_id,0, &pios_mpu6000_cfg);

			// To be safe map from UAVO enum to driver enum
			uint8_t hw_gyro_range;
			HwVbrainGyroRangeGet(&hw_gyro_range);
			switch(hw_gyro_range) {
				case HWVBRAIN_GYRORANGE_250:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_250_DEG);
					break;
				case HWVBRAIN_GYRORANGE_500:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_500_DEG);
					break;
				case HWVBRAIN_GYRORANGE_1000:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_1000_DEG);
					break;
				case HWVBRAIN_GYRORANGE_2000:
					PIOS_MPU6000_SetGyroRange(PIOS_MPU60X0_SCALE_2000_DEG);
					break;
			}

			uint8_t hw_accel_range;
			HwVbrainAccelRangeGet(&hw_accel_range);
			switch(hw_accel_range) {
				case HWVBRAIN_ACCELRANGE_2G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_2G);
					break;
				case HWVBRAIN_ACCELRANGE_4G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_4G);
					break;
				case HWVBRAIN_ACCELRANGE_8G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_8G);
					break;
				case HWVBRAIN_ACCELRANGE_16G:
					PIOS_MPU6000_SetAccelRange(PIOS_MPU60X0_ACCEL_16G);
					break;
			}

#endif
}

/**
 * @}
 * @}
 */

