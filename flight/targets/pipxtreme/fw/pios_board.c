/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup PipXtreme OpenPilot PipXtreme support files
 * @{
 *
 * @file       pios_board.c 
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
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

#include <pios.h>
#include <openpilot.h>
#include <board_hw_defs.c>
#include <modulesettings.h>
#include <hwtaulink.h>
#include <pios_hal.h>
#include <rfm22bstatus.h>

uintptr_t pios_com_telem_uart_bluetooth_id;
#if defined(PIOS_INCLUDE_PPM)
uintptr_t pios_ppm_rcvr_id;
#endif

uintptr_t pios_uavo_settings_fs_id;

#define PIOS_COM_TELEM_RX_BUF_LEN 256
#define PIOS_COM_TELEM_TX_BUF_LEN 256
#define PIOS_COM_FRSKYSPORT_TX_BUF_LEN 16

/**
 * PIOS_Board_Init()
 * initializes all the core subsystems on this specific hardware
 * called from System/openpilot.c
 */
void PIOS_Board_Init(void) {

	/* Delay system */
	PIOS_DELAY_Init();

	const struct pios_board_info * bdinfo = &pios_board_info_blob;

#if defined(PIOS_INCLUDE_FLASH) && defined(PIOS_INCLUDE_LOGFS_SETTINGS)
	/* Inititialize all flash drivers */
	PIOS_Flash_Internal_Init(&pios_internal_flash_id, &flash_internal_cfg);

	/* Register the partition table */
	const struct pios_flash_partition * flash_partition_table;
	uint32_t num_partitions;
	flash_partition_table = PIOS_BOARD_HW_DEFS_GetPartitionTable(bdinfo->board_rev, &num_partitions);
	PIOS_FLASH_register_partition_table(flash_partition_table, num_partitions);

	/* Mount all filesystems */
	PIOS_FLASHFS_Logfs_Init(&pios_uavo_settings_fs_id, &flashfs_internal_settings_cfg, FLASH_PARTITION_LABEL_SETTINGS);
#endif	/* PIOS_INCLUDE_FLASH && PIOS_INCLUDE_LOGFS_SETTINGS */

	/* Initialize the task monitor library */
	TaskMonitorInitialize();

	/* Initialize UAVObject libraries */
	EventDispatcherInitialize();
	UAVObjInitialize();

	/* Set up the SPI interface to the rfm22b */
	if (PIOS_SPI_Init(&pios_spi_rfm22b_id, &pios_spi_rfm22b_cfg)) {
		PIOS_DEBUG_Assert(0);
	}

#ifdef PIOS_INCLUDE_WDG
	/* Initialize watchdog as early as possible to catch faults during init */
	PIOS_WDG_Init();
#endif /* PIOS_INCLUDE_WDG */

#if defined(PIOS_INCLUDE_RTC)
	/* Initialize the real-time clock and its associated tick */
	PIOS_RTC_Init(&pios_rtc_main_cfg);
#endif /* PIOS_INCLUDE_RTC */

	HwTauLinkInitialize();
#if defined(PIOS_INCLUDE_RFM22B)
	RFM22BStatusInitialize();
#endif /* PIOS_INCLUDE_RFM22B */

#if defined(PIOS_INCLUDE_LED)
	PIOS_LED_Init(&pios_led_cfg);
#endif	/* PIOS_INCLUDE_LED */

#if defined(PIOS_INCLUDE_TIM)
	/* Set up pulse timers */
	PIOS_TIM_InitClock(&tim_1_cfg);
	PIOS_TIM_InitClock(&tim_2_cfg);
	PIOS_TIM_InitClock(&tim_3_cfg);
	PIOS_TIM_InitClock(&tim_4_cfg);
#endif	/* PIOS_INCLUDE_TIM */

	/* Initialize board specific USB data */
	PIOS_USB_BOARD_DATA_Init();

	/* Flags to determine if various USB interfaces are advertised */
	bool usb_cdc_present = false;

#if defined(PIOS_INCLUDE_USB_CDC)
	if (PIOS_USB_DESC_HID_CDC_Init()) {
		PIOS_Assert(0);
	}
	usb_cdc_present = true;
#else
	if (PIOS_USB_DESC_HID_ONLY_Init()) {
		PIOS_Assert(0);
	}
#endif

	// Configure the main port
	HwTauLinkData hwTauLink;
	HwTauLinkGet(&hwTauLink);

	/*Initialize the USB device */
	uintptr_t pios_usb_id;
	PIOS_USB_Init(&pios_usb_id, &pios_usb_main_cfg);

	PIOS_HAL_ConfigureHID(HWSHARED_USB_HIDPORT_USBTELEMETRY, pios_usb_id, &pios_usb_hid_cfg);

	/* Configure the USB virtual com port (VCP) */
#if defined(PIOS_INCLUDE_USB_CDC)
	if (usb_cdc_present)
	{
		PIOS_HAL_ConfigureCDC(hwTauLink.VCPPort, pios_usb_id, &pios_usb_cdc_cfg);
	}
#endif

	PIOS_HAL_ConfigurePort(hwTauLink.MainPort,   // port type protocol
			&pios_usart_serial_cfg,              // usart_port_cfg
			&pios_usart_serial_cfg,              // frsky usart_port_cfg
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
			false);                              // sbus_toggle

	// Update the com baud rate.
	uint32_t comBaud = 9600;
	switch (hwTauLink.ComSpeed) {
	case HWTAULINK_COMSPEED_4800:
		comBaud = 4800;
		break;
	case HWTAULINK_COMSPEED_9600:
		comBaud = 9600;
		break;
	case HWTAULINK_COMSPEED_19200:
		comBaud = 19200;
		break;
	case HWTAULINK_COMSPEED_38400:
		comBaud = 38400;
		break;
	case HWTAULINK_COMSPEED_57600:
		comBaud = 57600;
		break;
	case HWTAULINK_COMSPEED_115200:
		comBaud = 115200;
		break;
	}

	const struct pios_rfm22b_cfg *rfm22b_cfg = PIOS_BOARD_HW_DEFS_GetRfm22Cfg(bdinfo->board_rev);
	PIOS_HAL_ConfigureRFM22B(hwTauLink.Radio, bdinfo->board_type,
	    bdinfo->board_rev, hwTauLink.MaxRfPower,
	    hwTauLink.MaxRfSpeed, hwTauLink.RfBand, NULL, rfm22b_cfg,
	    hwTauLink.MinChannel, hwTauLink.MaxChannel,
	    hwTauLink.CoordID, HWSHARED_PORTTYPES_DISABLED, 0);

	if (bdinfo->board_rev == TAULINK_VERSION_MODULE) {
		// Configure the main serial port function
		switch (hwTauLink.BTPort) {
		case HWTAULINK_BTPORT_TELEMETRY:
		{
			// Note: if the main port is also on telemetry the bluetooth
			// port will take precedence
			PIOS_HAL_ConfigureUsart(&pios_usart_bluetooth_cfg, PIOS_COM_TELEM_RX_BUF_LEN, PIOS_COM_TELEM_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_uart_bluetooth_id);
			PIOS_COM_ChangeBaud(pios_com_telem_uart_bluetooth_id, 9600);

			// Note this doesn't actually send until RTOS is running
			PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT");
			PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+NAMETauLink");
			PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+PIN:000000");
			PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+MODE1");
			PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+SHOW1");
			PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+RESET");
			PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+BAUD4"); // 115200
			PIOS_COM_ChangeBaud(pios_com_telem_uart_bluetooth_id, 115200);
			pios_com_telem_serial_id = pios_com_telem_uart_bluetooth_id;
		}
			break;
		case HWTAULINK_BTPORT_COMBRIDGE:
		{
			// Note: if the main port is also on telemetry the bluetooth
			// port will take precedence
			PIOS_HAL_ConfigureUsart(&pios_usart_bluetooth_cfg, PIOS_COM_TELEM_RX_BUF_LEN, PIOS_COM_TELEM_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_telem_uart_bluetooth_id);

			// Since we don't expose the ModuleSettings object from TauLink to the GCS
			// we just map the baud rate from HwTauLink into this object
			ModuleSettingsInitialize();
			ModuleSettingsData moduleSettings;
			ModuleSettingsGet(&moduleSettings);
			switch(comBaud) {
				case 4800:
					moduleSettings.ComUsbBridgeSpeed = MODULESETTINGS_COMUSBBRIDGESPEED_4800;
					break;
				case 9600:
					moduleSettings.ComUsbBridgeSpeed = MODULESETTINGS_COMUSBBRIDGESPEED_9600;
					break;
				case 19200:
					moduleSettings.ComUsbBridgeSpeed = MODULESETTINGS_COMUSBBRIDGESPEED_19200;
					break;
				case 38400:
					moduleSettings.ComUsbBridgeSpeed = MODULESETTINGS_COMUSBBRIDGESPEED_38400;
					break;
				case 57600:
					moduleSettings.ComUsbBridgeSpeed = MODULESETTINGS_COMUSBBRIDGESPEED_57600;
					break;
				case 115200:
					moduleSettings.ComUsbBridgeSpeed = MODULESETTINGS_COMUSBBRIDGESPEED_115200;
					break;
				default:
					moduleSettings.ComUsbBridgeSpeed = MODULESETTINGS_COMUSBBRIDGESPEED_9600;
			}
			ModuleSettingsSet(&moduleSettings);

			PIOS_COM_ChangeBaud(pios_com_telem_uart_bluetooth_id, comBaud);
			pios_com_bridge_id = pios_com_telem_uart_bluetooth_id;
		}
			break;
		case HWTAULINK_BTPORT_DISABLED:
			break;
		}
	}

    // Configure the flexi port
    switch (hwTauLink.PPMPort) {
    case HWTAULINK_PPMPORT_PPM:
    {
#if defined(PIOS_INCLUDE_PPM)
				/* PPM input is configured on the coordinator modem and sent in the RFM22BReceiver UAVO. */
				uintptr_t pios_ppm_id;
				PIOS_PPM_Init(&pios_ppm_id, &pios_ppm_cfg);

				if (PIOS_RCVR_Init(&pios_ppm_rcvr_id, &pios_ppm_rcvr_driver, pios_ppm_id)) {
					PIOS_Assert(0);
				}

#endif /* PIOS_INCLUDE_PPM */
        break;
    }
    case HWTAULINK_PPMPORT_SPORT:
#if defined(PIOS_INCLUDE_TARANIS_SPORT)
        PIOS_HAL_ConfigureUsart(&pios_usart_sport_cfg, 0, PIOS_COM_FRSKYSPORT_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_frsky_sport_id);
#endif /* PIOS_INCLUDE_TARANIS_SPORT */
        break;
    case HWTAULINK_PPMPORT_PPMSPORT:
    {
#if defined(PIOS_HAL_ConfigureUsart)
        PIOS_HAL_ConfigureUsart(&pios_usart_sport_cfg, 0, PIOS_COM_FRSKYSPORT_TX_BUF_LEN, &pios_usart_com_driver, &pios_com_frsky_sport_id);
#endif /* PIOS_INCLUDE_TARANIS_SPORT */
#if defined(PIOS_INCLUDE_PPM)
        /* PPM input is configured on the coordinator modem and sent in the RFM22BReceiver UAVO. */
        uintptr_t pios_ppm_id;
        PIOS_PPM_Init(&pios_ppm_id, &pios_ppm_cfg);

        if (PIOS_RCVR_Init(&pios_ppm_rcvr_id, &pios_ppm_rcvr_driver, pios_ppm_id)) {
            PIOS_Assert(0);
        }

#endif /* PIOS_INCLUDE_PPM */
        break;
    }
    case HWTAULINK_PPMPORT_DISABLED:
    default:
        break;
    }

	if (PIOS_COM_TELEMETRY) {
		PIOS_COM_ChangeBaud(PIOS_COM_TELEMETRY, comBaud);
	}

	/* Remap AFIO pin */
	GPIO_PinRemapConfig( GPIO_Remap_SWJ_NoJTRST, ENABLE);

#ifdef PIOS_INCLUDE_ADC
	PIOS_ADC_Init();
#endif
	PIOS_GPIO_Init();
}

/**
 * @}
 */
