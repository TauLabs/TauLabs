/**
 ******************************************************************************
 * @addtogroup TauLabsTargets Tau Labs Targets
 * @{
 * @addtogroup PipXtreme OpenPilot PipXtreme support files
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

#include <pios.h>
#include <openpilot.h>
#include <board_hw_defs.c>
#include <modulesettings.h>
#include <hwtaulink.h>
#include <rfm22bstatus.h>

#define PIOS_COM_TELEM_RX_BUF_LEN 256
#define PIOS_COM_TELEM_TX_BUF_LEN 256

#define PIOS_COM_TELEM_USB_RX_BUF_LEN 256
#define PIOS_COM_TELEM_USB_TX_BUF_LEN 256

#define PIOS_COM_TELEM_VCP_RX_BUF_LEN 256
#define PIOS_COM_TELEM_VCP_TX_BUF_LEN 256

#define PIOS_COM_RFM22B_RF_RX_BUF_LEN 256
#define PIOS_COM_RFM22B_RF_TX_BUF_LEN 256

#define PIOS_COM_RFM22B_RF_RX_BUF_LEN 256
#define PIOS_COM_RFM22B_RF_TX_BUF_LEN 256

#define PIOS_COM_BRIDGE_RX_BUF_LEN 65
#define PIOS_COM_BRIDGE_TX_BUF_LEN 12

#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
#define PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN 40
uintptr_t pios_com_debug_id;
#endif /* PIOS_INCLUDE_DEBUG_CONSOLE */

uintptr_t pios_com_telem_usb_id;
uintptr_t pios_com_telem_vcp_id;
uintptr_t pios_com_vcp_id;
uintptr_t pios_com_bridge_id;
uintptr_t pios_com_telem_uart_main_id;
uintptr_t pios_com_telem_uart_flexi_id;
uintptr_t pios_com_telem_uart_telem_id;
uintptr_t pios_com_telem_uart_bluetooth_id;
uintptr_t pios_com_telemetry_id;
#if defined(PIOS_INCLUDE_PPM)
uintptr_t pios_ppm_rcvr_id;
#endif
#if defined(PIOS_INCLUDE_RFM22B)
uint32_t pios_rfm22b_id;
uintptr_t pios_com_rfm22b_id;
uintptr_t pios_com_radio_id;
#endif

uint8_t *pios_uart_rx_buffer;
uint8_t *pios_uart_tx_buffer;

uintptr_t pios_uavo_settings_fs_id;

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

#if defined(PIOS_INCLUDE_RFM22B)
    HwTauLinkInitialize();
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

	/*Initialize the USB device */
	uintptr_t pios_usb_id;
	PIOS_USB_Init(&pios_usb_id, &pios_usb_main_cfg);

	/* Configure the USB HID port */
	{
		uintptr_t pios_usb_hid_id;
		if (PIOS_USB_HID_Init(&pios_usb_hid_id, &pios_usb_hid_cfg, pios_usb_id)) {
			PIOS_Assert(0);
		}
		uint8_t *rx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_USB_RX_BUF_LEN);
		uint8_t *tx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_USB_TX_BUF_LEN);
		PIOS_Assert(rx_buffer);
		PIOS_Assert(tx_buffer);
		if (PIOS_COM_Init(&pios_com_telem_usb_id, &pios_usb_hid_com_driver, pios_usb_hid_id,
											rx_buffer, PIOS_COM_TELEM_USB_RX_BUF_LEN,
											tx_buffer, PIOS_COM_TELEM_USB_TX_BUF_LEN)) {
			PIOS_Assert(0);
		}
	}

	/* Configure the USB virtual com port (VCP) */
#if defined(PIOS_INCLUDE_USB_CDC)
	if (usb_cdc_present)
	{
		uintptr_t pios_usb_cdc_id;
		if (PIOS_USB_CDC_Init(&pios_usb_cdc_id, &pios_usb_cdc_cfg, pios_usb_id)) {
			PIOS_Assert(0);
		}
		uint8_t *rx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_VCP_RX_BUF_LEN);
		uint8_t *tx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_VCP_TX_BUF_LEN);
		PIOS_Assert(rx_buffer);
		PIOS_Assert(tx_buffer);
		if (PIOS_COM_Init(&pios_com_telem_vcp_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
											rx_buffer, PIOS_COM_TELEM_VCP_RX_BUF_LEN,
											tx_buffer, PIOS_COM_TELEM_VCP_TX_BUF_LEN)) {
			PIOS_Assert(0);
		}
	}
#endif

    /* Allocate the uart buffers. */
    pios_uart_rx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_RX_BUF_LEN);
    pios_uart_tx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_TX_BUF_LEN);

    // Configure the main port
    HwTauLinkData hwTauLink;
    HwTauLinkGet(&hwTauLink);

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


    bool is_oneway   = hwTauLink.Radio == HWTAULINK_RADIO_PPM;
    bool ppm_only    = hwTauLink.Radio == HWTAULINK_RADIO_PPM;
    bool ppm_mode    = hwTauLink.Radio == HWTAULINK_RADIO_TELEMPPM || hwTauLink.Radio == HWTAULINK_RADIO_PPM;

    // Configure the main serial port function
    switch (hwTauLink.MainPort) {
    case HWTAULINK_MAINPORT_TELEMETRY:
    case HWTAULINK_MAINPORT_COMBRIDGE:
    {
        /* Configure the main port for uart serial */
#ifndef PIOS_RFM22B_DEBUG_ON_TELEM
	{
		uintptr_t pios_usart1_id;
		if (PIOS_USART_Init(&pios_usart1_id, &pios_usart_serial_cfg)) {
			PIOS_Assert(0);
		}
		uint8_t *rx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_RX_BUF_LEN);
		uint8_t *tx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_TX_BUF_LEN);
		PIOS_Assert(rx_buffer);
		PIOS_Assert(tx_buffer);
		if (PIOS_COM_Init(&pios_com_telem_uart_telem_id, &pios_usart_com_driver, pios_usart1_id,
											rx_buffer, PIOS_COM_TELEM_RX_BUF_LEN,
											tx_buffer, PIOS_COM_TELEM_TX_BUF_LEN)) {
			PIOS_Assert(0);
		}

		pios_com_telemetry_id = pios_com_telem_uart_telem_id;
	}

#endif
        break;
    }
    case HWTAULINK_MAINPORT_DISABLED:
        break;
    }

    if (bdinfo->board_rev == TAULINK_VERSION_MODULE) {
        // Configure the main serial port function
        switch (hwTauLink.BTPort) {
        case HWTAULINK_BTPORT_TELEMETRY:
        {
            // Note: if the main port is also on telemetry the bluetooth
            // port will take precedence
            uintptr_t pios_usart2_id;
            if (PIOS_USART_Init(&pios_usart2_id, &pios_usart_bluetooth_cfg)) {
                PIOS_Assert(0);
            }
            uint8_t *rx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_RX_BUF_LEN);
            uint8_t *tx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_TX_BUF_LEN);
            PIOS_Assert(rx_buffer);
            PIOS_Assert(tx_buffer);
            if (PIOS_COM_Init(&pios_com_telem_uart_bluetooth_id, &pios_usart_com_driver, pios_usart2_id,
                                                rx_buffer, PIOS_COM_TELEM_RX_BUF_LEN,
                                                tx_buffer, PIOS_COM_TELEM_TX_BUF_LEN)) {
                PIOS_Assert(0);
            }

            PIOS_COM_ChangeBaud(pios_com_telem_uart_bluetooth_id, comBaud);

            // Note this doesn't actually send until RTOS is running
            PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT");
            PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+NAMETauLink");
            PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+PIN:000000");
            PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+MODE1");
            PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+SHOW1");
            PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+RESET");
            PIOS_COM_SendString(pios_com_telem_uart_bluetooth_id,"AT+BAUD4"); // 115200
            //PIOS_COM_ChangeBaud(pios_com_telem_uart_bluetooth_id, 115200);
            pios_com_telemetry_id = pios_com_telem_uart_bluetooth_id;
        }
            break;
        case HWTAULINK_BTPORT_COMBRIDGE:
        {
            // Note: if the main port is also on telemetry the bluetooth
            // port will take precedence
            uintptr_t pios_usart2_id;
            if (PIOS_USART_Init(&pios_usart2_id, &pios_usart_bluetooth_cfg)) {
                PIOS_Assert(0);
            }
            uint8_t *rx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_RX_BUF_LEN);
            uint8_t *tx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_TELEM_TX_BUF_LEN);
            PIOS_Assert(rx_buffer);
            PIOS_Assert(tx_buffer);
            if (PIOS_COM_Init(&pios_com_telem_uart_bluetooth_id, &pios_usart_com_driver, pios_usart2_id,
                                                rx_buffer, PIOS_COM_TELEM_RX_BUF_LEN,
                                                tx_buffer, PIOS_COM_TELEM_TX_BUF_LEN)) {
                PIOS_Assert(0);
            }

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
        case HWTAULINK_MAINPORT_DISABLED:
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
    case HWTAULINK_PPMPORT_DISABLED:
    default:
        break;
    }

    // Configure the USB VCP port
    switch (hwTauLink.VCPPort) {
    case HWTAULINK_VCPPORT_TELEMETRY:
        PIOS_COM_TELEMETRY = pios_com_telem_vcp_id;
        break;
    case HWTAULINK_VCPPORT_COMBRIDGE:
#if defined(PIOS_INCLUDE_COM)
        {
            uintptr_t pios_usb_cdc_id;
            if (PIOS_USB_CDC_Init(&pios_usb_cdc_id, &pios_usb_cdc_cfg, pios_usb_id)) {
                PIOS_Assert(0);
            }
            uint8_t * rx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_BRIDGE_RX_BUF_LEN);
            uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_BRIDGE_TX_BUF_LEN);
            PIOS_Assert(rx_buffer);
            PIOS_Assert(tx_buffer);
            if (PIOS_COM_Init(&pios_com_vcp_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
                        rx_buffer, PIOS_COM_BRIDGE_RX_BUF_LEN,
                        tx_buffer, PIOS_COM_BRIDGE_TX_BUF_LEN)) {
                PIOS_Assert(0);
            }
        }
#endif  /* PIOS_INCLUDE_COM */
        break;
    case HWTAULINK_VCPPORT_DEBUGCONSOLE:
#if defined(PIOS_INCLUDE_COM)
#if defined(PIOS_INCLUDE_DEBUG_CONSOLE)
        {
            uintptr_t pios_usb_cdc_id;
            if (PIOS_USB_CDC_Init(&pios_usb_cdc_id, &pios_usb_cdc_cfg, pios_usb_id)) {
                PIOS_Assert(0);
            }
            uint8_t * tx_buffer = (uint8_t *) PIOS_malloc(PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN);
            PIOS_Assert(tx_buffer);
            if (PIOS_COM_Init(&pios_com_debug_id, &pios_usb_cdc_com_driver, pios_usb_cdc_id,
                        NULL, 0,
                        tx_buffer, PIOS_COM_DEBUGCONSOLE_TX_BUF_LEN)) {
                PIOS_Assert(0);
            }
        }
#endif  /* PIOS_INCLUDE_DEBUG_CONSOLE */
#endif  /* PIOS_INCLUDE_COM */
    case HWTAULINK_VCPPORT_DISABLED:
        break;
    }

    // Initialize out status object.
    RFM22BStatusData rfm22bstatus;
    RFM22BStatusGet(&rfm22bstatus);

    rfm22bstatus.BoardType     = bdinfo->board_type;
    rfm22bstatus.BoardRevision = bdinfo->board_rev;

    /* Initalize the RFM22B radio COM device. */
    if (hwTauLink.MaxRfPower != HWTAULINK_MAXRFPOWER_0) {
        rfm22bstatus.LinkState = RFM22BSTATUS_LINKSTATE_ENABLED;

        // Configure the RFM22B device
        const struct pios_rfm22b_cfg *rfm22b_cfg = PIOS_BOARD_HW_DEFS_GetRfm22Cfg(bdinfo->board_rev);
        if (PIOS_RFM22B_Init(&pios_rfm22b_id, PIOS_RFM22_SPI_PORT, rfm22b_cfg->slave_num, rfm22b_cfg)) {
            PIOS_Assert(0);
        }

        rfm22bstatus.DeviceID = PIOS_RFM22B_DeviceID(pios_rfm22b_id);
        rfm22bstatus.BoardRevision = PIOS_RFM22B_ModuleVersion(pios_rfm22b_id);

        // Configure the radio com interface
        uint8_t *rx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_RFM22B_RF_RX_BUF_LEN);
        uint8_t *tx_buffer = (uint8_t *)PIOS_malloc(PIOS_COM_RFM22B_RF_TX_BUF_LEN);
        PIOS_Assert(rx_buffer);
        PIOS_Assert(tx_buffer);
        if (PIOS_COM_Init(&pios_com_rfm22b_id, &pios_rfm22b_com_driver, pios_rfm22b_id,
                          rx_buffer, PIOS_COM_RFM22B_RF_RX_BUF_LEN,
                          tx_buffer, PIOS_COM_RFM22B_RF_TX_BUF_LEN)) {
            PIOS_Assert(0);
        }

        // Set the RF data rate on the modem to ~2X the selected buad rate because the modem is half duplex.
        enum rfm22b_datarate datarate = RFM22_datarate_64000;
        switch (hwTauLink.MaxRfSpeed) {
        case HWTAULINK_MAXRFSPEED_9600:
            datarate = RFM22_datarate_9600;
            break;
        case HWTAULINK_MAXRFSPEED_19200:
            datarate = RFM22_datarate_19200;
            break;
        case HWTAULINK_MAXRFSPEED_32000:
            datarate = RFM22_datarate_32000;
            break;
        case HWTAULINK_MAXRFSPEED_64000:
            datarate = RFM22_datarate_64000;
            break;
        case HWTAULINK_MAXRFSPEED_100000:
            datarate = RFM22_datarate_100000;
            break;
        case HWTAULINK_MAXRFSPEED_192000:
            datarate = RFM22_datarate_192000;
            break;
        }

        /* Set the modem Tx poer level */
        switch (hwTauLink.MaxRfPower) {
        case HWTAULINK_MAXRFPOWER_125:
            PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_0);
            break;
        case HWTAULINK_MAXRFPOWER_16:
            PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_1);
            break;
        case HWTAULINK_MAXRFPOWER_316:
            PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_2);
            break;
        case HWTAULINK_MAXRFPOWER_63:
            PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_3);
            break;
        case HWTAULINK_MAXRFPOWER_126:
            PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_4);
            break;
        case HWTAULINK_MAXRFPOWER_25:
            PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_5);
            break;
        case HWTAULINK_MAXRFPOWER_50:
            PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_6);
            break;
        case HWTAULINK_MAXRFPOWER_100:
            PIOS_RFM22B_SetTxPower(pios_rfm22b_id, RFM22_tx_pwr_txpow_7);
            break;
        default:
            // do nothing
            break;
        }

        // Set the radio configuration parameters.
        PIOS_RFM22B_Config(pios_rfm22b_id, datarate, hwTauLink.MinChannel, hwTauLink.MaxChannel, hwTauLink.CoordID, is_oneway, ppm_mode, ppm_only);

        // Reinitilize the modem to affect te changes.
        PIOS_RFM22B_Reinit(pios_rfm22b_id);
    } else {
        rfm22bstatus.LinkState = RFM22BSTATUS_LINKSTATE_DISABLED;
    }

    // Update the object
    RFM22BStatusSet(&rfm22bstatus);

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
