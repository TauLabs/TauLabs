/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup ComUsbBridgeModule Com Port to USB VCP Bridge Module
 * @{ 
 *
 * @file       ComUsbBridge.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      Bridges selected Com Port to the USB VCP emulated serial port
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

// ****************

#include "openpilot.h"

#include "modulesettings.h"
#include "pios_thread.h"

#include <stdbool.h>

// ****************
// Private functions

static void com2UsbBridgeTask(void *parameters);
static void usb2ComBridgeTask(void *parameters);
static void updateSettings();

// ****************
// Private constants

#if defined(PIOS_COMUSBBRIDGE_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_COMUSBBRIDGE_STACK_SIZE
#else
#define STACK_SIZE_BYTES 384
#endif

#define TASK_PRIORITY                   PIOS_THREAD_PRIO_LOW

#define BRIDGE_BUF_LEN 10

// ****************
// Private variables

static struct pios_thread *com2UsbBridgeTaskHandle;
static struct pios_thread *usb2ComBridgeTaskHandle;

static uint8_t * com2usb_buf;
static uint8_t * usb2com_buf;

static uint32_t usart_port;
static uint32_t vcp_port;

static bool module_enabled = false;

/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */

static int32_t comUsbBridgeStart(void)
{
	if (module_enabled) {
		// Start tasks
		com2UsbBridgeTaskHandle = PIOS_Thread_Create(com2UsbBridgeTask, "Com2UsbBridge", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
		TaskMonitorAdd(TASKINFO_RUNNING_COM2USBBRIDGE, com2UsbBridgeTaskHandle);
		usb2ComBridgeTaskHandle = PIOS_Thread_Create(usb2ComBridgeTask, "Usb2ComBridge", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
		TaskMonitorAdd(TASKINFO_RUNNING_USB2COMBRIDGE, usb2ComBridgeTaskHandle);
		return 0;
	}

	return -1;
}
/**
 * Initialise the module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
static int32_t comUsbBridgeInitialize(void)
{
	// TODO: Get from settings object
	usart_port = PIOS_COM_BRIDGE;
	vcp_port = PIOS_COM_VCP;

	/* Only run the module if we have a VCP port and a selected USART port */
	if (!usart_port || !vcp_port) {
		module_enabled = false;
		return 0;
	}

#ifdef MODULE_ComUsbBridge_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_COMUSBBRIDGE] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (module_enabled) {
		com2usb_buf = pvPortMalloc(BRIDGE_BUF_LEN);
		PIOS_Assert(com2usb_buf);
		usb2com_buf = pvPortMalloc(BRIDGE_BUF_LEN);
		PIOS_Assert(usb2com_buf);

		updateSettings();
	}

	return 0;
}
MODULE_INITCALL(comUsbBridgeInitialize, comUsbBridgeStart)

/**
 * Main task. It does not return.
 */

static void com2UsbBridgeTask(void *parameters)
{
	/* Handle usart -> vcp direction */
	volatile uint32_t tx_errors = 0;
	while (1) {
		uint32_t rx_bytes;

		rx_bytes = PIOS_COM_ReceiveBuffer(usart_port, com2usb_buf, BRIDGE_BUF_LEN, 500);
		if (rx_bytes > 0) {
			/* Bytes available to transfer */
			if (PIOS_COM_SendBuffer(vcp_port, com2usb_buf, rx_bytes) != rx_bytes) {
				/* Error on transmit */
				tx_errors++;
			}
		}
	}
}

static void usb2ComBridgeTask(void * parameters)
{
	/* Handle vcp -> usart direction */
	volatile uint32_t tx_errors = 0;
	while (1) {
		uint32_t rx_bytes;

		rx_bytes = PIOS_COM_ReceiveBuffer(vcp_port, usb2com_buf, BRIDGE_BUF_LEN, 500);
		if (rx_bytes > 0) {
			/* Bytes available to transfer */
			if (PIOS_COM_SendBuffer(usart_port, usb2com_buf, rx_bytes) != rx_bytes) {
				/* Error on transmit */
				tx_errors++;
			}
		}
	}
}


static void updateSettings()
{
	if (usart_port) {

		// Retrieve settings
		uint8_t speed;
		ModuleSettingsComUsbBridgeSpeedGet(&speed);

		// Set port speed
		switch (speed) {
		case MODULESETTINGS_COMUSBBRIDGESPEED_2400:
			PIOS_COM_ChangeBaud(usart_port, 2400);
			break;
		case MODULESETTINGS_COMUSBBRIDGESPEED_4800:
			PIOS_COM_ChangeBaud(usart_port, 4800);
			break;
		case MODULESETTINGS_COMUSBBRIDGESPEED_9600:
			PIOS_COM_ChangeBaud(usart_port, 9600);
			break;
		case MODULESETTINGS_COMUSBBRIDGESPEED_19200:
			PIOS_COM_ChangeBaud(usart_port, 19200);
			break;
		case MODULESETTINGS_COMUSBBRIDGESPEED_38400:
			PIOS_COM_ChangeBaud(usart_port, 38400);
			break;
		case MODULESETTINGS_COMUSBBRIDGESPEED_57600:
			PIOS_COM_ChangeBaud(usart_port, 57600);
			break;
		case MODULESETTINGS_COMUSBBRIDGESPEED_115200:
			PIOS_COM_ChangeBaud(usart_port, 115200);
			break;
		}
	}
}

/**
 * @}
 * @}
 */
 
