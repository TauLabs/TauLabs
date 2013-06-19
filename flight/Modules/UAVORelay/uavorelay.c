/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup UAVORelay UAVORelay Module
 * @{ 
 *
 * @file       uavorelay.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Forward a set of UAVObjects when updated out a PIOS_COM port
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

#include "openpilot.h"
#include "modulesettings.h"
#include "cameradesired.h"

// Private constants
#define MAX_QUEUE_SIZE   5
#define STACK_SIZE_BYTES 512
#define TASK_PRIORITY (tskIDLE_PRIORITY + 0)

// Private types

// Private variables
static xQueueHandle queue;
static UAVTalkConnection uavTalkCon;
static xTaskHandle uavoRelayTaskHandle;
static bool module_enabled;

// Private functions
static void    uavoRelayTask(void *parameters);
static int32_t send_data(uint8_t *data, int32_t length);
static void    register_object(UAVObjHandle obj);

// Local variables
static uintptr_t uavorelay_com_id;

// External variables
extern uintptr_t pios_com_can_id;

/**
 * Initialise the UAVORelay module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t UAVORelayInitialize(void)
{
	// TODO: make selectable
	uavorelay_com_id = pios_com_can_id;

#ifdef MODULE_UAVORelay_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_UAVORELAY] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif

	if (!uavorelay_com_id)
		module_enabled = false;

	if (!module_enabled)
		return -1;

	// Create object queues
	queue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));
	
	// Initialise UAVTalk
	uavTalkCon = UAVTalkInitialize(&send_data);

	CameraDesiredInitialize();

	return 0;
}

/**
 * Start the UAVORelay module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t UAVORelayStart(void)
{
	//Check if module is enabled or not
	if (module_enabled == false) {
		return -1;
	}
	
	// Register objects to relay
	if (CameraDesiredHandle())
		register_object(CameraDesiredHandle());
	
	// Start relay task
	xTaskCreate(uavoRelayTask, (signed char *)"UAVORelay", STACK_SIZE_BYTES/4,
	            NULL, TASK_PRIORITY, &uavoRelayTaskHandle);

	TaskMonitorAdd(TASKINFO_RUNNING_UAVORELAY, uavoRelayTaskHandle);
	
	return 0;
}

MODULE_INITCALL(UAVORelayInitialize, UAVORelayStart)
;
/**
 * Register a new object, adds object to local list and connects the queue depending on the object's
 * telemetry settings.
 * \param[in] obj Object to connect
 */
static void register_object(UAVObjHandle obj)
{
	int32_t eventMask;
	eventMask = EV_UPDATED | EV_UPDATED_MANUAL | EV_UPDATE_REQ | EV_UNPACKED;
	UAVObjConnectQueue(obj, queue, eventMask);
}

static void uavoRelayTask(void *parameters)
{
	UAVObjEvent ev;

	// Loop forever
	while (1) {

		vTaskDelay(50);

		// Wait for queue message
		if (xQueueReceive(queue, &ev, 2) == pdTRUE) {
			// Process event.  This calls transmitData
			UAVTalkSendObject(uavTalkCon, ev.obj, ev.instId, false, 0);
		}

		// Process incoming data in sufficient chunks that we keep up
		uint8_t serial_data[8];
		uint16_t bytes_to_process;

		bytes_to_process = PIOS_COM_ReceiveBuffer(uavorelay_com_id, serial_data, sizeof(serial_data), 0);
		do {
			bytes_to_process = PIOS_COM_ReceiveBuffer(uavorelay_com_id, serial_data, sizeof(serial_data), 0);
			for (uint8_t i = 0; i < bytes_to_process; i++)
				UAVTalkProcessInputStream(uavTalkCon,serial_data[i]);
		} while (bytes_to_process > 0);

	}
}

/**
 * Forward data from UAVTalk out the serial port
 * \param[in] data Data buffer to send
 * \param[in] length Length of buffer
 * \return -1 on failure
 * \return number of bytes transmitted on success
 */
static int32_t send_data(uint8_t *data, int32_t length)
{
	if( PIOS_COM_SendBufferNonBlocking(uavorelay_com_id, data, length) < 0)
		return -1;

	return length;
}

/**
  * @}
  * @}
  */
