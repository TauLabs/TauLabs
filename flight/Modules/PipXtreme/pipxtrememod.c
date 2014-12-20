/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup PipXtremeModule PipXtreme Module
 * @{ 
 *
 * @file       pipxtrememod.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      This starts and handles the RF tasks for radio links
 *
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

#include <uavobjectmanager.h>
#include <openpilot.h>

#include <rfm22bstatus.h>
#include <taskinfo.h>

#include <pios_rfm22b.h>
#include <pios_board_info.h>
#include <oplinksettings.h>
#include "systemmod.h"
#include "pios_thread.h"

// Private constants
#define SYSTEM_UPDATE_PERIOD_MS 1000

#if defined(PIOS_SYSTEM_STACK_SIZE)
#define STACK_SIZE_BYTES        PIOS_SYSTEM_STACK_SIZE
#else
#define STACK_SIZE_BYTES        924
#endif

#define TASK_PRIORITY PIOS_THREAD_PRIO_NORMAL

// Private types

// Private variables
static struct pios_thread *systemTaskHandle;
static bool stackOverflow;
static bool mallocFailed;

// Private functions
static void systemTask(void *parameters);

/**
 * Create the module task.
 * \returns 0 on success or -1 if initialization failed
 */
int32_t PipXtremeModStart(void)
{
	// Initialize vars
	stackOverflow = false;
	mallocFailed = false;
	// Create pipxtreme system task
	systemTaskHandle = PIOS_Thread_Create(systemTask, "PipXtreme", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	// Register task
	TaskMonitorAdd(TASKINFO_RUNNING_SYSTEM, systemTaskHandle);

	return 0;
}

/**
 * Initialize the module, called on startup.
 * \returns 0 on success or -1 if initialization failed
 */
int32_t PipXtremeModInitialize(void)
{
	// Must registers objects here for system thread because ObjectManager started in OpenPilotInit

	// Call the module start function.
	PipXtremeModStart();

	return 0;
}

MODULE_INITCALL(PipXtremeModInitialize, 0);

/**
 * System task, periodically executes every SYSTEM_UPDATE_PERIOD_MS
 */
static void systemTask(void *parameters)
{

	uint32_t lastSysTime;
    uint16_t prev_tx_count = 0;
    uint16_t prev_rx_count = 0;
    bool first_time = true;

    /* create all modules thread */
    MODULE_TASKCREATE_ALL;

    lastSysTime = PIOS_Thread_Systime();

    if (mallocFailed) {
        /* We failed to malloc during task creation,
         * system behaviour is undefined.  Reset and let
         * the BootFault code recover for us.
         */
        PIOS_SYS_Reset();
    }

    // Initialize vars
    lastSysTime = PIOS_Thread_Systime();

    // Main system loop
    while (1) {
        // Flash the heartbeat LED

#if defined(PIOS_LED_HEARTBEAT)
		PIOS_LED_Toggle(PIOS_LED_HEARTBEAT);
#endif /* PIOS_LED_HEARTBEAT */

		// Update the RFM22BStatus UAVO
		RFM22BStatusData rfm22bStatus;
		RFM22BStatusGet(&rfm22bStatus);

		// Get the stats from the radio device
		struct rfm22b_stats radio_stats;
		PIOS_RFM22B_GetStats(pios_rfm22b_id, &radio_stats);

		if (pios_rfm22b_id) {
			// Update the status
			rfm22bStatus.HeapRemaining = PIOS_heap_get_free_size();
			rfm22bStatus.DeviceID = PIOS_RFM22B_DeviceID(pios_rfm22b_id);
			rfm22bStatus.BoardRevision = PIOS_RFM22B_ModuleVersion(pios_rfm22b_id);
			rfm22bStatus.RxGood = radio_stats.rx_good;
			rfm22bStatus.RxCorrected = radio_stats.rx_corrected;
			rfm22bStatus.RxErrors = radio_stats.rx_error;
			rfm22bStatus.RxSyncMissed = radio_stats.rx_sync_missed;
			rfm22bStatus.TxMissed = radio_stats.tx_missed;
			rfm22bStatus.RxFailure = radio_stats.rx_failure;
			rfm22bStatus.Resets = radio_stats.resets;
			rfm22bStatus.Timeouts = radio_stats.timeouts;
			rfm22bStatus.RSSI = radio_stats.rssi;
			rfm22bStatus.LinkQuality = radio_stats.link_quality;
			if (first_time) {
				first_time = false;
			} else {
				uint16_t tx_count = radio_stats.tx_byte_count;
				uint16_t rx_count = radio_stats.rx_byte_count;
				uint16_t tx_bytes =
				    (tx_count <
				     prev_tx_count) ? (0xffff -
						       prev_tx_count +
						       tx_count)
				    : (tx_count - prev_tx_count);
				uint16_t rx_bytes =
				    (rx_count <
				     prev_rx_count) ? (0xffff -
						       prev_rx_count +
						       rx_count)
				    : (rx_count - prev_rx_count);
				rfm22bStatus.TXRate =
				    (uint16_t) ((float)(tx_bytes * 1000) /
						SYSTEM_UPDATE_PERIOD_MS);
				rfm22bStatus.RXRate =
				    (uint16_t) ((float)(rx_bytes * 1000) /
						SYSTEM_UPDATE_PERIOD_MS);
				prev_tx_count = tx_count;
				prev_rx_count = rx_count;
			}
			rfm22bStatus.LinkState = radio_stats.link_state;
		} else {
			rfm22bStatus.LinkState =
			    RFM22BSTATUS_LINKSTATE_DISABLED;
		}

		// Update the object
		RFM22BStatusSet(&rfm22bStatus);

		// Wait until next period
		PIOS_Thread_Sleep_Until(&lastSysTime, SYSTEM_UPDATE_PERIOD_MS);
	}

}

/**
 * Called by the RTOS when the CPU is idle, used to measure the CPU idle time.
 */
void vApplicationIdleHook(void)
{
}

/**
 * Called by the RTOS when a stack overflow is detected.
 */
#define DEBUG_STACK_OVERFLOW 0
void vApplicationStackOverflowHook(uintptr_t pxTask, signed char * pcTaskName)
{
	stackOverflow = true;
#if DEBUG_STACK_OVERFLOW
	static volatile bool wait_here = true;
	while (wait_here) {
		;
	}
	wait_here = true;
#endif
}

/**
 * Called by the RTOS when a malloc call fails.
 */
#define DEBUG_MALLOC_FAILURES 0
void vApplicationMallocFailedHook(void)
{
	mallocFailed = true;
#if DEBUG_MALLOC_FAILURES
	static volatile bool wait_here = true;
	while (wait_here) {
		;
	}
	wait_here = true;
#endif
}

/**
 * @}
 * @}
 */
