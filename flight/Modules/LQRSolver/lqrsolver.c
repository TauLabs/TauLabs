/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{ 
 * @addtogroup LQRSolver LQR Module
 * @{ 
 *
 * @file       lqrsolver.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2016
 * @brief      Solve LQR control matrix slowly in background
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
#include "pios_thread.h"
#include "pios_queue.h"

#include "rate_torque_lqr_optimize.h"

#include "lqrsettings.h"
#include "lqrsolution.h"
#include "systemident.h"

// Private constants
#define STACK_SIZE_BYTES 10000
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW

// Private types

// Private variables
static struct pios_queue *queue;

// Private functions
static void lqrSolverTask(void *parameters);
static void settings_updated_cb(UAVObjEvent * objEv);

// Local variables
static bool settings_updated;

/**
 * Initialise the Logging module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t LQRSolverInitialize(void)
{

	LqrSettingsInitialize();
	LQRSolutionInitialize();
	SystemIdentInitialize();

	return 0;
}

/**
 * Start the Logging module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t LQRSolverStart(void)
{
	queue = PIOS_Queue_Create(1, sizeof(UAVObjEvent));

	// Listen for updates.
	LqrSettingsConnectCallback(settings_updated_cb);
	SystemIdentConnectCallback(settings_updated_cb);
	settings_updated = true;

	// Start main task
	struct pios_thread *taskHandle = PIOS_Thread_Create(lqrSolverTask, "LQRSolver", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	TaskMonitorAdd(TASKINFO_RUNNING_LQRSOLVER, taskHandle);
	
	return 0;
}

MODULE_INITCALL(LQRSolverInitialize, LQRSolverStart);

static void lqrSolverTask(void *parameters)
{
	LqrSettingsData lqr_settings;
	LQRSolutionData lqr;
	SystemIdentData si;

	while(1) {

		if (settings_updated) {
			settings_updated = false;

			uintptr_t start_time = PIOS_Thread_Systime();

			SystemIdentGet(&si);
			LqrSettingsGet(&lqr_settings);
			LQRSolutionGet(&lqr);

			rtlqro_init(1.0f/400.0f);
			rtlqro_set_tau(si.Tau);
			rtlqro_set_gains(si.Beta);
			rtlqro_set_costs(lqr_settings.LqrRateParams[LQRSETTINGS_LQRRATEPARAMS_RATE],
				lqr_settings.LqrRateParams[LQRSETTINGS_LQRRATEPARAMS_TORQUE], 
				lqr_settings.LqrRateParams[LQRSETTINGS_LQRRATEPARAMS_INTEGRAL], 
				lqr_settings.LqrRateParams[LQRSETTINGS_LQRRATEPARAMS_ROLLPITCH], 
				lqr_settings.LqrRateParams[LQRSETTINGS_LQRRATEPARAMS_YAW]);

			rtlqro_solver();

			rtlqro_get_roll_gain(lqr.RollRate);
			rtlqro_get_pitch_gain(lqr.PitchRate);
			rtlqro_get_yaw_gain(lqr.YawRate);

			lqr.SolutionTime = (PIOS_Thread_Systime() - start_time);

			LQRSolutionSet(&lqr);
		}

		PIOS_Thread_Sleep(10);
	}
}

static void settings_updated_cb(UAVObjEvent * objEv)
{
	settings_updated = true;
}

/**
 * @}
 * @}
 */
