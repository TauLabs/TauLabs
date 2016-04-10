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

// Private constants
#define STACK_SIZE_BYTES 3000
#define TASK_PRIORITY PIOS_THREAD_PRIO_LOW

// Private types

// Private variables

// Private functions
static void lqrSolverTask(void *parameters);

// Local variables


/**
 * Initialise the Logging module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t LQRSolverInitialize(void)
{

	return 0;
}

/**
 * Start the Logging module
 * \return -1 if initialisation failed
 * \return 0 on success
 */
int32_t LQRSolverStart(void)
{
	// Start main task
	struct pios_thread *taskHandle = PIOS_Thread_Create(lqrSolverTask, "LQRSolver", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	TaskMonitorAdd(TASKINFO_RUNNING_LQRSOLVER, taskHandle);
	
	return 0;
}

MODULE_INITCALL(LQRSolverInitialize, LQRSolverStart);

static void lqrSolverTask(void *parameters)
{
	static float g[4] = {9.67f, 9.84f, 5.2f, 8.29f};

	rtlqro_init(1.0f/400.0f);
	rtlqro_set_tau(-3.39f);
	rtlqro_set_gains(g);
	rtlqro_set_costs(10, 1, 10000, 1e4, 1e5);

	rtlqro_solver();
}

/**
 * @}
 * @}
 */
