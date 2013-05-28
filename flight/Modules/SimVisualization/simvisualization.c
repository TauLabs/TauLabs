/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup VisualizationModule Visualization module for simulation
 * @{
 * @file       visualization.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Sends the state of the UAV out a UDP port to be visualized in a
 *             standalone application
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

#include "pios.h"
#include "physical_constants.h"
#include "openpilot.h"

#include "cameradesired.h"
#include "attitudesimulated.h"

#include <pthread.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <netinet/in.h>

// Private constants
#define STACK_SIZE_BYTES 1540
#define TASK_PRIORITY (tskIDLE_PRIORITY+2)
#define VISUALIZATION_PERIOD 20

// Private types

// Private variables
static xTaskHandle visualizationTaskHandle;

// Private functions
static void VisualizationTask(void *parameters);

/**
 * Initialise the module.  Called before the start function
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t SimVisualizationInitialize(void)
{

	return 0;
}

/**
 * Start the task.  Expects all objects to be initialized by this point.
 *pick \returns 0 on success or -1 if initialisation failed
 */
int32_t SimVisualizationStart(void)
{
	// Start main task
	xTaskCreate(VisualizationTask, (signed char *)"Visualization", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &visualizationTaskHandle);

	return 0;
}

MODULE_INITCALL(VisualizationInitialize, VisualizationStart)

struct uav_data {
	double q[4];
	double NED[3];
	double camera_roll;
	double camera_pitch;
};

/**
 * Pump the some information over UDP to a visualization
 */
static void VisualizationTask(void *parameters)
{
	int s;
	struct sockaddr_in server;

	s = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);

	memset(&server,0,sizeof(server));
  	server.sin_family = AF_INET;
	server.sin_addr.s_addr = inet_addr("127.0.0.1");
	server.sin_port = htons(3000);
	inet_aton("127.0.0.1", &server.sin_addr);

	struct uav_data uav_data;
	AttitudeSimulatedData simData;
	CameraDesiredData camera;

	while (1) {
		AttitudeSimulatedGet(&simData);
		CameraDesiredGet(&camera);

		uav_data.q[0] = simData.q1;
		uav_data.q[1] = simData.q2;
		uav_data.q[2] = simData.q3;
		uav_data.q[3] = simData.q4;
		uav_data.NED[0] = simData.Position[0];
		uav_data.NED[1] = simData.Position[1];
		uav_data.NED[2] = simData.Position[2];
		uav_data.camera_roll = camera.Roll * 30;
		uav_data.camera_pitch = camera.Pitch * 45;
		sendto(s, (struct sockaddr *) &uav_data, sizeof(uav_data), 0, (struct sockaddr *) &server, sizeof(server));
		usleep(100000);
		vTaskDelay(100);

	}
}

/**
  * @}
  * @}
  */
