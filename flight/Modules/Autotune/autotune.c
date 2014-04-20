/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup AutotuningModule Autotuning Module
 * @{
 *
 * @file       autotune.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @brief      State machine to run autotuning. Low level work done by @ref
 *             StabilizationModule 
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 ******************************************************************************/
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
#include "pios.h"
#include "physical_constants.h"
#include "flightstatus.h"
#include "modulesettings.h"
#include "manualcontrolcommand.h"
#include "manualcontrolsettings.h"
#include "gyros.h"
#include "actuatordesired.h"
#include "relaytuning.h"
#include "relaytuningsettings.h"
#include "stabilizationdesired.h"
#include "stabilizationsettings.h"
#include <pios_board_info.h>
 
// Private constants
#define STACK_SIZE_BYTES 2500
#define TASK_PRIORITY (tskIDLE_PRIORITY+2)

#define AF_NUMX 13

// Private types
enum AUTOTUNE_STATE {AT_INIT, AT_START, AT_RUN, AT_FINISHED, AT_SET};

// Private variables
static xTaskHandle taskHandle;
static bool module_enabled;

// Private functions
static void AutotuneTask(void *parameters);
static void update_stabilization_settings();
static void af_predict(float X[AF_NUMX], float P[AF_NUMX][AF_NUMX], float u_in[3], float gyro[3]);
static void af_init(float X[AF_NUMX], float P[AF_NUMX][AF_NUMX]);

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t AutotuneInitialize(void)
{
	// Create a queue, connect to manual control command and flightstatus
#ifdef MODULE_Autotune_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_AUTOTUNE] == MODULESETTINGS_ADMINSTATE_ENABLED)
		module_enabled = true;
	else
		module_enabled = false;
#endif

	if (module_enabled) {
		RelayTuningSettingsInitialize();
		RelayTuningInitialize();
	}

	return 0;
}

/**
 * Initialise the module, called on startup
 * \returns 0 on success or -1 if initialisation failed
 */
int32_t AutotuneStart(void)
{
	// Start main task if it is enabled
	if(module_enabled) {
		xTaskCreate(AutotuneTask, (signed char *)"Autotune", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);

		TaskMonitorAdd(TASKINFO_RUNNING_AUTOTUNE, taskHandle);
		PIOS_WDG_RegisterFlag(PIOS_WDG_AUTOTUNE);
	}
	return 0;
}

MODULE_INITCALL(AutotuneInitialize, AutotuneStart)

/**
 * Module thread, should not return.
 */
static void AutotuneTask(void *parameters)
{
	//AlarmsClear(SYSTEMALARMS_ALARM_ATTITUDE);
	
	enum AUTOTUNE_STATE state = AT_INIT;

	portTickType lastUpdateTime = xTaskGetTickCount();

	float X[AF_NUMX] = {0};
	float P[AF_NUMX][AF_NUMX] = {{0}};

	af_init(X,P);

	while(1) {

		PIOS_WDG_UpdateFlag(PIOS_WDG_AUTOTUNE);
		// TODO:
		// 1. get from queue
		// 2. based on whether it is flightstatus or manualcontrol

		portTickType diffTime;

		const uint32_t PREPARE_TIME = 2000;
		const uint32_t MEAURE_TIME = 30000;

		FlightStatusData flightStatus;
		FlightStatusGet(&flightStatus);

		// Only allow this module to run when autotuning
		if (flightStatus.FlightMode != FLIGHTSTATUS_FLIGHTMODE_AUTOTUNE) {
			state = AT_INIT;
			vTaskDelay(50);
			continue;
		}

		StabilizationDesiredData stabDesired;
		StabilizationDesiredGet(&stabDesired);

		StabilizationSettingsData stabSettings;
		StabilizationSettingsGet(&stabSettings);

		ManualControlSettingsData manualSettings;
		ManualControlSettingsGet(&manualSettings);

		ManualControlCommandData manualControl;
		ManualControlCommandGet(&manualControl);

		RelayTuningSettingsData relaySettings;
		RelayTuningSettingsGet(&relaySettings);

		bool rate = relaySettings.Mode == RELAYTUNINGSETTINGS_MODE_RATE;

		if (rate) { // rate mode
			stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL]  = STABILIZATIONDESIRED_STABILIZATIONMODE_RATE;
			stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_RATE;

			stabDesired.Roll = manualControl.Roll * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_ROLL];
			stabDesired.Pitch = manualControl.Pitch * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_PITCH];
		} else {
			stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL]  = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;
			stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE;

			stabDesired.Roll = manualControl.Roll * stabSettings.RollMax;
			stabDesired.Pitch = manualControl.Pitch * stabSettings.PitchMax;
		}

		stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_RATE;
		stabDesired.Yaw = manualControl.Yaw * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_YAW];
		stabDesired.Throttle = manualControl.Throttle;

		switch(state) {
			case AT_INIT:

				lastUpdateTime = xTaskGetTickCount();

				af_init(X,P);

				RelayTuningData relay;
				relay.Beta[RELAYTUNING_BETA_ROLL]   = X[6];
				relay.Beta[RELAYTUNING_BETA_PITCH]  = X[7];
				relay.Beta[RELAYTUNING_BETA_YAW]    = X[8];
				relay.Bias[RELAYTUNING_BIAS_ROLL]   = X[10];
				relay.Bias[RELAYTUNING_BIAS_PITCH]  = X[11];
				relay.Bias[RELAYTUNING_BIAS_YAW]    = X[12];
				relay.Tau                           = X[9];
				RelayTuningSet(&relay);

				// Only start when armed and flying
				if (flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED && stabDesired.Throttle > 0)
					state = AT_START;
				break;

			case AT_START:

				diffTime = xTaskGetTickCount() - lastUpdateTime;

				// Spend the first block of time in normal rate mode to get airborne
				if (diffTime > PREPARE_TIME) {
					state = AT_RUN;
					lastUpdateTime = xTaskGetTickCount();
				}
				break;

			case AT_RUN:

				diffTime = xTaskGetTickCount() - lastUpdateTime;

				// Run relay mode on the roll axis for the measurement time
				stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL] = rate ? STABILIZATIONDESIRED_STABILIZATIONMODE_RELAYRATE :
					STABILIZATIONDESIRED_STABILIZATIONMODE_RELAYATTITUDE;

				// Run relay mode on the pitch axis for the measurement time
				stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = rate ? STABILIZATIONDESIRED_STABILIZATIONMODE_RELAYRATE :
					STABILIZATIONDESIRED_STABILIZATIONMODE_RELAYATTITUDE;

				stabDesired.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW] = STABILIZATIONDESIRED_STABILIZATIONMODE_RELAYRATE;

				// Update the system identification
				{
					GyrosData gyros;
					GyrosGet(&gyros);

					ActuatorDesiredData desired;
					ActuatorDesiredGet(&desired);

					float y[3] = {gyros.x, gyros.y, gyros.z};
					float u[3] = {desired.Roll, desired.Pitch, desired.Yaw};

					af_predict(X,P,u,y);

					RelayTuningData relay;
					relay.Beta[RELAYTUNING_BETA_ROLL]   = X[6];
					relay.Beta[RELAYTUNING_BETA_PITCH]  = X[7];
					relay.Beta[RELAYTUNING_BETA_YAW]    = X[8];
					relay.Bias[RELAYTUNING_BIAS_ROLL]   = X[10];
					relay.Bias[RELAYTUNING_BIAS_PITCH]  = X[11];
					relay.Bias[RELAYTUNING_BIAS_YAW]    = X[12];
					relay.Tau                           = X[9];
					RelayTuningSet(&relay);
				}

				if (diffTime > MEAURE_TIME) { // Move on to next state
					state = AT_FINISHED;
					lastUpdateTime = xTaskGetTickCount();
				}

				break;

			case AT_FINISHED:

				// Wait until disarmed and landed before updating the settings
				if (flightStatus.Armed == FLIGHTSTATUS_ARMED_DISARMED && stabDesired.Throttle <= 0)
					state = AT_SET;

				break;

			case AT_SET:
				update_stabilization_settings();
				state = AT_INIT;
				break;

			default:
				// Set an alarm or some shit like that
				break;
		}

		StabilizationDesiredSet(&stabDesired);

		vTaskDelay(3);
	}
}

/**
 * Called after measuring roll and pitch to update the
 * stabilization settings
 * 
 * takes in @ref RelayTuning and outputs @ref StabilizationSettings
 */
static void update_stabilization_settings()
{
	RelayTuningData relayTuning;
	RelayTuningGet(&relayTuning);

	RelayTuningSettingsData relaySettings;
	RelayTuningSettingsGet(&relaySettings);

	StabilizationSettingsData stabSettings;
	StabilizationSettingsGet(&stabSettings);

	switch(relaySettings.Behavior) {
		case RELAYTUNINGSETTINGS_BEHAVIOR_MEASURE:
			// Just measure, don't update the stab settings
			break;
		case RELAYTUNINGSETTINGS_BEHAVIOR_COMPUTE:
			StabilizationSettingsSet(&stabSettings);
			break;
		case RELAYTUNINGSETTINGS_BEHAVIOR_SAVE:
			StabilizationSettingsSet(&stabSettings);
			UAVObjSave(StabilizationSettingsHandle(), 0);
			break;
	}
	
}


/**
 * Prediction step for EKF on control inputs to quad that
 * learns the system properties
 * @param X the current state estimate which is updated in place
 * @param P the current covariance matrix, updated in place
 * @param[in] the current control inputs (roll, pitch, yaw)
 * @param[in] the gyro measurements
 */
/**
 * Prediction step for EKF on control inputs to quad that
 * learns the system properties
 * @param X the current state estimate which is updated in place
 * @param P the current covariance matrix, updated in place
 * @param[in] the current control inputs (roll, pitch, yaw)
 * @param[in] the gyro measurements
 */
static void af_predict(float X[AF_NUMX], float P[AF_NUMX][AF_NUMX], float u_in[3], float gyro[3])
{

	float Ts = 1.0f / 666.0f;
	float Tsq = Ts * Ts;
	float Tsq3 = Tsq * Ts;
    float Tsq4 = Tsq * Tsq;

	// for convenience and clarity code below uses the named versions of
	// the state variables
	float w1 = X[0];           // roll rate estimate
	float w2 = X[1];           // pitch rate estimate
	float w3 = X[2];           // yaw rate estimate
	float u1 = X[3];           // scaled roll torque 
	float u2 = X[4];           // scaled pitch torque
	float u3 = X[5];           // scaled yaw torque
	const float e_b1 = expf(X[6]);   // roll torque scale
    const float b1 = X[6];
	const float e_b2 = expf(X[7]);   // pitch torque scale
    const float b2 = X[7];
	const float e_b3 = expf(X[8]);   // yaw torque scale
    const float b3 = X[8];
	const float e_tau = expf(X[9]); // time response of the motors
    const float tau = X[9];
	const float bias1 = X[10];        // bias in the roll torque
	const float bias2 = X[11];       // bias in the pitch torque
	const float bias3 = X[12];       // bias in the yaw torque

	// inputs to the system (roll, pitch, yaw)
	const float u1_in = u_in[0];
	const float u2_in = u_in[1];
	const float u3_in = u_in[2];

	// measurements from gyro
	const float gyro_x = gyro[0];
	const float gyro_y = gyro[1];
	const float gyro_z = gyro[2];

	// update named variables because we want to use predicted
	// values below
	w1 = X[0] = w1 + Ts*e_b1*(u1 - bias1);
	w2 = X[1] = w2 + Ts*e_b2*(u2 - bias2);
	w3 = X[2] = w3 + Ts*e_b3*(u3 - bias3);
	u1 = X[3] = (Ts * u1_in)/(Ts + e_tau) + (u1*e_tau)/(Ts + e_tau);
	u2 = X[4] = (Ts * u2_in)/(Ts + e_tau) + (u2*e_tau)/(Ts + e_tau);
    u3 = X[5] = (Ts*u3_in)/(Ts + e_tau) + (u3*e_tau)/(Ts + e_tau);
    // X[6] to X[12] unchanged

	/**** filter parameters ****/
	const float q_w = 1e-4f;
	const float q_ud = 1e-4f;
	const float q_B = 1e-7f;
	const float q_tau = 1e-7f;
	const float q_bias = 1e-19f;
	const float s_a = 1000.0f; // expected gyro noise

	const float Q[AF_NUMX] = {q_w, q_w, q_w, q_ud, q_ud, q_ud, q_B, q_B, q_B, q_tau, q_bias, q_bias, q_bias};

	float D[AF_NUMX][AF_NUMX];
	for (uint32_t i = 0; i < AF_NUMX; i++)
		for (uint32_t j = 0; j < AF_NUMX; j++)
			D[i][j] = P[i][j];

    const float e_tau2    = e_tau * e_tau;
    const float e_tau3    = e_tau * e_tau2;
    const float e_tau4    = e_tau2 * e_tau2;
    const float Ts_e_tau2 = (Ts + e_tau) * (Ts + e_tau);
    const float Ts_e_tau4 = Ts_e_tau2 * Ts_e_tau2;

	// covariance propagation - D is stored copy of covariance	
	P[0][0] = D[0][0] + Q[0] + 2*Ts*e_b1*(D[0][3] - D[0][10] - D[0][6]*bias1 + D[0][6]*u1) + Tsq*(e_b1*e_b1)*(D[3][3] - 2*D[3][10] + D[10][10] - 2*D[3][6]*bias1 + 2*D[6][10]*bias1 + 2*D[3][6]*u1 - 2*D[6][10]*u1 + D[6][6]*(bias1*bias1) + D[6][6]*(u1*u1) - 2*D[6][6]*bias1*u1);
	P[0][1] = P[1][0] = 0;
	P[0][2] = P[2][0] = 0;
	P[0][3] = P[3][0] = (D[0][3]*e_tau2 + D[0][3]*Ts*e_tau + D[3][3]*Ts*(e_b1*e_tau2) - D[3][10]*Ts*(e_b1*e_tau2) + D[3][3]*Tsq*(e_b1*e_tau) - D[3][10]*Tsq*(e_b1*e_tau) + D[0][9]*Ts*u1*e_tau - D[0][9]*Ts*u1_in*e_tau - D[3][6]*Ts*bias1*(e_b1*e_tau2) - D[3][6]*Tsq*bias1*(e_b1*e_tau) + D[3][6]*Ts*u1*(e_b1*e_tau2) + D[3][6]*Tsq*u1*(e_b1*e_tau) + D[3][9]*Tsq*u1*(e_b1*e_tau) - D[9][10]*Tsq*u1*(e_b1*e_tau) - D[3][9]*Tsq*u1_in*(e_b1*e_tau) + D[9][10]*Tsq*u1_in*(e_b1*e_tau) + D[6][9]*Tsq*(u1*u1)*(e_b1*e_tau) - D[6][9]*Tsq*bias1*u1*(e_b1*e_tau) + D[6][9]*Tsq*bias1*u1_in*(e_b1*e_tau) - D[6][9]*Tsq*u1*u1_in*(e_b1*e_tau))/Ts_e_tau2;
	P[0][4] = P[4][0] = 0;
	P[0][5] = P[5][0] = 0;
	P[0][6] = P[6][0] = D[0][6] - Ts*(D[6][10]*e_b1 - D[3][6]*e_b1 + D[6][6]*e_b1*(bias1 - u1));
	P[0][7] = P[7][0] = 0;
	P[0][8] = P[8][0] = 0;
	P[0][9] = P[9][0] = D[0][9] - Ts*(D[9][10]*e_b1 - D[3][9]*e_b1 + D[6][9]*e_b1*(bias1 - u1));
	P[0][10] = P[10][0] = D[0][10] - Ts*(D[10][10]*e_b1 - D[3][10]*e_b1 + D[6][10]*e_b1*(bias1 - u1));
	P[0][11] = P[11][0] = 0;
	P[0][12] = P[12][0] = 0;
	P[1][1] = D[1][1] + Q[1] + 2*Ts*e_b2*(D[1][4] - D[1][11] - D[1][7]*bias2 + D[1][7]*u2) + Tsq*(e_b2*e_b2)*(D[4][4] - 2*D[4][11] + D[11][11] - 2*D[4][7]*bias2 + 2*D[7][11]*bias2 + 2*D[4][7]*u2 - 2*D[7][11]*u2 + D[7][7]*(bias2*bias2) + D[7][7]*(u2*u2) - 2*D[7][7]*bias2*u2);
	P[1][2] = P[2][1] = 0;
	P[1][3] = P[3][1] = 0;
	P[1][4] = P[4][1] = (D[1][4]*e_tau2 + D[1][4]*Ts*e_tau + D[4][4]*Ts*(e_b2*e_tau2) - D[4][11]*Ts*(e_b2*e_tau2) + D[4][4]*Tsq*(e_b2*e_tau) - D[4][11]*Tsq*(e_b2*e_tau) + D[1][9]*Ts*u2*e_tau - D[1][9]*Ts*u2_in*e_tau - D[4][7]*Ts*bias2*(e_b2*e_tau2) - D[4][7]*Tsq*bias2*(e_b2*e_tau) + D[4][7]*Ts*u2*(e_b2*e_tau2) + D[4][7]*Tsq*u2*(e_b2*e_tau) + D[4][9]*Tsq*u2*(e_b2*e_tau) - D[9][11]*Tsq*u2*(e_b2*e_tau) - D[4][9]*Tsq*u2_in*(e_b2*e_tau) + D[9][11]*Tsq*u2_in*(e_b2*e_tau) + D[7][9]*Tsq*(u2*u2)*(e_b2*e_tau) - D[7][9]*Tsq*bias2*u2*(e_b2*e_tau) + D[7][9]*Tsq*bias2*u2_in*(e_b2*e_tau) - D[7][9]*Tsq*u2*u2_in*(e_b2*e_tau))/Ts_e_tau2;
	P[1][5] = P[5][1] = 0;
	P[1][6] = P[6][1] = 0;
	P[1][7] = P[7][1] = D[1][7] - Ts*(D[7][11]*e_b2 - D[4][7]*e_b2 + D[7][7]*e_b2*(bias2 - u2));
	P[1][8] = P[8][1] = 0;
	P[1][9] = P[9][1] = D[1][9] - Ts*(D[9][11]*e_b2 - D[4][9]*e_b2 + D[7][9]*e_b2*(bias2 - u2));
	P[1][10] = P[10][1] = 0;
	P[1][11] = P[11][1] = D[1][11] - Ts*(D[11][11]*e_b2 - D[4][11]*e_b2 + D[7][11]*e_b2*(bias2 - u2));
	P[1][12] = P[12][1] = 0;
	P[2][2] = D[2][2] + Q[2] + 2*Ts*e_b3*(D[2][5] - D[2][12] - D[2][8]*bias3 + D[2][8]*u3) + Tsq*(e_b3*e_b3)*(D[5][5] - 2*D[5][12] + D[12][12] - 2*D[5][8]*bias3 + 2*D[8][12]*bias3 + 2*D[5][8]*u3 - 2*D[8][12]*u3 + D[8][8]*(bias3*bias3) + D[8][8]*(u3*u3) - 2*D[8][8]*bias3*u3);
	P[2][3] = P[3][2] = 0;
	P[2][4] = P[4][2] = 0;
	P[2][5] = P[5][2] = (D[2][5]*e_tau2 + D[2][5]*Ts*e_tau + D[5][5]*Ts*(e_b3*e_tau2) - D[5][12]*Ts*(e_b3*e_tau2) + D[5][5]*Tsq*(e_b3*e_tau) - D[5][12]*Tsq*(e_b3*e_tau) + D[2][9]*Ts*u3*e_tau - D[2][9]*Ts*u3_in*e_tau - D[5][8]*Ts*bias3*(e_b3*e_tau2) - D[5][8]*Tsq*bias3*(e_b3*e_tau) + D[5][8]*Ts*u3*(e_b3*e_tau2) + D[5][8]*Tsq*u3*(e_b3*e_tau) + D[5][9]*Tsq*u3*(e_b3*e_tau) - D[9][12]*Tsq*u3*(e_b3*e_tau) - D[5][9]*Tsq*u3_in*(e_b3*e_tau) + D[9][12]*Tsq*u3_in*(e_b3*e_tau) + D[8][9]*Tsq*(u3*u3)*(e_b3*e_tau) - D[8][9]*Tsq*bias3*u3*(e_b3*e_tau) + D[8][9]*Tsq*bias3*u3_in*(e_b3*e_tau) - D[8][9]*Tsq*u3*u3_in*(e_b3*e_tau))/Ts_e_tau2;
	P[2][6] = P[6][2] = 0;
	P[2][7] = P[7][2] = 0;
	P[2][8] = P[8][2] = D[2][8] - Ts*(D[8][12]*e_b3 - D[5][8]*e_b3 + D[8][8]*e_b3*(bias3 - u3));
	P[2][9] = P[9][2] = D[2][9] - Ts*(D[9][12]*e_b3 - D[5][9]*e_b3 + D[8][9]*e_b3*(bias3 - u3));
	P[2][10] = P[10][2] = 0;
	P[2][11] = P[11][2] = 0;
	P[2][12] = P[12][2] = D[2][12] - Ts*(D[12][12]*e_b3 - D[5][12]*e_b3 + D[8][12]*e_b3*(bias3 - u3));
	P[3][3] = (Q[3]*Tsq4 + D[3][3]*e_tau4 + Q[3]*e_tau4 + 2*D[3][3]*Ts*e_tau3 + 4*Q[3]*Ts*e_tau3 + 4*Q[3]*Tsq3*e_tau + D[3][3]*Tsq*e_tau2 + 6*Q[3]*Tsq*e_tau2 + 2*D[3][9]*Tsq*u1*e_tau2 - 2*D[3][9]*Tsq*u1_in*e_tau2 + D[9][9]*Tsq*(u1*u1)*e_tau2 + D[9][9]*Tsq*(u1_in*u1_in)*e_tau2 + 2*D[3][9]*Ts*u1*e_tau3 - 2*D[3][9]*Ts*u1_in*e_tau3 - 2*D[9][9]*Tsq*u1*u1_in*e_tau2)/Ts_e_tau4;
	P[3][4] = P[4][3] = 0;
	P[3][5] = P[5][3] = 0;
	P[3][6] = P[6][3] = (e_tau*(D[3][6]*Ts + D[3][6]*e_tau + D[6][9]*Ts*u1 - D[6][9]*Ts*u1_in))/Ts_e_tau2;
	P[3][7] = P[7][3] = 0;
	P[3][8] = P[8][3] = 0;
	P[3][9] = P[9][3] = (e_tau*(D[3][9]*Ts + D[3][9]*e_tau + D[9][9]*Ts*u1 - D[9][9]*Ts*u1_in))/Ts_e_tau2;
	P[3][10] = P[10][3] = (e_tau*(D[3][10]*Ts + D[3][10]*e_tau + D[9][10]*Ts*u1 - D[9][10]*Ts*u1_in))/Ts_e_tau2;
	P[3][11] = P[11][3] = 0;
	P[3][12] = P[12][3] = 0;
	P[4][4] = (Q[4]*Tsq4 + D[4][4]*e_tau4 + Q[4]*e_tau4 + 2*D[4][4]*Ts*e_tau3 + 4*Q[4]*Ts*e_tau3 + 4*Q[4]*Tsq3*e_tau + D[4][4]*Tsq*e_tau2 + 6*Q[4]*Tsq*e_tau2 + 2*D[4][9]*Tsq*u2*e_tau2 - 2*D[4][9]*Tsq*u2_in*e_tau2 + D[9][9]*Tsq*(u2*u2)*e_tau2 + D[9][9]*Tsq*(u2_in*u2_in)*e_tau2 + 2*D[4][9]*Ts*u2*e_tau3 - 2*D[4][9]*Ts*u2_in*e_tau3 - 2*D[9][9]*Tsq*u2*u2_in*e_tau2)/Ts_e_tau4;
	P[4][5] = P[5][4] = 0;
	P[4][6] = P[6][4] = 0;
	P[4][7] = P[7][4] = (e_tau*(D[4][7]*Ts + D[4][7]*e_tau + D[7][9]*Ts*u2 - D[7][9]*Ts*u2_in))/Ts_e_tau2;
	P[4][8] = P[8][4] = 0;
	P[4][9] = P[9][4] = (e_tau*(D[4][9]*Ts + D[4][9]*e_tau + D[9][9]*Ts*u2 - D[9][9]*Ts*u2_in))/Ts_e_tau2;
	P[4][10] = P[10][4] = 0;
	P[4][11] = P[11][4] = (e_tau*(D[4][11]*Ts + D[4][11]*e_tau + D[9][11]*Ts*u2 - D[9][11]*Ts*u2_in))/Ts_e_tau2;
	P[4][12] = P[12][4] = 0;
	P[5][5] = (Q[5]*Tsq4 + D[5][5]*e_tau4 + Q[5]*e_tau4 + 2*D[5][5]*Ts*e_tau3 + 4*Q[5]*Ts*e_tau3 + 4*Q[5]*Tsq3*e_tau + D[5][5]*Tsq*e_tau2 + 6*Q[5]*Tsq*e_tau2 + 2*D[5][9]*Tsq*u3*e_tau2 - 2*D[5][9]*Tsq*u3_in*e_tau2 + D[9][9]*Tsq*(u3*u3)*e_tau2 + D[9][9]*Tsq*(u3_in*u3_in)*e_tau2 + 2*D[5][9]*Ts*u3*e_tau3 - 2*D[5][9]*Ts*u3_in*e_tau3 - 2*D[9][9]*Tsq*u3*u3_in*e_tau2)/Ts_e_tau4;
	P[5][6] = P[6][5] = 0;
	P[5][7] = P[7][5] = 0;
	P[5][8] = P[8][5] = (e_tau*(D[5][8]*Ts + D[5][8]*e_tau + D[8][9]*Ts*u3 - D[8][9]*Ts*u3_in))/Ts_e_tau2;
	P[5][9] = P[9][5] = (e_tau*(D[5][9]*Ts + D[5][9]*e_tau + D[9][9]*Ts*u3 - D[9][9]*Ts*u3_in))/Ts_e_tau2;
	P[5][10] = P[10][5] = 0;
	P[5][11] = P[11][5] = 0;
	P[5][12] = P[12][5] = (e_tau*(D[5][12]*Ts + D[5][12]*e_tau + D[9][12]*Ts*u3 - D[9][12]*Ts*u3_in))/Ts_e_tau2;
	P[6][6] = D[6][6] + Q[6];
	P[6][7] = P[7][6] = 0;
	P[6][8] = P[8][6] = 0;
	P[6][9] = P[9][6] = D[6][9];
	P[6][10] = P[10][6] = D[6][10];
	P[6][11] = P[11][6] = 0;
	P[6][12] = P[12][6] = 0;
	P[7][7] = D[7][7] + Q[7];
	P[7][8] = P[8][7] = 0;
	P[7][9] = P[9][7] = D[7][9];
	P[7][10] = P[10][7] = 0;
	P[7][11] = P[11][7] = D[7][11];
	P[7][12] = P[12][7] = 0;
	P[8][8] = D[8][8] + Q[8];
	P[8][9] = P[9][8] = D[8][9];
	P[8][10] = P[10][8] = 0;
	P[8][11] = P[11][8] = 0;
	P[8][12] = P[12][8] = D[8][12];
	P[9][9] = D[9][9] + Q[9];
	P[9][10] = P[10][9] = D[9][10];
	P[9][11] = P[11][9] = D[9][11];
	P[9][12] = P[12][9] = D[9][12];
	P[10][10] = D[10][10] + Q[10];
	P[10][11] = P[11][10] = 0;
	P[10][12] = P[12][10] = 0;
	P[11][11] = D[11][11] + Q[11];
	P[11][12] = P[12][11] = 0;
	P[12][12] = D[12][12] + Q[12];

    
	/********* this is the update part of the equation ***********/

    float S[3] = {P[0][0] + s_a, P[1][1] + s_a, P[2][2] + s_a};

	X[0] = w1 + (P[0][0]*(gyro_x - w1))/S[0];
	X[1] = w2 + (P[1][1]*(gyro_y - w2))/S[1];
	X[2] = w3 + (P[2][2]*(gyro_z - w3))/S[2];
	X[3] = u1 + (P[0][3]*(gyro_x - w1))/S[0];
	X[4] = u2 + (P[1][4]*(gyro_y - w2))/S[1];
	X[5] = u3 + (P[2][5]*(gyro_z - w3))/S[2];
	X[6] = b1 + (P[0][6]*(gyro_x - w1))/S[0];
	X[7] = b2 + (P[1][7]*(gyro_y - w2))/S[1];
	X[8] = b3 + (P[2][8]*(gyro_z - w3))/S[2];
	X[9] = tau + (P[0][9]*(gyro_x - w1))/S[0] + (P[1][9]*(gyro_y - w2))/S[1] + (P[2][9]*(gyro_z - w3))/S[2];
	X[10] = bias1 + (P[0][10]*(gyro_x - w1))/S[0];
	X[11] = bias2 + (P[1][11]*(gyro_y - w2))/S[1];
	X[12] = bias3 + (P[2][12]*(gyro_z - w3))/S[2];

	// update the duplicate cache
	for (uint32_t i = 0; i < AF_NUMX; i++)
		for (uint32_t j = 0; j < AF_NUMX; j++)
			D[i][j] = P[i][j];
    
	// This is an approximation that removes some cross axis uncertainty but
	// substantially reduces the number of calculations
	P[0][0] = -D[0][0]*(D[0][0]/S[0] - 1);
	P[0][1] = P[1][0] = 0;
	P[0][2] = P[2][0] = 0;
	P[0][3] = P[3][0] = -D[0][3]*(D[0][0]/S[0] - 1);
	P[0][4] = P[4][0] = 0;
	P[0][5] = P[5][0] = 0;
	P[0][6] = P[6][0] = -D[0][6]*(D[0][0]/S[0] - 1);
	P[0][7] = P[7][0] = 0;
	P[0][8] = P[8][0] = 0;
	P[0][9] = P[9][0] = -D[0][9]*(D[0][0]/S[0] - 1);
	P[0][10] = P[10][0] = -D[0][10]*(D[0][0]/S[0] - 1);
	P[0][11] = P[11][0] = 0;
	P[0][12] = P[12][0] = 0;
	P[1][1] = -D[1][1]*(D[1][1]/S[1] - 1);
	P[1][2] = P[2][1] = 0;
	P[1][3] = P[3][1] = 0;
	P[1][4] = P[4][1] = -D[1][4]*(D[1][1]/S[1] - 1);
	P[1][5] = P[5][1] = 0;
	P[1][6] = P[6][1] = 0;
	P[1][7] = P[7][1] = -D[1][7]*(D[1][1]/S[1] - 1);
	P[1][8] = P[8][1] = 0;
	P[1][9] = P[9][1] = -D[1][9]*(D[1][1]/S[1] - 1);
	P[1][10] = P[10][1] = 0;
	P[1][11] = P[11][1] = -D[1][11]*(D[1][1]/S[1] - 1);
	P[1][12] = P[12][1] = 0;
	P[2][2] = -D[2][2]*(D[2][2]/S[2] - 1);
	P[2][3] = P[3][2] = 0;
	P[2][4] = P[4][2] = 0;
	P[2][5] = P[5][2] = -D[2][5]*(D[2][2]/S[2] - 1);
	P[2][6] = P[6][2] = 0;
	P[2][7] = P[7][2] = 0;
	P[2][8] = P[8][2] = -D[2][8]*(D[2][2]/S[2] - 1);
	P[2][9] = P[9][2] = -D[2][9]*(D[2][2]/S[2] - 1);
	P[2][10] = P[10][2] = 0;
	P[2][11] = P[11][2] = 0;
	P[2][12] = P[12][2] = -D[2][12]*(D[2][2]/S[2] - 1);
	P[3][3] = D[3][3] - D[0][3]*D[0][3]/S[0];
	P[3][4] = P[4][3] = 0;
	P[3][5] = P[5][3] = 0;
	P[3][6] = P[6][3] = D[3][6] - (D[0][3]*D[0][6])/S[0];
	P[3][7] = P[7][3] = 0;
	P[3][8] = P[8][3] = 0;
	P[3][9] = P[9][3] = D[3][9] - (D[0][3]*D[0][9])/S[0];
	P[3][10] = P[10][3] = D[3][10] - (D[0][3]*D[0][10])/S[0];
	P[3][11] = P[11][3] = 0;
	P[3][12] = P[12][3] = 0;
	P[4][4] = D[4][4] - D[1][4]*D[1][4]/S[1];
	P[4][5] = P[5][4] = 0;
	P[4][6] = P[6][4] = 0;
	P[4][7] = P[7][4] = D[4][7] - (D[1][4]*D[1][7])/S[1];
	P[4][8] = P[8][4] = 0;
	P[4][9] = P[9][4] = D[4][9] - (D[1][4]*D[1][9])/S[1];
	P[4][10] = P[10][4] = 0;
	P[4][11] = P[11][4] = D[4][11] - (D[1][4]*D[1][11])/S[1];
	P[4][12] = P[12][4] = 0;
	P[5][5] = D[5][5] - D[2][5]*D[2][5]/S[2];
	P[5][6] = P[6][5] = 0;
	P[5][7] = P[7][5] = 0;
	P[5][8] = P[8][5] = D[5][8] - (D[2][5]*D[2][8])/S[2];
	P[5][9] = P[9][5] = D[5][9] - (D[2][5]*D[2][9])/S[2];
	P[5][10] = P[10][5] = 0;
	P[5][11] = P[11][5] = 0;
	P[5][12] = P[12][5] = D[5][12] - (D[2][5]*D[2][12])/S[2];
	P[6][6] = D[6][6] - D[0][6]*D[0][6]/S[0];
	P[6][7] = P[7][6] = 0;
	P[6][8] = P[8][6] = 0;
	P[6][9] = P[9][6] = D[6][9] - (D[0][6]*D[0][9])/S[0];
	P[6][10] = P[10][6] = D[6][10] - (D[0][6]*D[0][10])/S[0];
	P[6][11] = P[11][6] = 0;
	P[6][12] = P[12][6] = 0;
	P[7][7] = D[7][7] - D[1][7]*D[1][7]/S[1];
	P[7][8] = P[8][7] = 0;
	P[7][9] = P[9][7] = D[7][9] - (D[1][7]*D[1][9])/S[1];
	P[7][10] = P[10][7] = 0;
	P[7][11] = P[11][7] = D[7][11] - (D[1][7]*D[1][11])/S[1];
	P[7][12] = P[12][7] = 0;
	P[8][8] = D[8][8] - D[2][8]*D[2][8]/S[2];
	P[8][9] = P[9][8] = D[8][9] - (D[2][8]*D[2][9])/S[2];
	P[8][10] = P[10][8] = 0;
	P[8][11] = P[11][8] = 0;
	P[8][12] = P[12][8] = D[8][12] - (D[2][8]*D[2][12])/S[2];
	P[9][9] = D[9][9] - D[0][9]*D[0][9]/S[0] - D[1][9]*D[1][9]/S[1] - D[2][9]*D[2][9]/S[2];
	P[9][10] = P[10][9] = D[9][10] - (D[0][9]*D[0][10])/S[0];
	P[9][11] = P[11][9] = D[9][11] - (D[1][9]*D[1][11])/S[1];
	P[9][12] = P[12][9] = D[9][12] - (D[2][9]*D[2][12])/S[2];
	P[10][10] = D[10][10] - D[0][10]*D[0][10]/S[0];
	P[10][11] = P[11][10] = 0;
	P[10][12] = P[12][10] = 0;
	P[11][11] = D[11][11] - D[1][11]*D[1][11]/S[1];
	P[11][12] = P[12][11] = 0;
	P[12][12] = D[12][12] - D[2][12]*D[2][12]/S[2];
    
    // apply limits to some of the state variables
    if (X[9] > -1.5f)
        X[9] = -1.5f;
    if (X[9] < -5.0f)
        X[9] = -5.0f;
    // if (X[10] > 0.5f)
    //     X[10] = 0.5f;
    // if (X[10] < -0.5f)
    //     X[10] = -0.5f;
    // if (X[11] > 0.5f)
    //     X[11] = 0.5f;
    // if (X[11] < -0.5f)
    //     X[11] = -0.5f;
    // if (X[12] > 0.5f)
    //     X[12] = 0.5f;
    // if (X[12] < -0.5f)
    //     X[12] = -0.5f;
    
}

/**
 * Initialize the state variable and covariance matrix
 * for the system identification EKF
 */
static void af_init(float X[AF_NUMX], float P[AF_NUMX][AF_NUMX])
{
	X[0] = X[1] = X[2] = 0.0f;    // assume no rotation
	X[3] = X[4] = X[5] = 0.0f;    // and no net torque
	X[6] = X[7]        = 10.0f;   // medium amount of strength
    X[8]               = 3.0f;    // yaw
	X[9] = -2.0f;                 // and 50 ms time scale
	X[10] = X[11] = X[12] = 0.0f; // zero bias

	for (uint32_t i = 0; i < AF_NUMX; i++) {
		for (uint32_t j = 0; j < AF_NUMX; j++)
			P[i][j] = 0.0f;
	}

	P[0][0] = P[1][1] = P[2][2] = 1;
	P[3][3] = P[4][4] = P[5][5] = 1;
	P[6][6] = P[7][7] = P[8][8] = 0.05f;
	P[9][9] = 0.05f;
	P[10][10] = P[11][11] = P[12][12] = 0.05f;
}

/**
 * @}
 * @}
 */
