/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup StabilizationModule Stabilization Module
 * @{
 * @brief      Control the UAV attitude to @ref StabilizationDesired
 *
 * The main control code which keeps the UAV at the attitude requested by
 * @ref StabilizationDesired.  This is done by comparing against 
 * @ref AttitudeActual to compute the error in roll pitch and yaw then
 * then based on the mode and values in @ref StabilizationSettings computing
 * the desired outputs and placing them in @ref ActuatorDesired.
 *
 * @file       stabilization.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @brief      Attitude stabilization.
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

#include "openpilot.h"
#include "stabilization.h"

#include "accels.h"
#include "actuatordesired.h"
#include "attitudeactual.h"
#include "cameradesired.h"
#include "flightstatus.h"
#include "gyros.h"
#include "ratedesired.h"
#include "stabilizationdesired.h"
#include "stabilizationsettings.h"
#include "trimangles.h"
#include "trimanglessettings.h"

// Math libraries
#include "coordinate_conversions.h"
#include "pid.h"
#include "sin_lookup.h"
#include "misc_math.h"

// Includes for various stabilization algorithms
#include "relay_tuning.h"
#include "virtualflybar.h"

// Private constants
#define MAX_QUEUE_SIZE 1

#if defined(PIOS_STABILIZATION_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_STABILIZATION_STACK_SIZE
#else
#define STACK_SIZE_BYTES 724
#endif

#define TASK_PRIORITY (tskIDLE_PRIORITY+4)
#define FAILSAFE_TIMEOUT_MS 30
#define COORDINATED_FLIGHT_MIN_ROLL_THRESHOLD 3.0f
#define COORDINATED_FLIGHT_MAX_YAW_THRESHOLD 0.05f

//! Set the stick position that maximally transitions to rate
#define HORIZON_MODE_MAX_BLEND               0.85f

enum {
	PID_RATE_ROLL,   // Rate controller settings
	PID_RATE_PITCH,
	PID_RATE_YAW,
	PID_ATT_ROLL,    // Attitude controller settings
	PID_ATT_PITCH,
	PID_ATT_YAW,
	PID_VBAR_ROLL,   // Virtual flybar settings
	PID_VBAR_PITCH,
	PID_VBAR_YAW,
	PID_COORDINATED_FLIGHT_YAW,
	PID_MAX
};


// Private variables
static xTaskHandle taskHandle;
static StabilizationSettingsData settings;
static TrimAnglesData trimAngles;
static xQueueHandle queue;
float gyro_alpha = 0;
float axis_lock_accum[3] = {0,0,0};
uint8_t max_axis_lock = 0;
uint8_t max_axislock_rate = 0;
float weak_leveling_kp = 0;
uint8_t weak_leveling_max = 0;
bool lowThrottleZeroIntegral;
float vbar_decay = 0.991f;
struct pid pids[PID_MAX];

// Private functions
static void stabilizationTask(void* parameters);
static void ZeroPids(void);
static void SettingsUpdatedCb(UAVObjEvent * ev);

/**
 * Module initialization
 */
int32_t StabilizationStart()
{
	// Initialize variables
	// Create object queue
	queue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));

	// Listen for updates.
	//	AttitudeActualConnectQueue(queue);
	GyrosConnectQueue(queue);
	
	// Connect settings callback
	StabilizationSettingsConnectCallback(SettingsUpdatedCb);
	TrimAnglesSettingsConnectCallback(SettingsUpdatedCb);

	// Start main task
	xTaskCreate(stabilizationTask, (signed char*)"Stabilization", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_STABILIZATION, taskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_STABILIZATION);
	return 0;
}

/**
 * Module initialization
 */
int32_t StabilizationInitialize()
{
	// Initialize variables
	StabilizationSettingsInitialize();
	ActuatorDesiredInitialize();
	TrimAnglesInitialize();
	TrimAnglesSettingsInitialize();
#if defined(RATEDESIRED_DIAGNOSTICS)
	RateDesiredInitialize();
#endif

	// Code required for relay tuning
	sin_lookup_initialize();

	return 0;
}

MODULE_INITCALL(StabilizationInitialize, StabilizationStart);

/**
 * Module task
 */
static void stabilizationTask(void* parameters)
{
	UAVObjEvent ev;
	
	uint32_t timeval = PIOS_DELAY_GetRaw();
	
	ActuatorDesiredData actuatorDesired;
	StabilizationDesiredData stabDesired;
	RateDesiredData rateDesired;
	AttitudeActualData attitudeActual;
	GyrosData gyrosData;
	FlightStatusData flightStatus;

	float *stabDesiredAxis = &stabDesired.Roll;
	float *actuatorDesiredAxis = &actuatorDesired.Roll;
	float *rateDesiredAxis = &rateDesired.Roll;
	float horizonRateFraction = 0.0f;

	// Force refresh of all settings immediately before entering main task loop
	SettingsUpdatedCb((UAVObjEvent *) NULL);
	
	// Main task loop
	ZeroPids();
	while(1) {
		float dT;
		
		PIOS_WDG_UpdateFlag(PIOS_WDG_STABILIZATION);
		
		// Wait until the AttitudeRaw object is updated, if a timeout then go to failsafe
		if ( xQueueReceive(queue, &ev, MS2TICKS(FAILSAFE_TIMEOUT_MS)) != pdTRUE )
		{
			AlarmsSet(SYSTEMALARMS_ALARM_STABILIZATION,SYSTEMALARMS_ALARM_WARNING);
			continue;
		}
		
		dT = PIOS_DELAY_DiffuS(timeval) * 1.0e-6f;
		timeval = PIOS_DELAY_GetRaw();
		
		FlightStatusGet(&flightStatus);
		StabilizationDesiredGet(&stabDesired);
		AttitudeActualGet(&attitudeActual);
		GyrosGet(&gyrosData);
		ActuatorDesiredGet(&actuatorDesired);
#if defined(RATEDESIRED_DIAGNOSTICS)
		RateDesiredGet(&rateDesired);
#endif

		struct TrimmedAttitudeSetpoint {
			float Roll;
			float Pitch;
			float Yaw;
		} trimmedAttitudeSetpoint;
		
		// Mux in level trim values, and saturate the trimmed attitude setpoint.
		trimmedAttitudeSetpoint.Roll = bound_sym(stabDesired.Roll + trimAngles.Roll, settings.RollMax);
		trimmedAttitudeSetpoint.Pitch = bound_sym(stabDesired.Pitch + trimAngles.Pitch, settings.PitchMax);
		trimmedAttitudeSetpoint.Yaw = stabDesired.Yaw;

		// For horizon mode we need to compute the desire attitude from an unscaled value and apply the
		// trim offset. Also track the stick with the most deflection to choose rate blending.
		horizonRateFraction = 0.0f;
		if (stabDesired.StabilizationMode[ROLL] == STABILIZATIONDESIRED_STABILIZATIONMODE_HORIZON) {
			trimmedAttitudeSetpoint.Roll = stabDesired.Roll * settings.RollMax;
			trimmedAttitudeSetpoint.Roll = bound_sym(stabDesired.Roll + trimAngles.Roll, settings.RollMax);
			horizonRateFraction = fabsf(stabDesired.Roll);
		}
		if (stabDesired.StabilizationMode[PITCH] == STABILIZATIONDESIRED_STABILIZATIONMODE_HORIZON) {
			trimmedAttitudeSetpoint.Pitch = stabDesired.Pitch * settings.PitchMax;
			trimmedAttitudeSetpoint.Pitch = bound_sym(stabDesired.Pitch + trimAngles.Pitch, settings.PitchMax);
			horizonRateFraction = MAX(horizonRateFraction, fabsf(stabDesired.Pitch));
		}
		if (stabDesired.StabilizationMode[YAW] == STABILIZATIONDESIRED_STABILIZATIONMODE_HORIZON) {
			trimmedAttitudeSetpoint.Yaw = stabDesired.Yaw * settings.YawMax;
			horizonRateFraction = MAX(horizonRateFraction, fabsf(stabDesired.Yaw));
		}

		// For weak leveling mode the attitude setpoint is the trim value (drifts back towards "0")
		if (stabDesired.StabilizationMode[ROLL] == STABILIZATIONDESIRED_STABILIZATIONMODE_WEAKLEVELING) {
			trimmedAttitudeSetpoint.Roll = trimAngles.Roll;
		}
		if (stabDesired.StabilizationMode[PITCH] == STABILIZATIONDESIRED_STABILIZATIONMODE_WEAKLEVELING) {
			trimmedAttitudeSetpoint.Pitch = trimAngles.Pitch;
		}
		if (stabDesired.StabilizationMode[YAW] == STABILIZATIONDESIRED_STABILIZATIONMODE_WEAKLEVELING) {
			trimmedAttitudeSetpoint.Yaw = 0;
		}

		// Note we divide by the maximum limit here so the fraction ranges from 0 to 1 depending on
		// how much is requested.
		horizonRateFraction = bound_sym(horizonRateFraction, HORIZON_MODE_MAX_BLEND) / HORIZON_MODE_MAX_BLEND;

		// Calculate the errors in each axis. The local error is used in the following modes:
		//  ATTITUDE, HORIZON, WEAKLEVELING, RELAYATTITUDE
		float local_attitude_error[3];
		local_attitude_error[0] = trimmedAttitudeSetpoint.Roll - attitudeActual.Roll;
		local_attitude_error[1] = trimmedAttitudeSetpoint.Pitch - attitudeActual.Pitch;
		local_attitude_error[2] = trimmedAttitudeSetpoint.Yaw - attitudeActual.Yaw;
		
		// Wrap yaw error to [-180,180]
		local_attitude_error[2] = circular_modulus_deg(local_attitude_error[2]);

		static float gyro_filtered[3];
		gyro_filtered[0] = gyro_filtered[0] * gyro_alpha + gyrosData.x * (1 - gyro_alpha);
		gyro_filtered[1] = gyro_filtered[1] * gyro_alpha + gyrosData.y * (1 - gyro_alpha);
		gyro_filtered[2] = gyro_filtered[2] * gyro_alpha + gyrosData.z * (1 - gyro_alpha);

		// A flag to track which stabilization mode each axis is in
		static uint8_t previous_mode[MAX_AXES] = {255,255,255};
		bool error = false;

		//Run the selected stabilization algorithm on each axis:
		for(uint8_t i=0; i< MAX_AXES; i++)
		{
			// Check whether this axis mode needs to be reinitialized
			bool reinit = (stabDesired.StabilizationMode[i] != previous_mode[i]);
			previous_mode[i] = stabDesired.StabilizationMode[i];

			// Apply the selected control law
			switch(stabDesired.StabilizationMode[i])
			{
				case STABILIZATIONDESIRED_STABILIZATIONMODE_RATE:
					if(reinit)
						pids[PID_RATE_ROLL + i].iAccumulator = 0;

					// Store to rate desired variable for storing to UAVO
					rateDesiredAxis[i] = bound_sym(stabDesiredAxis[i], settings.ManualRate[i]);

					// Compute the inner loop
					actuatorDesiredAxis[i] = pid_apply_setpoint(&pids[PID_RATE_ROLL + i],  rateDesiredAxis[i],  gyro_filtered[i], dT);
					actuatorDesiredAxis[i] = bound_sym(actuatorDesiredAxis[i],1.0f);

					break;

				case STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE:
					if(reinit) {
						pids[PID_ATT_ROLL + i].iAccumulator = 0;
						pids[PID_RATE_ROLL + i].iAccumulator = 0;
					}

					// Compute the outer loop
					rateDesiredAxis[i] = pid_apply(&pids[PID_ATT_ROLL + i], local_attitude_error[i], dT);
					rateDesiredAxis[i] = bound_sym(rateDesiredAxis[i], settings.MaximumRate[i]);

					// Compute the inner loop
					actuatorDesiredAxis[i] = pid_apply_setpoint(&pids[PID_RATE_ROLL + i],  rateDesiredAxis[i],  gyro_filtered[i], dT);
					actuatorDesiredAxis[i] = bound_sym(actuatorDesiredAxis[i],1.0f);

					break;

				case STABILIZATIONDESIRED_STABILIZATIONMODE_VIRTUALBAR:
					// Store for debugging output
					rateDesiredAxis[i] = stabDesiredAxis[i];

					// Run a virtual flybar stabilization algorithm on this axis
					stabilization_virtual_flybar(gyro_filtered[i], rateDesiredAxis[i], &actuatorDesiredAxis[i], dT, reinit, i, &pids[PID_VBAR_ROLL + i], &settings);

					break;
				case STABILIZATIONDESIRED_STABILIZATIONMODE_WEAKLEVELING:
				{
					if (reinit)
						pids[PID_RATE_ROLL + i].iAccumulator = 0;

					float weak_leveling = local_attitude_error[i] * weak_leveling_kp;
					weak_leveling = bound_sym(weak_leveling, weak_leveling_max);

					// Compute desired rate as input biased towards leveling
					rateDesiredAxis[i] = stabDesiredAxis[i] + weak_leveling;
					actuatorDesiredAxis[i] = pid_apply_setpoint(&pids[PID_RATE_ROLL + i],  rateDesiredAxis[i],  gyro_filtered[i], dT);
					actuatorDesiredAxis[i] = bound_sym(actuatorDesiredAxis[i],1.0f);

					break;
				}
				case STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK:
					if (reinit)
						pids[PID_RATE_ROLL + i].iAccumulator = 0;

					if(fabs(stabDesiredAxis[i]) > max_axislock_rate) {
						// While getting strong commands act like rate mode
						rateDesiredAxis[i] = stabDesiredAxis[i];
						axis_lock_accum[i] = 0;
					} else {
						// For weaker commands or no command simply attitude lock (almost) on no gyro change
						axis_lock_accum[i] += (stabDesiredAxis[i] - gyro_filtered[i]) * dT;
						axis_lock_accum[i] = bound_sym(axis_lock_accum[i], max_axis_lock);
						rateDesiredAxis[i] = pid_apply(&pids[PID_ATT_ROLL + i], axis_lock_accum[i], dT);
					}

					rateDesiredAxis[i] = bound_sym(rateDesiredAxis[i], settings.MaximumRate[i]);

					actuatorDesiredAxis[i] = pid_apply_setpoint(&pids[PID_RATE_ROLL + i],  rateDesiredAxis[i],  gyro_filtered[i], dT);
					actuatorDesiredAxis[i] = bound_sym(actuatorDesiredAxis[i],1.0f);

					break;

				case STABILIZATIONDESIRED_STABILIZATIONMODE_HORIZON:
					if(reinit) {
						pids[PID_RATE_ROLL + i].iAccumulator = 0;
					}

					// The unscaled input (-1,1)
					float *raw_input = &stabDesired.Roll;

					// Do not allow outer loop integral to wind up in this mode since the controller
					// is often disengaged.
					pids[PID_ATT_ROLL + i].iAccumulator = 0;

					// Compute the outer loop for the attitude control
					float rateDesiredAttitude = pid_apply(&pids[PID_ATT_ROLL + i], local_attitude_error[i], dT);
					// Compute the desire rate for a rate control
					float rateDesiredRate = expo3(raw_input[i], settings.RateExpo[i]) * settings.ManualRate[i];

					// Blend from one rate to another. The maximum of all stick positions is used for the
					// amount so that when one axis goes completely to rate the other one does too. This
					// prevents doing flips while one axis tries to stay in attitude mode.
					rateDesiredAxis[i] = rateDesiredAttitude * (1.0f-horizonRateFraction) + rateDesiredRate * horizonRateFraction;
					rateDesiredAxis[i] = bound_sym(rateDesiredAxis[i], settings.MaximumRate[i]);

					// Compute the inner loop
					actuatorDesiredAxis[i] = pid_apply_setpoint(&pids[PID_RATE_ROLL + i],  rateDesiredAxis[i],  gyro_filtered[i], dT);
					actuatorDesiredAxis[i] = bound_sym(actuatorDesiredAxis[i],1.0f);

					break;
				case STABILIZATIONDESIRED_STABILIZATIONMODE_RELAYRATE:
					// Store to rate desired variable for storing to UAVO
					rateDesiredAxis[i] = bound_sym(stabDesiredAxis[i], settings.ManualRate[i]);

					// Run the relay controller which also estimates the oscillation parameters
					stabilization_relay_rate(rateDesiredAxis[i] - gyro_filtered[i], &actuatorDesiredAxis[i], i, reinit);
					actuatorDesiredAxis[i] = bound_sym(actuatorDesiredAxis[i],1.0);
					
					break;
					
				case STABILIZATIONDESIRED_STABILIZATIONMODE_RELAYATTITUDE:
					if(reinit)
						pids[PID_ATT_ROLL + i].iAccumulator = 0;

					// Compute the outer loop like attitude mode
					rateDesiredAxis[i] = pid_apply(&pids[PID_ATT_ROLL + i], local_attitude_error[i], dT);
					rateDesiredAxis[i] = bound_sym(rateDesiredAxis[i], settings.MaximumRate[i]);

					// Run the relay controller which also estimates the oscillation parameters
					stabilization_relay_rate(rateDesiredAxis[i] - gyro_filtered[i], &actuatorDesiredAxis[i], i, reinit);
					actuatorDesiredAxis[i] = bound_sym(actuatorDesiredAxis[i],1.0);

					break;

				case STABILIZATIONDESIRED_STABILIZATIONMODE_COORDINATEDFLIGHT:
					switch (i) {
						case YAW:
							if (reinit) {
								pids[PID_COORDINATED_FLIGHT_YAW].iAccumulator = 0;
								pids[PID_RATE_YAW].iAccumulator = 0;
								axis_lock_accum[YAW] = 0;
							}

							//If we are not in roll attitude mode, trigger an error
							if (stabDesired.StabilizationMode[ROLL] != STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE)
							{
								error = true;
								break ;
							}

							if (fabsf(stabDesired.Yaw) < COORDINATED_FLIGHT_MAX_YAW_THRESHOLD) { //If yaw is within the deadband...
								if (fabsf(stabDesired.Roll) > COORDINATED_FLIGHT_MIN_ROLL_THRESHOLD) { // We're requesting more roll than the threshold
									float accelsDataY;
									AccelsyGet(&accelsDataY);

									//Reset integral if we have changed roll to opposite direction from rudder. This implies that we have changed desired turning direction.
									if ((stabDesired.Roll > 0 && actuatorDesiredAxis[YAW] < 0) ||
											(stabDesired.Roll < 0 && actuatorDesiredAxis[YAW] > 0)){
										pids[PID_COORDINATED_FLIGHT_YAW].iAccumulator = 0;
									}

									// Coordinate flight can simply be seen as ensuring that there is no lateral acceleration in the
									// body frame. As such, we use the (noisy) accelerometer data as our measurement. Ideally, at
									// some point in the future we will estimate acceleration and then we can use the estimated value
									// instead of the measured value.
									float errorSlip = -accelsDataY;

									float command = pid_apply(&pids[PID_COORDINATED_FLIGHT_YAW], errorSlip, dT);
									actuatorDesiredAxis[YAW] = bound_sym(command ,1.0);

									// Reset axis-lock integrals
									pids[PID_RATE_YAW].iAccumulator = 0;
									axis_lock_accum[YAW] = 0;
								} else if (fabsf(stabDesired.Roll) <= COORDINATED_FLIGHT_MIN_ROLL_THRESHOLD) { // We're requesting less roll than the threshold
									// Axis lock on no gyro change
									axis_lock_accum[YAW] += (0 - gyro_filtered[YAW]) * dT;

									rateDesiredAxis[YAW] = pid_apply(&pids[PID_ATT_YAW], axis_lock_accum[YAW], dT);
									rateDesiredAxis[YAW] = bound_sym(rateDesiredAxis[YAW], settings.MaximumRate[YAW]);

									actuatorDesiredAxis[YAW] = pid_apply_setpoint(&pids[PID_RATE_YAW],  rateDesiredAxis[YAW],  gyro_filtered[YAW], dT);
									actuatorDesiredAxis[YAW] = bound_sym(actuatorDesiredAxis[YAW],1.0f);

									// Reset coordinated-flight integral
									pids[PID_COORDINATED_FLIGHT_YAW].iAccumulator = 0;
								}
							} else { //... yaw is outside the deadband. Pass the manual input directly to the actuator.
								actuatorDesiredAxis[YAW] = bound_sym(stabDesiredAxis[YAW], 1.0);

								// Reset all integrals
								pids[PID_COORDINATED_FLIGHT_YAW].iAccumulator = 0;
								pids[PID_RATE_YAW].iAccumulator = 0;
								axis_lock_accum[YAW] = 0;
							}
							break;
						case ROLL:
						case PITCH:
						default:
							//Coordinated Flight has no effect in these modes. Trigger a configuration error.
							error = true;
							break;
					}

					break;

				case STABILIZATIONDESIRED_STABILIZATIONMODE_POI:
					// The sanity check enforces this is only selectable for Yaw
					// for a gimbal you can select pitch too.
					if(reinit) {
						pids[PID_ATT_ROLL + i].iAccumulator = 0;
						pids[PID_RATE_ROLL + i].iAccumulator = 0;
					}

					float error;
					float angle;
					if (CameraDesiredHandle()) {
						switch(i) {
						case PITCH:
							CameraDesiredDeclinationGet(&angle);
							error = circular_modulus_deg(angle - attitudeActual.Pitch);
							break;
						case YAW:
							CameraDesiredBearingGet(&angle);
							error = circular_modulus_deg(angle - attitudeActual.Yaw);
							break;
						default:
							error = true;
						}
					} else
						error = true;

					// Compute the outer loop
					rateDesiredAxis[i] = pid_apply(&pids[PID_ATT_ROLL + i], error, dT);
					rateDesiredAxis[i] = bound_sym(rateDesiredAxis[i], settings.PoiMaximumRate[i]);

					// Compute the inner loop
					actuatorDesiredAxis[i] = pid_apply_setpoint(&pids[PID_RATE_ROLL + i],  rateDesiredAxis[i],  gyro_filtered[i], dT);
					actuatorDesiredAxis[i] = bound_sym(actuatorDesiredAxis[i],1.0f);

					break;
				case STABILIZATIONDESIRED_STABILIZATIONMODE_NONE:
					actuatorDesiredAxis[i] = bound_sym(stabDesiredAxis[i],1.0f);
					break;
				default:
					error = true;
					break;
			}
		}

		if (settings.VbarPiroComp == STABILIZATIONSETTINGS_VBARPIROCOMP_TRUE)
			stabilization_virtual_flybar_pirocomp(gyro_filtered[2], dT);

#if defined(RATEDESIRED_DIAGNOSTICS)
		RateDesiredSet(&rateDesired);
#endif

		// Save dT
		actuatorDesired.UpdateTime = dT * 1000;
		actuatorDesired.Throttle = stabDesired.Throttle;

		if(flightStatus.FlightMode != FLIGHTSTATUS_FLIGHTMODE_MANUAL) {
			ActuatorDesiredSet(&actuatorDesired);
		} else {
			// Force all axes to reinitialize when engaged
			for(uint8_t i=0; i< MAX_AXES; i++)
				previous_mode[i] = 255;
		}

		if(flightStatus.Armed != FLIGHTSTATUS_ARMED_ARMED ||
		   (lowThrottleZeroIntegral && stabDesired.Throttle < 0))
		{
			// Force all axes to reinitialize when engaged
			for(uint8_t i=0; i< MAX_AXES; i++)
				previous_mode[i] = 255;
		}

		// Clear or set alarms.  Done like this to prevent toggling each cycle
		// and hammering system alarms
		if (error)
			AlarmsSet(SYSTEMALARMS_ALARM_STABILIZATION,SYSTEMALARMS_ALARM_ERROR);
		else
			AlarmsClear(SYSTEMALARMS_ALARM_STABILIZATION);
	}
}


/**
 * Clear the accumulators and derivatives for all the axes
 */
static void ZeroPids(void)
{
	for(uint32_t i = 0; i < PID_MAX; i++)
		pid_zero(&pids[i]);


	for(uint8_t i = 0; i < 3; i++)
		axis_lock_accum[i] = 0.0f;
}


static void SettingsUpdatedCb(UAVObjEvent * ev)
{
	if (ev == NULL || ev->obj == TrimAnglesSettingsHandle())
	{
		TrimAnglesSettingsData trimAnglesSettings;

		TrimAnglesGet(&trimAngles);
		TrimAnglesSettingsGet(&trimAnglesSettings);

		// Set the trim angles
		trimAngles.Roll = trimAnglesSettings.Roll;
		trimAngles.Pitch = trimAnglesSettings.Pitch;

		TrimAnglesSet(&trimAngles);
	}

	if (ev == NULL || ev->obj == StabilizationSettingsHandle())
	{
		StabilizationSettingsGet(&settings);
		// Set the roll rate PID constants
		pid_configure(&pids[PID_RATE_ROLL],
		              settings.RollRatePID[STABILIZATIONSETTINGS_ROLLRATEPID_KP],
		              settings.RollRatePID[STABILIZATIONSETTINGS_ROLLRATEPID_KI],
		              settings.RollRatePID[STABILIZATIONSETTINGS_ROLLRATEPID_KD],
		              settings.RollRatePID[STABILIZATIONSETTINGS_ROLLRATEPID_ILIMIT]);

		// Set the pitch rate PID constants
		pid_configure(&pids[PID_RATE_PITCH],
		              settings.PitchRatePID[STABILIZATIONSETTINGS_PITCHRATEPID_KP],
		              settings.PitchRatePID[STABILIZATIONSETTINGS_PITCHRATEPID_KI],
		              settings.PitchRatePID[STABILIZATIONSETTINGS_PITCHRATEPID_KD],
		              settings.PitchRatePID[STABILIZATIONSETTINGS_PITCHRATEPID_ILIMIT]);

		// Set the yaw rate PID constants
		pid_configure(&pids[PID_RATE_YAW],
		              settings.YawRatePID[STABILIZATIONSETTINGS_YAWRATEPID_KP],
		              settings.YawRatePID[STABILIZATIONSETTINGS_YAWRATEPID_KI],
		              settings.YawRatePID[STABILIZATIONSETTINGS_YAWRATEPID_KD],
		              settings.YawRatePID[STABILIZATIONSETTINGS_YAWRATEPID_ILIMIT]);

		// Set the roll attitude PI constants
		pid_configure(&pids[PID_ATT_ROLL],
		              settings.RollPI[STABILIZATIONSETTINGS_ROLLPI_KP],
		              settings.RollPI[STABILIZATIONSETTINGS_ROLLPI_KI], 0,
		              settings.RollPI[STABILIZATIONSETTINGS_ROLLPI_ILIMIT]);

		// Set the pitch attitude PI constants
		pid_configure(&pids[PID_ATT_PITCH],
		              settings.PitchPI[STABILIZATIONSETTINGS_PITCHPI_KP],
		              settings.PitchPI[STABILIZATIONSETTINGS_PITCHPI_KI], 0,
		              settings.PitchPI[STABILIZATIONSETTINGS_PITCHPI_ILIMIT]);

		// Set the yaw attitude PI constants
		pid_configure(&pids[PID_ATT_YAW],
		              settings.YawPI[STABILIZATIONSETTINGS_YAWPI_KP],
		              settings.YawPI[STABILIZATIONSETTINGS_YAWPI_KI], 0,
		              settings.YawPI[STABILIZATIONSETTINGS_YAWPI_ILIMIT]);

		// Set the vbar roll settings
		pid_configure(&pids[PID_VBAR_ROLL],
		              settings.VbarRollPID[STABILIZATIONSETTINGS_VBARROLLPID_KP],
		              settings.VbarRollPID[STABILIZATIONSETTINGS_VBARROLLPID_KI],
		              settings.VbarRollPID[STABILIZATIONSETTINGS_VBARROLLPID_KD],
		              0);

		// Set the vbar pitch settings
		pid_configure(&pids[PID_VBAR_PITCH],
		              settings.VbarPitchPID[STABILIZATIONSETTINGS_VBARPITCHPID_KP],
		              settings.VbarPitchPID[STABILIZATIONSETTINGS_VBARPITCHPID_KI],
		              settings.VbarPitchPID[STABILIZATIONSETTINGS_VBARPITCHPID_KD],
		              0);

		// Set the vbar yaw settings
		pid_configure(&pids[PID_VBAR_YAW],
		              settings.VbarYawPID[STABILIZATIONSETTINGS_VBARYAWPID_KP],
		              settings.VbarYawPID[STABILIZATIONSETTINGS_VBARYAWPID_KI],
		              settings.VbarYawPID[STABILIZATIONSETTINGS_VBARYAWPID_KD],
		              0);

		// Set the coordinated flight settings
		pid_configure(&pids[PID_COORDINATED_FLIGHT_YAW],
		              settings.CoordinatedFlightYawPI[STABILIZATIONSETTINGS_COORDINATEDFLIGHTYAWPI_KP],
		              settings.CoordinatedFlightYawPI[STABILIZATIONSETTINGS_COORDINATEDFLIGHTYAWPI_KI],
		              0, /* No derivative term */
		              settings.CoordinatedFlightYawPI[STABILIZATIONSETTINGS_COORDINATEDFLIGHTYAWPI_ILIMIT]);

		// Set up the derivative term
		pid_configure_derivative(settings.DerivativeCutoff, settings.DerivativeGamma);

		// Maximum deviation to accumulate for axis lock
		max_axis_lock = settings.MaxAxisLock;
		max_axislock_rate = settings.MaxAxisLockRate;

		// Settings for weak leveling
		weak_leveling_kp = settings.WeakLevelingKp;
		weak_leveling_max = settings.MaxWeakLevelingRate;

		// Whether to zero the PID integrals while throttle is low
		lowThrottleZeroIntegral = settings.LowThrottleZeroIntegral == STABILIZATIONSETTINGS_LOWTHROTTLEZEROINTEGRAL_TRUE;

		// The dT has some jitter iteration to iteration that we don't want to
		// make thie result unpredictable.  Still, it's nicer to specify the constant
		// based on a time (in ms) rather than a fixed multiplier.  The error between
		// update rates on OP (~300 Hz) and CC (~475 Hz) is negligible for this
		// calculation
		const float fakeDt = 0.0025f;
		if(settings.GyroTau < 0.0001f)
			gyro_alpha = 0;   // not trusting this to resolve to 0
		else
			gyro_alpha = expf(-fakeDt  / settings.GyroTau);

		// Compute time constant for vbar decay term based on a tau
		vbar_decay = expf(-fakeDt / settings.VbarTau);
	}
}


/**
 * @}
 * @}
 */

