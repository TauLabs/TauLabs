/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup ActuatorModule Actuator Module
 * @{
 * @brief      Take the values in @ref ActuatorDesired and mix to set the outputs
 *
 * This module ultimately controls the outputs.  The values from @ref ActuatorDesired
 * are combined based on the values in @ref MixerSettings and then scaled by the
 * values in @ref ActuatorSettings to create the output PWM times.
 *
 * @file       actuator.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2015
 * @brief      Actuator module. Drives the actuators (servos, motors etc).
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
#include "accessorydesired.h"
#include "actuatorsettings.h"
#include "systemsettings.h"
#include "actuatordesired.h"
#include "actuatorcommand.h"
#include "flightstatus.h"
#include "mixersettings.h"
#include "mixerstatus.h"
#include "cameradesired.h"
#include "manualcontrolcommand.h"
#include "pios_thread.h"
#include "pios_queue.h"
#include "misc_math.h"

// Private constants
#define MAX_QUEUE_SIZE 2

#if defined(PIOS_ACTUATOR_STACK_SIZE)
#define STACK_SIZE_BYTES PIOS_ACTUATOR_STACK_SIZE
#else
#define STACK_SIZE_BYTES 1312
#endif

#define TASK_PRIORITY PIOS_THREAD_PRIO_HIGHEST
#define FAILSAFE_TIMEOUT_MS 100
#define MAX_MIX_ACTUATORS ACTUATORCOMMAND_CHANNEL_NUMELEM
#define MULTIROTOR_MIXER_UPPER_BOUND 128

// Private types

// Private variables
static struct pios_queue *queue;
static struct pios_thread *taskHandle;

// used to inform the actuator thread that actuator / mixer settings are updated
static volatile bool settings_updated;

// The actual mixer settings data, pulled at the top of the actuator thread
static MixerSettingsData mixerSettings;

// Ditto, for the actuator settings.
static ActuatorSettingsData actuatorSettings;

// Private functions
static void actuator_task(void* parameters);
static float scale_channel(float value, int idx);
static void set_failsafe();
static float throt_curve(const float input, const float* curve, uint8_t num_points);
static float collective_curve(const float input, const float* curve, uint8_t num_points);
static bool set_channel(uint8_t mixer_channel, float value);
static void actuator_update_rate_if_changed(bool force_update);
static void settings_update_cb(UAVObjEvent * objEv, void *ctx, void *obj, int len);
float process_mixer(const int index, const float curve1, const float curve2,
		ActuatorDesiredData *desired);
static float mix_channel(int ct, ActuatorDesiredData *desired,
		float curve1, float curve2);

static MixerSettingsMixer1TypeOptions get_mixer_type(int idx);
static int8_t *get_mixer_vec(int idx);

/**
 * @brief Module initialization
 * @return 0
 */
int32_t ActuatorStart()
{
	// Start main task
	taskHandle = PIOS_Thread_Create(actuator_task, "Actuator", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
	TaskMonitorAdd(TASKINFO_RUNNING_ACTUATOR, taskHandle);
	PIOS_WDG_RegisterFlag(PIOS_WDG_ACTUATOR);

	return 0;
}

/**
 * @brief Module initialization
 * @return 0
 */
int32_t ActuatorInitialize()
{
	// Register for notification of changes to ActuatorSettings
	ActuatorSettingsInitialize();
	ActuatorSettingsConnectCallback(settings_update_cb);

	// Register for notification of changes to MixerSettings
	MixerSettingsInitialize();
	MixerSettingsConnectCallback(settings_update_cb);

	// Listen for ActuatorDesired updates (Primary input to this module)
	ActuatorDesiredInitialize();
	queue = PIOS_Queue_Create(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));
	ActuatorDesiredConnectQueue(queue);

	// Primary output of this module
	ActuatorCommandInitialize();

#if defined(MIXERSTATUS_DIAGNOSTICS)
	// UAVO only used for inspecting the internal status of the mixer during debug
	MixerStatusInitialize();
#endif

	return 0;
}
MODULE_INITCALL(ActuatorInitialize, ActuatorStart);

static float get_curve2_source(ActuatorDesiredData *desired, MixerSettingsCurve2SourceOptions source)
{
	float tmp;

	switch (source) {
	case MIXERSETTINGS_CURVE2SOURCE_THROTTLE:
		return desired->Throttle;
		break;
	case MIXERSETTINGS_CURVE2SOURCE_ROLL:
		return desired->Roll;
		break;
	case MIXERSETTINGS_CURVE2SOURCE_PITCH:
		return desired->Pitch;
		break;
	case MIXERSETTINGS_CURVE2SOURCE_YAW:
		return desired->Yaw;
		break;
	case MIXERSETTINGS_CURVE2SOURCE_COLLECTIVE:
		ManualControlCommandCollectiveGet(&tmp);
		return tmp;
		break;
	case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY0:
	case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY1:
	case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY2:
	case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY3:
	case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY4:
	case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY5:
		(void) 0;
		AccessoryDesiredData accessory;

		if (AccessoryDesiredInstGet(source - MIXERSETTINGS_CURVE2SOURCE_ACCESSORY0,&accessory) == 0)
			return accessory.AccessoryVal;
		return 0;
		break;
	}

	/* Can't get here */
	return 0;
}

/**
 * @brief Main Actuator module task
 *
 * Universal matrix based mixer for VTOL, helis and fixed wing.
 * Converts desired roll,pitch,yaw and throttle to servo/ESC outputs.
 *
 * Because of how the Throttle ranges from 0 to 1, the motors should too!
 *
 * Note this code depends on the UAVObjects for the mixers being all being the same
 * and in sequence. If you change the object definition, make sure you check the code!
 *
 * @return -1 if error, 0 if success
 */
static void actuator_task(void* parameters)
{
	uint32_t lastSysTime;
	float dT = 0.0f;

	ActuatorCommandData command;
	ActuatorDesiredData desired;
	MixerStatusData mixerStatus;
	FlightStatusData flightStatus;

	/* Read initial values of ActuatorSettings */
	settings_updated = true;

	// Main task loop
	lastSysTime = PIOS_Thread_Systime();

	bool rc = false;

	while (1) {
		if (settings_updated) {
			settings_updated = false;
			ActuatorSettingsGet(&actuatorSettings);
			actuator_update_rate_if_changed(false);
			MixerSettingsGet(&mixerSettings);
		}

		if (rc != true) {
			/* Update of ActuatorDesired timed out,
			 * or first iteration.  Go to failsafe */
			set_failsafe();
		}

		PIOS_WDG_UpdateFlag(PIOS_WDG_ACTUATOR);

		UAVObjEvent ev;

		// Wait until the ActuatorDesired object is updated
		rc = PIOS_Queue_Receive(queue, &ev, FAILSAFE_TIMEOUT_MS);

		/* If we timed out, go to top of loop, which sets failsafe
		 * and waits again. */
		if (rc != true) {
			continue;
		}

		// Check how long since last update
		uint32_t thisSysTime = PIOS_Thread_Systime();
		if (thisSysTime > lastSysTime) // reuse dt in case of wraparound
			dT = (thisSysTime - lastSysTime) / 1000.0f;
		lastSysTime = thisSysTime;

		FlightStatusGet(&flightStatus);
		ActuatorDesiredGet(&desired);
		ActuatorCommandGet(&command);

#if defined(MIXERSTATUS_DIAGNOSTICS)
		MixerStatusGet(&mixerStatus);
#endif
		int nMixers = 0;

		for (int ct = 0; ct < MAX_MIX_ACTUATORS; ct++) {
			if (get_mixer_type(ct) != MIXERSETTINGS_MIXER1TYPE_DISABLED) {
				nMixers++;
			}
		}
		if ((nMixers < 2) && !ActuatorCommandReadOnly()) { //Nothing can fly with less than two mixers.
			set_failsafe(); // So that channels like PWM buzzer keep working
			continue;
		}

		AlarmsClear(SYSTEMALARMS_ALARM_ACTUATOR);

		bool armed = flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED;
		bool positiveThrottle = desired.Throttle >= 0.00f;
		bool spinWhileArmed = actuatorSettings.MotorsSpinWhileArmed == ACTUATORSETTINGS_MOTORSSPINWHILEARMED_TRUE;

		float curve1 = throt_curve(desired.Throttle, mixerSettings.ThrottleCurve1, MIXERSETTINGS_THROTTLECURVE1_NUMELEM);

		//The source for the secondary curve is selectable
		float curve2 = collective_curve(
				get_curve2_source(&desired, mixerSettings.Curve2Source),
				mixerSettings.ThrottleCurve2,
				MIXERSETTINGS_THROTTLECURVE2_NUMELEM);

		float * status = (float *)&mixerStatus; //access status objects as an array of floats

		for (int ct = 0; ct < MAX_MIX_ACTUATORS; ct++) {
			status[ct] = mix_channel(ct, &desired, curve1, curve2);

			// Motors have additional protection for when to be on
			if (get_mixer_type(ct) == MIXERSETTINGS_MIXER1TYPE_MOTOR) {

				// If not armed or motors aren't meant to spin all the time
				if (!armed ||
						(!spinWhileArmed && !positiveThrottle)) {
					status[ct] = -1;  //force min throttle
				}
				// If armed meant to keep spinning,
				else if ((spinWhileArmed && !positiveThrottle) ||
						(status[ct] < 0) )
					status[ct] = 0;
			}

			command.Channel[ct] = scale_channel(status[ct], ct);
		}

		// Store update time
		command.UpdateTime = 1000.0f*dT;
		if (1000.0f*dT > command.MaxUpdateTime)
			command.MaxUpdateTime = 1000.0f*dT;

		// Update output object
		ActuatorCommandSet(&command);
		// Update in case read only (eg. during servo configuration)
		ActuatorCommandGet(&command);

#if defined(MIXERSTATUS_DIAGNOSTICS)
		MixerStatusSet(&mixerStatus);
#endif

		// Update servo outputs
		bool success = true;

		for (int n = 0; n < ACTUATORCOMMAND_CHANNEL_NUMELEM; ++n) {
			success &= set_channel(n, command.Channel[n]);
		}
#if defined(PIOS_INCLUDE_HPWM)
		PIOS_Servo_Update();
#endif

		if (!success) {
			command.NumFailedUpdates++;
			ActuatorCommandSet(&command);
			AlarmsSet(SYSTEMALARMS_ALARM_ACTUATOR, SYSTEMALARMS_ALARM_CRITICAL);
		}

	}
}

/**
 *Process mixing for one actuator
 */
float process_mixer(const int index, const float curve1, const float curve2,
		ActuatorDesiredData *desired)
{
	int8_t *vector = get_mixer_vec(index);
	float result = ((vector[MIXERSETTINGS_MIXER1VECTOR_THROTTLECURVE1] * curve1) +
			(vector[MIXERSETTINGS_MIXER1VECTOR_THROTTLECURVE2] * curve2) +
			(vector[MIXERSETTINGS_MIXER1VECTOR_ROLL] * desired->Roll) +
			(vector[MIXERSETTINGS_MIXER1VECTOR_PITCH] * desired->Pitch) +
			(vector[MIXERSETTINGS_MIXER1VECTOR_YAW] * desired->Yaw)) * (1.0f / MULTIROTOR_MIXER_UPPER_BOUND);

	return (result);
}

/**
 * Interpolate a throttle curve
 *
 * throttle curve assumes input is [0,1]
 * this means that the throttle channel neutral value is nearly the same as its min value
 * this is convenient for throttle, since the neutral value is used as a failsafe and would thus shut off the motor
 *
 * @param input the input value, in [0,1]
 * @param curve the array of points in the curve
 * @param num_points the number of points in the curve
 * @return the output value, in [0,1]
 */
static float throt_curve(float const input, float const * curve, uint8_t num_points)
{
	return linear_interpolate(input, curve, num_points, 0.0f, 1.0f);
}

/**
 * Interpolate a collective curve
 *
 * we need to accept input in [-1,1] so that the neutral point may be set arbitrarily within the typical channel input range, which is [-1,1]
 *
 * @param input The input value, in [-1,1]
 * @param curve Array of points in the curve
 * @param num_points Number of points in the curve
 * @return the output value, in [-1,1]
 */
static float collective_curve(float const input, float const * curve, uint8_t num_points)
{
	return linear_interpolate(input, curve, num_points, -1.0f, 1.0f);
}

/**
 * Convert channel from -1/+1 to servo pulse duration in microseconds
 */
static float scale_channel(float value, int idx)
{
	float max = actuatorSettings.ChannelMax[idx];
	float min = actuatorSettings.ChannelMin[idx];
	float neutral = actuatorSettings.ChannelNeutral[idx];

	float valueScaled;
	// Scale
	if (value >= 0.0f) {
		valueScaled = value*(max-neutral) + neutral;
	} else {
		valueScaled = value*(neutral-min) + neutral;
	}

	if (max>min) {
		if (valueScaled > max) valueScaled = max;
		if (valueScaled < min) valueScaled = min;
	} else {
		if (valueScaled < max) valueScaled = max;
		if (valueScaled > min) valueScaled = min;
	}

	return valueScaled;
}

static float channel_failsafe_value(int idx)
{
	switch (get_mixer_type(idx)) {
	case MIXERSETTINGS_MIXER1TYPE_MOTOR:
		return actuatorSettings.ChannelMin[idx];
	case MIXERSETTINGS_MIXER1TYPE_SERVO:
		return actuatorSettings.ChannelNeutral[idx];
	default:
		// TODO: is this actually right/safe?
		return 0;
	}

}

/**
 * Set actuator output to the neutral values (failsafe)
 */
static void set_failsafe()
{
	/* grab only the parts that we are going to use */
	float Channel[ACTUATORCOMMAND_CHANNEL_NUMELEM];
	ActuatorCommandChannelGet(Channel);

	// Reset ActuatorCommand to safe values
	for (int n = 0; n < ACTUATORCOMMAND_CHANNEL_NUMELEM; ++n) {
		Channel[n] = channel_failsafe_value(n);
	}

	// Set alarm
	AlarmsSet(SYSTEMALARMS_ALARM_ACTUATOR, SYSTEMALARMS_ALARM_CRITICAL);

	// Update servo outputs
	for (int n = 0; n < ACTUATORCOMMAND_CHANNEL_NUMELEM; ++n) {
		set_channel(n, Channel[n]);
	}
#if defined(PIOS_INCLUDE_HPWM) // TODO: this is actually about the synchronous updating and not resolution
	PIOS_Servo_Update();
#endif

	// Update output object's parts that we changed
	ActuatorCommandChannelSet(Channel);
}

static bool set_channel(uint8_t mixer_channel, float value)
#if defined(ARCH_POSIX) || defined(ARCH_WIN32)
{
	return true;
}
#else
{
	switch (actuatorSettings.ChannelType[mixer_channel]) {
	case ACTUATORSETTINGS_CHANNELTYPE_PWMALARM:
	case ACTUATORSETTINGS_CHANNELTYPE_ARMINGLED:
	case ACTUATORSETTINGS_CHANNELTYPE_INFOLED:
	{
		// This is for buzzers that take a PWM input

		static uint32_t currBuzzTune = 0;
		static uint32_t currBuzzTuneState;

		static uint32_t currArmingTune = 0;
		static uint32_t currArmingTuneState;

		static uint32_t currInfoTune = 0;
		static uint32_t currInfoTuneState;

		uint32_t newTune = 0;
		if (actuatorSettings.ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_PWMALARM) {

			// Decide what tune to play
			if (AlarmsGet(SYSTEMALARMS_ALARM_BATTERY) > SYSTEMALARMS_ALARM_WARNING) {
				newTune = 0b11110110110000;     // pause, short, short, short, long
			} else if (AlarmsGet(SYSTEMALARMS_ALARM_GPS) >= SYSTEMALARMS_ALARM_WARNING) {
				newTune = 0x80000000;                           // pause, short
			} else {
				newTune = 0;
			}

			// Do we need to change tune?
			if (newTune != currBuzzTune) {
				currBuzzTune = newTune;
				currBuzzTuneState = currBuzzTune;
			}
		} else {     // ACTUATORSETTINGS_CHANNELTYPE_ARMINGLED || ACTUATORSETTINGS_CHANNELTYPE_INFOLED
			uint8_t arming;
			FlightStatusArmedGet(&arming);
			//base idle tune
			newTune =  0x80000000;          // 0b1000...

			// Merge the error pattern for InfoLed
			if (actuatorSettings.ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_INFOLED) {
				if (AlarmsGet(SYSTEMALARMS_ALARM_BATTERY) > SYSTEMALARMS_ALARM_WARNING) {
					newTune |= 0b00000000001111111011111110000000;
				} else if (AlarmsGet(SYSTEMALARMS_ALARM_GPS) >= SYSTEMALARMS_ALARM_WARNING) {
					newTune |= 0b00000000000000110110110000000000;
				}
			}
			// fast double blink pattern if armed
			if (arming == FLIGHTSTATUS_ARMED_ARMED)
				newTune |= 0xA0000000;  // 0b101000...

			// Do we need to change tune?
			if (actuatorSettings.ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_ARMINGLED) {
				if (newTune != currArmingTune) {
					currArmingTune = newTune;
					// note: those are both updated so that Info and Arming are in sync if used simultaneously
					currArmingTuneState = currArmingTune;
					currInfoTuneState = currInfoTune;
				}
			} else {
				if (newTune != currInfoTune) {
					currInfoTune = newTune;
					currArmingTuneState = currArmingTune;
					currInfoTuneState = currInfoTune;
				}
			}
		}

		// Play tune
		bool buzzOn = false;
		static uint32_t lastSysTime = 0;
		uint32_t thisSysTime = PIOS_Thread_Systime();
		uint32_t dT = 0;

		// For now, only look at the battery alarm, because functions like AlarmsHasCritical() can block for some time; to be discussed
		if (currBuzzTune||currArmingTune||currInfoTune) {
			if (thisSysTime > lastSysTime)
				dT = thisSysTime - lastSysTime;
			if (actuatorSettings.ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_PWMALARM)
				buzzOn = (currBuzzTuneState&1);  // Buzz when the LS bit is 1
			else if (actuatorSettings.ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_ARMINGLED)
				buzzOn = (currArmingTuneState&1);
			else if (actuatorSettings.ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_INFOLED)
				buzzOn = (currInfoTuneState&1);

			if (dT > 80) {
				// Go to next bit in alarm_seq_state
				currArmingTuneState >>= 1;
				currInfoTuneState >>= 1;
				currBuzzTuneState >>= 1;

				if (currBuzzTuneState == 0)
					currBuzzTuneState = currBuzzTune;       // All done, re-start the tune
				if (currArmingTuneState == 0)
					currArmingTuneState = currArmingTune;
				if (currInfoTuneState == 0)
					currInfoTuneState = currInfoTune;
				lastSysTime = thisSysTime;
			}
		}
		PIOS_Servo_Set(mixer_channel,
					buzzOn ? actuatorSettings.ChannelMax[mixer_channel] : actuatorSettings.ChannelMin[mixer_channel]);
		return true;
	}
	case ACTUATORSETTINGS_CHANNELTYPE_PWM:
#if defined(PIOS_INCLUDE_HPWM)
		// The HPWM method will convert from us to the appropriate settings
		PIOS_Servo_Set(mixer_channel, value);
#else
		PIOS_Servo_Set(mixer_channel, value);
#endif
		return true;
	default:
		return false;
	}

	return false;

}
#endif

/**
 * @brief Update the servo update rate
 */
static void actuator_update_rate_if_changed(bool force_update)
{
	static uint16_t prevChannelUpdateFreq[ACTUATORSETTINGS_TIMERUPDATEFREQ_NUMELEM];

	// check if the any rate setting is changed
	if (force_update ||
			memcmp(prevChannelUpdateFreq,
				actuatorSettings.TimerUpdateFreq,
				sizeof(prevChannelUpdateFreq)) != 0) {
		/* Something has changed, apply the settings to HW */
		memcpy(prevChannelUpdateFreq,
				actuatorSettings.TimerUpdateFreq,
				sizeof(prevChannelUpdateFreq));
		PIOS_Servo_SetMode(actuatorSettings.TimerUpdateFreq, actuatorSettings.TimerPwmResolution,
				ACTUATORSETTINGS_TIMERPWMRESOLUTION_NUMELEM);
	}
}

static void settings_update_cb(UAVObjEvent * objEv, void *ctx, void *obj, int len)
{
	(void) ctx; (void) obj; (void) len;

	settings_updated = true;
}

static float mix_channel(int ct, ActuatorDesiredData *desired,
		float curve1, float curve2)
{
	MixerSettingsMixer1TypeOptions type = get_mixer_type(ct);

	switch (type) {
	case MIXERSETTINGS_MIXER1TYPE_DISABLED:
		// Set to minimum if disabled.  This is not the same as saying PWM pulse = 0 us
		return -1;
		break;

	case MIXERSETTINGS_MIXER1TYPE_SERVO:
		return process_mixer(ct, curve1, curve2, desired);
		break;

	case MIXERSETTINGS_MIXER1TYPE_MOTOR:
		(void) 0;               // nil statement
		float val = process_mixer(ct, curve1, curve2, desired);

		if (val > 0) {
			// Apply curve fitting, mapping the input to the propeller output.
			val = actuatorSettings.MotorInputOutputCurveFit[ACTUATORSETTINGS_MOTORINPUTOUTPUTCURVEFIT_A] *
					powf(val, actuatorSettings.MotorInputOutputCurveFit[ACTUATORSETTINGS_MOTORINPUTOUTPUTCURVEFIT_B]);
		} else {
			// Idle throttle
			val = 0.0f;
		}

		return val;
		break;
	// If an accessory channel is selected for direct bypass mode
	// In this configuration the accessory channel is scaled and
	// mapped directly to output.
	// Note: THERE IS NO SAFETY CHECK HERE FOR ARMING
	// these also will not be updated in failsafe mode.  I'm not sure
	// what the correct behavior is since it seems domain specific.
	// I don't love this code
	case MIXERSETTINGS_MIXER1TYPE_ACCESSORY0:
	case MIXERSETTINGS_MIXER1TYPE_ACCESSORY1:
	case MIXERSETTINGS_MIXER1TYPE_ACCESSORY2:
	case MIXERSETTINGS_MIXER1TYPE_ACCESSORY3:
	case MIXERSETTINGS_MIXER1TYPE_ACCESSORY4:
	case MIXERSETTINGS_MIXER1TYPE_ACCESSORY5:
		(void) 0;

		AccessoryDesiredData accessory;
		if (AccessoryDesiredInstGet(type - MIXERSETTINGS_MIXER1TYPE_ACCESSORY0,&accessory) == 0)
			return accessory.AccessoryVal;
		else
			return -1;
		break;
	case MIXERSETTINGS_MIXER1TYPE_CAMERAPITCH:
	case MIXERSETTINGS_MIXER1TYPE_CAMERAROLL:
	case MIXERSETTINGS_MIXER1TYPE_CAMERAYAW:
		(void) 0;

		CameraDesiredData cameraDesired;
		if (CameraDesiredGet(&cameraDesired) == 0) {
			switch (type) {
			case MIXERSETTINGS_MIXER1TYPE_CAMERAROLL:
				return cameraDesired.Roll;
			case MIXERSETTINGS_MIXER1TYPE_CAMERAPITCH:
				return cameraDesired.Pitch;
			case MIXERSETTINGS_MIXER1TYPE_CAMERAYAW:
				return cameraDesired.Yaw;
			default:
				break;
			}
		}
		return -1;
		/* No default-- make sure we handle all types positively here */
	}

	/* Can't get here. */
	return -1;
}

static MixerSettingsMixer1TypeOptions get_mixer_type(int idx)
{
	switch (idx) {
	case 0:
		return mixerSettings.Mixer1Type;
		break;
	case 1:
		return mixerSettings.Mixer2Type;
		break;
	case 2:
		return mixerSettings.Mixer3Type;
		break;
	case 3:
		return mixerSettings.Mixer4Type;
		break;
	case 4:
		return mixerSettings.Mixer5Type;
		break;
	case 5:
		return mixerSettings.Mixer6Type;
		break;
	case 6:
		return mixerSettings.Mixer7Type;
		break;
	case 7:
		return mixerSettings.Mixer8Type;
		break;
	case 8:
		return mixerSettings.Mixer9Type;
		break;
	case 9:
		return mixerSettings.Mixer10Type;
		break;
	default:
		// We can never get here unless there are mixer channels not handled in the above. Fail out.
		PIOS_Assert(0);
	}
}

static int8_t *get_mixer_vec(int idx)
{
	switch (idx) {
	case 0:
		return mixerSettings.Mixer1Vector;
		break;
	case 1:
		return mixerSettings.Mixer2Vector;
		break;
	case 2:
		return mixerSettings.Mixer3Vector;
		break;
	case 3:
		return mixerSettings.Mixer4Vector;
		break;
	case 4:
		return mixerSettings.Mixer5Vector;
		break;
	case 5:
		return mixerSettings.Mixer6Vector;
		break;
	case 6:
		return mixerSettings.Mixer7Vector;
		break;
	case 7:
		return mixerSettings.Mixer8Vector;
		break;
	case 8:
		return mixerSettings.Mixer9Vector;
		break;
	case 9:
		return mixerSettings.Mixer10Vector;
		break;
	default:
		// We can never get here unless there are mixer channels not handled in the above. Fail out.
		PIOS_Assert(0);
	}
}

#define OUTPUT_MODE_ASSUMPTIONS ( ( (int) PWM_MODE_1MHZ == ACTUATORSETTINGS_TIMERPWMRESOLUTION_1MHZ ) && \
	( (int) PWM_MODE_12MHZ == ACTUATORSETTINGS_TIMERPWMRESOLUTION_12MHZ ) )

DONT_BUILD_IF(!OUTPUT_MODE_ASSUMPTIONS, OutputModeAssumptions);

/**
 * @}
 * @}
 */
