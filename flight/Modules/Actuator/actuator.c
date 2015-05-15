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

// Private types


// Private variables
static struct pios_queue *queue;
static struct pios_thread *taskHandle;

// used to inform the actuator thread that actuator update rate is changed
static volatile bool actuator_settings_updated;
// used to inform the actuator thread that mixer settings are changed
static volatile bool mixer_settings_updated;

// Private functions
static void actuatorTask(void* parameters);
static float scaleChannel(float value, float max, float min, float neutral);
static void setFailsafe(const ActuatorSettingsData * actuatorSettings, const MixerSettingsData * mixerSettings);
static float MixerCurve(const float throttle, const float* curve, uint8_t elements);
static bool set_channel(uint8_t mixer_channel, float value, const ActuatorSettingsData * actuatorSettings);
static void actuator_update_rate_if_changed(const ActuatorSettingsData * actuatorSettings, bool force_update);
static void MixerSettingsUpdatedCb(UAVObjEvent * ev);
static void ActuatorSettingsUpdatedCb(UAVObjEvent * ev);
float ProcessMixer(const int index, const float curve1, const float curve2,
		   const MixerSettingsData* mixerSettings, ActuatorDesiredData* desired,
		   const float period);

//this structure is equivalent to the UAVObjects for one mixer.
typedef struct {
	uint8_t type;
	int8_t matrix[5];
} __attribute__((packed)) Mixer_t;

/**
 * @brief Module initialization
 * @return 0
 */
int32_t ActuatorStart()
{
	// Start main task
	taskHandle = PIOS_Thread_Create(actuatorTask, "Actuator", STACK_SIZE_BYTES, NULL, TASK_PRIORITY);
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
	ActuatorSettingsConnectCallback(ActuatorSettingsUpdatedCb);

	// Register for notification of changes to MixerSettings
	MixerSettingsInitialize();
	MixerSettingsConnectCallback(MixerSettingsUpdatedCb);

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
MODULE_INITCALL(ActuatorInitialize, ActuatorStart)

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
static void actuatorTask(void* parameters)
{
	UAVObjEvent ev;
	uint32_t lastSysTime;
	uint32_t thisSysTime;
	float dT = 0.0f;

	ActuatorCommandData command;
	ActuatorDesiredData desired;
	MixerStatusData mixerStatus;
	FlightStatusData flightStatus;

	/* Read initial values of ActuatorSettings */
	ActuatorSettingsData actuatorSettings;
	actuator_settings_updated = false;
	ActuatorSettingsGet(&actuatorSettings);

	/* Read initial values of MixerSettings */
	MixerSettingsData mixerSettings;
	mixer_settings_updated = false;
	MixerSettingsGet(&mixerSettings);

	/* Force an initial configuration of the actuator update rates */
	actuator_update_rate_if_changed(&actuatorSettings, true);

	// Go to the neutral (failsafe) values until an ActuatorDesired update is received
	setFailsafe(&actuatorSettings, &mixerSettings);

	// Main task loop
	lastSysTime = PIOS_Thread_Systime();
	while (1)
	{
		PIOS_WDG_UpdateFlag(PIOS_WDG_ACTUATOR);

		// Wait until the ActuatorDesired object is updated
		bool rc = PIOS_Queue_Receive(queue, &ev, FAILSAFE_TIMEOUT_MS);

		/* Process settings updated events even in timeout case so we always act on the latest settings */
		if (actuator_settings_updated) {
			actuator_settings_updated = false;
			ActuatorSettingsGet (&actuatorSettings);
			actuator_update_rate_if_changed (&actuatorSettings, false);
		}
		if (mixer_settings_updated) {
			mixer_settings_updated = false;
			MixerSettingsGet (&mixerSettings);
		}

		if (rc != true) {
			/* Update of ActuatorDesired timed out.  Go to failsafe */
			setFailsafe(&actuatorSettings, &mixerSettings);
			continue;
		}

		// Check how long since last update
		thisSysTime = PIOS_Thread_Systime();
		if(thisSysTime > lastSysTime) // reuse dt in case of wraparound
			dT = (thisSysTime - lastSysTime) / 1000.0f;
		lastSysTime = thisSysTime;

		FlightStatusGet(&flightStatus);
		ActuatorDesiredGet(&desired);
		ActuatorCommandGet(&command);

#if defined(MIXERSTATUS_DIAGNOSTICS)
		MixerStatusGet(&mixerStatus);
#endif
		int nMixers = 0;
		Mixer_t * mixers = (Mixer_t *)&mixerSettings.Mixer1Type;
		for(int ct=0; ct < MAX_MIX_ACTUATORS; ct++)
		{
			if(mixers[ct].type != MIXERSETTINGS_MIXER1TYPE_DISABLED)
			{
				nMixers ++;
			}
		}
		if((nMixers < 2) && !ActuatorCommandReadOnly()) //Nothing can fly with less than two mixers.
		{
			setFailsafe(&actuatorSettings, &mixerSettings); // So that channels like PWM buzzer keep working
			continue;
		}

		AlarmsClear(SYSTEMALARMS_ALARM_ACTUATOR);

		bool armed = flightStatus.Armed == FLIGHTSTATUS_ARMED_ARMED;
		bool positiveThrottle = desired.Throttle >= 0.00f;
		bool spinWhileArmed = actuatorSettings.MotorsSpinWhileArmed == ACTUATORSETTINGS_MOTORSSPINWHILEARMED_TRUE;

		float curve1 = MixerCurve(desired.Throttle,mixerSettings.ThrottleCurve1,MIXERSETTINGS_THROTTLECURVE1_NUMELEM);
		
		//The source for the secondary curve is selectable
		float curve2 = 0;
		AccessoryDesiredData accessory;
		switch(mixerSettings.Curve2Source) {
			case MIXERSETTINGS_CURVE2SOURCE_THROTTLE:
				curve2 = MixerCurve(desired.Throttle,mixerSettings.ThrottleCurve2,MIXERSETTINGS_THROTTLECURVE2_NUMELEM);
				break;
			case MIXERSETTINGS_CURVE2SOURCE_ROLL:
				curve2 = MixerCurve(desired.Roll,mixerSettings.ThrottleCurve2,MIXERSETTINGS_THROTTLECURVE2_NUMELEM);
				break;
			case MIXERSETTINGS_CURVE2SOURCE_PITCH:
				curve2 = MixerCurve(desired.Pitch,mixerSettings.ThrottleCurve2,
				MIXERSETTINGS_THROTTLECURVE2_NUMELEM);
				break;
			case MIXERSETTINGS_CURVE2SOURCE_YAW:
				curve2 = MixerCurve(desired.Yaw,mixerSettings.ThrottleCurve2,MIXERSETTINGS_THROTTLECURVE2_NUMELEM);
				break;
			case MIXERSETTINGS_CURVE2SOURCE_COLLECTIVE:
				ManualControlCommandCollectiveGet(&curve2);
				curve2 = MixerCurve(curve2,mixerSettings.ThrottleCurve2,
				MIXERSETTINGS_THROTTLECURVE2_NUMELEM);
				break;
			case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY0:
			case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY1:
			case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY2:
			case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY3:
			case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY4:
			case MIXERSETTINGS_CURVE2SOURCE_ACCESSORY5:
				if(AccessoryDesiredInstGet(mixerSettings.Curve2Source - MIXERSETTINGS_CURVE2SOURCE_ACCESSORY0,&accessory) == 0)
					curve2 = MixerCurve(accessory.AccessoryVal,mixerSettings.ThrottleCurve2,MIXERSETTINGS_THROTTLECURVE2_NUMELEM);
				else
					curve2 = 0;
				break;
		}

		float * status = (float *)&mixerStatus; //access status objects as an array of floats

		for(int ct=0; ct < MAX_MIX_ACTUATORS; ct++)
		{
			if(mixers[ct].type == MIXERSETTINGS_MIXER1TYPE_DISABLED) {
				// Set to minimum if disabled.  This is not the same as saying PWM pulse = 0 us
				status[ct] = -1;
				command.Channel[ct] = 0.0f;
				continue;
			}

			if((mixers[ct].type == MIXERSETTINGS_MIXER1TYPE_MOTOR) || (mixers[ct].type == MIXERSETTINGS_MIXER1TYPE_SERVO))
				status[ct] = ProcessMixer(ct, curve1, curve2, &mixerSettings, &desired, dT);
			else
				status[ct] = -1;



			// Motors have additional protection for when to be on
			if(mixers[ct].type == MIXERSETTINGS_MIXER1TYPE_MOTOR) {

				// If not armed or motors aren't meant to spin all the time
				if( !armed ||
				   (!spinWhileArmed && !positiveThrottle))
				{
					status[ct] = -1;  //force min throttle
				}
				// If armed meant to keep spinning,
				else if ((spinWhileArmed && !positiveThrottle) ||
					 (status[ct] < 0) )
					status[ct] = 0;
			}

			// If an accessory channel is selected for direct bypass mode
			// In this configuration the accessory channel is scaled and mapped
			// directly to output.  Note: THERE IS NO SAFETY CHECK HERE FOR ARMING
			// these also will not be updated in failsafe mode.  I'm not sure what
			// the correct behavior is since it seems domain specific.  I don't love
			// this code
			if( (mixers[ct].type >= MIXERSETTINGS_MIXER1TYPE_ACCESSORY0) &&
			   (mixers[ct].type <= MIXERSETTINGS_MIXER1TYPE_ACCESSORY5))
			{
				if(AccessoryDesiredInstGet(mixers[ct].type - MIXERSETTINGS_MIXER1TYPE_ACCESSORY0,&accessory) == 0)
					status[ct] = accessory.AccessoryVal;
				else
					status[ct] = -1;
			}
			if( (mixers[ct].type >= MIXERSETTINGS_MIXER1TYPE_CAMERAROLL) &&
			   (mixers[ct].type <= MIXERSETTINGS_MIXER1TYPE_CAMERAYAW))
			{
				CameraDesiredData cameraDesired;
				if( CameraDesiredGet(&cameraDesired) == 0 ) {
					switch(mixers[ct].type) {
						case MIXERSETTINGS_MIXER1TYPE_CAMERAROLL:
							status[ct] = cameraDesired.Roll;
							break;
						case MIXERSETTINGS_MIXER1TYPE_CAMERAPITCH:
							status[ct] = cameraDesired.Pitch;
							break;
						case MIXERSETTINGS_MIXER1TYPE_CAMERAYAW:
							status[ct] = cameraDesired.Yaw;
							break;
						default:
							break;
					}
				}
				else
					status[ct] = -1;
			}
		}
		
		for(int i = 0; i < MAX_MIX_ACTUATORS; i++) 
			command.Channel[i] = scaleChannel(status[i],
							   actuatorSettings.ChannelMax[i],
							   actuatorSettings.ChannelMin[i],
							   actuatorSettings.ChannelNeutral[i]);
			
		// Store update time
		command.UpdateTime = 1000.0f*dT;
		if(1000.0f*dT > command.MaxUpdateTime)
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

		for (int n = 0; n < ACTUATORCOMMAND_CHANNEL_NUMELEM; ++n)
		{
			success &= set_channel(n, command.Channel[n], &actuatorSettings);
		}
#if defined(PIOS_INCLUDE_HPWM)
		PIOS_Servo_Update();
#endif

		if(!success) {
			command.NumFailedUpdates++;
			ActuatorCommandSet(&command);
			AlarmsSet(SYSTEMALARMS_ALARM_ACTUATOR, SYSTEMALARMS_ALARM_CRITICAL);
		}

	}
}



/**
 *Process mixing for one actuator
 */
float ProcessMixer(const int index, const float curve1, const float curve2,
		   const MixerSettingsData* mixerSettings, ActuatorDesiredData* desired, const float period)
{
	const Mixer_t * mixers = (Mixer_t *)&mixerSettings->Mixer1Type; //pointer to array of mixers in UAVObjects
	const Mixer_t * mixer = &mixers[index];

	float result = (((float)mixer->matrix[MIXERSETTINGS_MIXER1VECTOR_THROTTLECURVE1] / 128.0f) * curve1) +
		       (((float)mixer->matrix[MIXERSETTINGS_MIXER1VECTOR_THROTTLECURVE2] / 128.0f) * curve2) +
		       (((float)mixer->matrix[MIXERSETTINGS_MIXER1VECTOR_ROLL] / 128.0f) * desired->Roll) +
		       (((float)mixer->matrix[MIXERSETTINGS_MIXER1VECTOR_PITCH] / 128.0f) * desired->Pitch) +
		       (((float)mixer->matrix[MIXERSETTINGS_MIXER1VECTOR_YAW] / 128.0f) * desired->Yaw);

	if((mixer->type == MIXERSETTINGS_MIXER1TYPE_MOTOR) && (result < 0.0f))
	{
			result = 0.0f; //idle throttle
	}

	return(result);
}


/**
 *Interpolate a throttle curve. Throttle input should be in the range 0 to 1.
 *Output is in the range 0 to 1.
 */
static float MixerCurve(const float throttle, const float* curve, uint8_t elements)
{
	float scale = throttle * (float) (elements - 1);
	int idx1 = scale;
	scale -= (float)idx1; //remainder
	if(curve[0] < -1)
	{
		return(throttle);
	}
	if (idx1 < 0)
	{
		idx1 = 0; //clamp to lowest entry in table
		scale = 0;
	}
	int idx2 = idx1 + 1;
	if(idx2 >= elements)
	{
		idx2 = elements -1; //clamp to highest entry in table
		if(idx1 >= elements)
		{
			idx1 = elements -1;
		}
	}
	return curve[idx1] * (1.0f - scale) + curve[idx2] * scale;
}


/**
 * Convert channel from -1/+1 to servo pulse duration in microseconds
 */
static float scaleChannel(float value, float max, float min, float neutral)
{
	float valueScaled;
	// Scale
	if ( value >= 0.0f)
	{
		valueScaled = value*(max-neutral) + neutral;
	}
	else
	{
		valueScaled = value*(neutral-min) + neutral;
	}

	if (max>min)
	{
		if( valueScaled > max ) valueScaled = max;
		if( valueScaled < min ) valueScaled = min;
	}
	else
	{
		if( valueScaled < max ) valueScaled = max;
		if( valueScaled > min ) valueScaled = min;
	}

	return valueScaled;
}

/**
 * Set actuator output to the neutral values (failsafe)
 */
static void setFailsafe(const ActuatorSettingsData * actuatorSettings, const MixerSettingsData * mixerSettings)
{
	/* grab only the parts that we are going to use */
	float Channel[ACTUATORCOMMAND_CHANNEL_NUMELEM];
	ActuatorCommandChannelGet(Channel);

	const Mixer_t * mixers = (Mixer_t *)&mixerSettings->Mixer1Type; //pointer to array of mixers in UAVObjects

	// Reset ActuatorCommand to safe values
	for (int n = 0; n < ACTUATORCOMMAND_CHANNEL_NUMELEM; ++n)
	{

		if(mixers[n].type == MIXERSETTINGS_MIXER1TYPE_MOTOR)
		{
			Channel[n] = actuatorSettings->ChannelMin[n];
		}
		else if(mixers[n].type == MIXERSETTINGS_MIXER1TYPE_SERVO)
		{
			Channel[n] = actuatorSettings->ChannelNeutral[n];
		}
		else
		{
			Channel[n] = 0.0f;
		}
		
		
	}

	// Set alarm
	AlarmsSet(SYSTEMALARMS_ALARM_ACTUATOR, SYSTEMALARMS_ALARM_CRITICAL);

	// Update servo outputs
	for (int n = 0; n < ACTUATORCOMMAND_CHANNEL_NUMELEM; ++n)
	{
		set_channel(n, Channel[n], actuatorSettings);
	}
#if defined(PIOS_INCLUDE_HPWM) // TODO: this is actually about the synchronous updating and not resolution
	PIOS_Servo_Update();
#endif

	// Update output object's parts that we changed
	ActuatorCommandChannelSet(Channel);
}

static bool set_channel(uint8_t mixer_channel, float value, const ActuatorSettingsData * actuatorSettings)
#if defined(ARCH_POSIX) || defined(ARCH_WIN32)
{
	return true;
}
#else
{
	switch(actuatorSettings->ChannelType[mixer_channel]) {
		case ACTUATORSETTINGS_CHANNELTYPE_PWMALARMBUZZER: 
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
                        if(actuatorSettings->ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_PWMALARMBUZZER)
                        {
                            
                            // Decide what tune to play
                            if (AlarmsGet(SYSTEMALARMS_ALARM_BATTERY) > SYSTEMALARMS_ALARM_WARNING) {
                                    newTune = 0b11110110110000;	// pause, short, short, short, long
                            } else if (AlarmsGet(SYSTEMALARMS_ALARM_GPS) >= SYSTEMALARMS_ALARM_WARNING) {
                                    newTune = 0x80000000;			// pause, short
                            } else {
                                    newTune = 0;
                            }

                            // Do we need to change tune?
                            if (newTune != currBuzzTune) {
                                    currBuzzTune = newTune;
                                    currBuzzTuneState = currBuzzTune;
                            }
                        }
                        else // ACTUATORSETTINGS_CHANNELTYPE_ARMINGLED || ACTUATORSETTINGS_CHANNELTYPE_INFOLED
                        {
                            uint8_t arming;
                            FlightStatusArmedGet(&arming);
                            //base idle tune  
                            newTune =  0x80000000;      // 0b1000...
                            
                            // Merge the error pattern for InfoLed
                            if(actuatorSettings->ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_INFOLED)
                            {
                                if (AlarmsGet(SYSTEMALARMS_ALARM_BATTERY) > SYSTEMALARMS_ALARM_WARNING) 
                                {
                                    newTune |= 0b00000000001111111011111110000000;
                                }
                                else if(AlarmsGet(SYSTEMALARMS_ALARM_GPS) >= SYSTEMALARMS_ALARM_WARNING) 
                                {             
                                    newTune |= 0b00000000000000110110110000000000;
                                }
                            }
                            // fast double blink pattern if armed 
                            if (arming == FLIGHTSTATUS_ARMED_ARMED) 
                               newTune |= 0xA0000000;   // 0b101000... 

                            // Do we need to change tune?
                            if(actuatorSettings->ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_ARMINGLED)
                            {
                                if (newTune != currArmingTune) {
                                    currArmingTune = newTune;
                                    // note: those are both updated so that Info and Arming are in sync if used simultaneously
                                    currArmingTuneState = currArmingTune;
                                    currInfoTuneState = currInfoTune;
                                }
                            }
                            else
                            {
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
                            if(thisSysTime > lastSysTime)
                                dT = thisSysTime - lastSysTime;
                            if(actuatorSettings->ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_PWMALARMBUZZER)
                                buzzOn = (currBuzzTuneState&1);	// Buzz when the LS bit is 1
                            else if(actuatorSettings->ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_ARMINGLED)
                                buzzOn = (currArmingTuneState&1);	
                            else if(actuatorSettings->ChannelType[mixer_channel] == ACTUATORSETTINGS_CHANNELTYPE_INFOLED)
                                buzzOn = (currInfoTuneState&1);

                            if (dT > 80) {
                                    // Go to next bit in alarm_seq_state
                                    currArmingTuneState >>=1;
                                    currInfoTuneState >>= 1;
                                    currBuzzTuneState >>= 1;

                                    if (currBuzzTuneState == 0)
                                            currBuzzTuneState = currBuzzTune;	// All done, re-start the tune
                                    if (currArmingTuneState == 0)
                                            currArmingTuneState = currArmingTune;
                                    if (currInfoTuneState == 0)
                                            currInfoTuneState = currInfoTune;	
                                    lastSysTime = thisSysTime;
                            }
			}
			PIOS_Servo_Set(actuatorSettings->ChannelAddr[mixer_channel],
							buzzOn?actuatorSettings->ChannelMax[mixer_channel]:actuatorSettings->ChannelMin[mixer_channel]);
			return true;
		}
		case ACTUATORSETTINGS_CHANNELTYPE_PWM:
#if defined(PIOS_INCLUDE_HPWM)
			// The HPWM method will convert from us to the appropriate settings
			PIOS_Servo_Set(actuatorSettings->ChannelAddr[mixer_channel], value);
#else
			PIOS_Servo_Set(actuatorSettings->ChannelAddr[mixer_channel], value);
#endif
			return true;
#if defined(PIOS_INCLUDE_I2C_ESC)
		case ACTUATORSETTINGS_CHANNELTYPE_MK:
			return PIOS_SetMKSpeed(actuatorSettings->ChannelAddr[mixer_channel],value);
		case ACTUATORSETTINGS_CHANNELTYPE_ASTEC4:
			return PIOS_SetAstec4Speed(actuatorSettings->ChannelAddr[mixer_channel],value);
			break;
#endif
		default:
			return false;
	}

	return false;

}
#endif

/**
 * @brief Update the servo update rate
 */
static void actuator_update_rate_if_changed(const ActuatorSettingsData * actuatorSettings, bool force_update)
{
	static uint16_t prevChannelUpdateFreq[ACTUATORSETTINGS_TIMERUPDATEFREQ_NUMELEM];

	// check if the any rate setting is changed
	if (force_update ||
		memcmp (prevChannelUpdateFreq,
			actuatorSettings->TimerUpdateFreq,
			sizeof(prevChannelUpdateFreq)) != 0) {
		/* Something has changed, apply the settings to HW */
		memcpy (prevChannelUpdateFreq,
			actuatorSettings->TimerUpdateFreq,
			sizeof(prevChannelUpdateFreq));
		PIOS_Servo_SetMode(actuatorSettings->TimerUpdateFreq, actuatorSettings->TimerPwmResolution, ACTUATORSETTINGS_TIMERPWMRESOLUTION_NUMELEM);
	}
}

static void ActuatorSettingsUpdatedCb(UAVObjEvent * ev)
{
	actuator_settings_updated = true;
}

static void MixerSettingsUpdatedCb(UAVObjEvent * ev)
{
	mixer_settings_updated = true;
}

#define OUTPUT_MODE_ASSUMPTIONS ( PWM_MODE_1US == ACTUATORSETTINGS_TIMERPWMRESOLUTION_1US ) && \
                                ( PWM_MODE_80NS == ACTUATORSETTINGS_TIMERPWMRESOLUTION_80NS )
#if !(OUTPUT_MODE_ASSUMPTIONS)
    #error "ActuatorSettings.TimerPwmResolution does not match PWM_MODE enum"
#endif
/**
 * @}
 * @}
 */
