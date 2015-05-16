/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup Control Control Module
 * @{
 *
 * @file       transmitter_control.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      Handles R/C link and flight mode.
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
#include "control.h"
#include "transmitter_control.h"
#include "pios_thread.h"

#include "accessorydesired.h"
#include "actuatordesired.h"
#include "altitudeholddesired.h"
#include "altitudeholdsettings.h"
#include "baroaltitude.h"
#include "flighttelemetrystats.h"
#include "flightstatus.h"
#include "loitercommand.h"
#include "manualcontrolsettings.h"
#include "manualcontrolcommand.h"
#include "pathdesired.h"
#include "positionactual.h"
#include "receiveractivity.h"
#include "stabilizationsettings.h"
#include "stabilizationdesired.h"
#include "systemsettings.h"

#include "misc_math.h"

#if defined(PIOS_INCLUDE_USB_RCTX)
#include "pios_usb_rctx.h"
#endif	/* PIOS_INCLUDE_USB_RCTX */

#define ARMED_THRESHOLD    0.50f
//safe band to allow a bit of calibration error or trim offset (in microseconds)
#define CONNECTION_OFFSET_THROTTLE 100
#define CONNECTION_OFFSET          250

#define RCVR_ACTIVITY_MONITOR_CHANNELS_PER_GROUP 12
#define RCVR_ACTIVITY_MONITOR_MIN_RANGE 10
struct rcvr_activity_fsm {
	ManualControlSettingsChannelGroupsOptions group;
	uint16_t prev[RCVR_ACTIVITY_MONITOR_CHANNELS_PER_GROUP];
	uint8_t sample_count;
};


// Private variables
static FlightStatusData           flightStatus;
static ManualControlCommandData   cmd;
static ManualControlSettingsData  settings;
static uint8_t                    disconnected_count = 0;
static uint8_t                    connected_count = 0;
static struct rcvr_activity_fsm   activity_fsm;
static uint32_t               lastActivityTime;
static uint32_t               lastSysTime;
static float                      flight_mode_value;
static enum control_events        pending_control_event;
static bool                       settings_updated;

// Private functions
static void update_actuator_desired(ManualControlCommandData * cmd);
static void update_stabilization_desired(ManualControlCommandData * cmd, ManualControlSettingsData * settings);
static void altitude_hold_desired(ManualControlCommandData * cmd, bool flightModeChanged);
static void set_flight_mode();
static void process_transmitter_events(ManualControlCommandData * cmd, ManualControlSettingsData * settings, bool valid);
static void set_manual_control_error(SystemAlarmsManualControlOptions errorCode);
static float scaleChannel(int16_t value, int16_t max, int16_t min, int16_t neutral);
static uint32_t timeDifferenceMs(uint32_t start_time, uint32_t end_time);
static bool validInputRange(int16_t min, int16_t max, uint16_t value, uint16_t offset);
static void applyDeadband(float *value, float deadband);
static void resetRcvrActivity(struct rcvr_activity_fsm * fsm);
static bool updateRcvrActivity(struct rcvr_activity_fsm * fsm);
static void manual_control_settings_updated(UAVObjEvent * ev);
static void set_loiter_command(ManualControlCommandData * cmd);

#define assumptions (assumptions1 && assumptions3 && assumptions5 && assumptions_flightmode && assumptions_channelcount)

//! Initialize the transmitter control mode
int32_t transmitter_control_initialize()
{
	/* Check the assumptions about uavobject enum's are correct */
	if(!assumptions)
		return -1;

	AccessoryDesiredInitialize();
	ManualControlCommandInitialize();
	FlightStatusInitialize();
	StabilizationDesiredInitialize();
	ReceiverActivityInitialize();
	ManualControlSettingsInitialize();

	// Both the gimbal and coptercontrol do not support loitering
#if !defined(COPTERCONTROL) && !defined(GIMBAL)
	LoiterCommandInitialize();
#endif

	/* For now manual instantiate extra instances of Accessory Desired.  In future  */
	/* should be done dynamically this includes not even registering it if not used */
	AccessoryDesiredCreateInstance();
	AccessoryDesiredCreateInstance();

	/* No pending control events */
	pending_control_event = CONTROL_EVENTS_NONE;

	/* Initialize the RcvrActivty FSM */
	lastActivityTime = PIOS_Thread_Systime();
	resetRcvrActivity(&activity_fsm);

	// Use callback to update the settings when they change
	ManualControlSettingsConnectCallback(manual_control_settings_updated);
	manual_control_settings_updated(NULL);

	// Main task loop
	lastSysTime = PIOS_Thread_Systime();
	return 0;
}

/**
  * @brief Process inputs and arming
  *
  * This will always process the arming signals as for now the transmitter
  * is always in charge.  When a transmitter is not detected control will
  * fall back to the failsafe module.  If the flight mode is in tablet
  * control position then control will be ceeded to that module.
  */
int32_t transmitter_control_update()
{
	lastSysTime = PIOS_Thread_Systime();

	float scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_NUMELEM];
	bool validChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_NUMELEM];

	if (settings_updated) {
		settings_updated = false;
		ManualControlSettingsGet(&settings);
	}

	/* Update channel activity monitor */
	if (flightStatus.Armed == FLIGHTSTATUS_ARMED_DISARMED) {
		if (updateRcvrActivity(&activity_fsm)) {
			/* Reset the aging timer because activity was detected */
			lastActivityTime = lastSysTime;
		}
	}
	if (timeDifferenceMs(lastActivityTime, lastSysTime) > 5000) {
		resetRcvrActivity(&activity_fsm);
		lastActivityTime = lastSysTime;
	}

	if (ManualControlCommandReadOnly()) {
		FlightTelemetryStatsData flightTelemStats;
		FlightTelemetryStatsGet(&flightTelemStats);
		if(flightTelemStats.Status != FLIGHTTELEMETRYSTATS_STATUS_CONNECTED) {
			/* trying to fly via GCS and lost connection.  fall back to transmitter */
			UAVObjMetadata metadata;
			ManualControlCommandGetMetadata(&metadata);
			UAVObjSetAccess(&metadata, ACCESS_READWRITE);
			ManualControlCommandSetMetadata(&metadata);
		}

		// Don't process anything else when GCS is overriding the objects
		return 0;
	}

	if (settings.RssiType != MANUALCONTROLSETTINGS_RSSITYPE_NONE) {
		int32_t value = 0;
		extern uintptr_t pios_rcvr_group_map[];
		switch (settings.RssiType) {
		case MANUALCONTROLSETTINGS_RSSITYPE_PWM:
			value = PIOS_RCVR_Read(pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PWM], settings.RssiChannelNumber);
			if(settings.RssiPwmPeriod != 0)
				value = (value) % (settings.RssiPwmPeriod);
			break;
		case MANUALCONTROLSETTINGS_RSSITYPE_PPM:
			value = PIOS_RCVR_Read(pios_rcvr_group_map[MANUALCONTROLSETTINGS_CHANNELGROUPS_PPM], settings.RssiChannelNumber);
			break;
		case MANUALCONTROLSETTINGS_RSSITYPE_ADC:
#if defined(PIOS_INCLUDE_ADC)
			value = PIOS_ADC_GetChannelRaw(settings.RssiChannelNumber);
#endif
			break;
		}
		if(value < 0)
			value = 0;
		if (settings.RssiMax == settings.RssiMin)
			cmd.Rssi = 0;
		else
			cmd.Rssi = ((float)(value - settings.RssiMin)/((float)settings.RssiMax-settings.RssiMin)) * 100;
		cmd.RawRssi = value;
	}

	bool valid_input_detected = true;

	// Read channel values in us
	for (uint8_t n = 0; 
	     n < MANUALCONTROLSETTINGS_CHANNELGROUPS_NUMELEM && n < MANUALCONTROLCOMMAND_CHANNEL_NUMELEM;
	     ++n) {
		extern uintptr_t pios_rcvr_group_map[];

		if (settings.ChannelGroups[n] >= MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE) {
			cmd.Channel[n] = PIOS_RCVR_INVALID;
		} else {
			cmd.Channel[n] = PIOS_RCVR_Read(pios_rcvr_group_map[settings.ChannelGroups[n]],
							settings.ChannelNumber[n]);
		}

		// If a channel has timed out this is not valid data and we shouldn't update anything
		// until we decide to go to failsafe
		if(cmd.Channel[n] == (uint16_t) PIOS_RCVR_TIMEOUT) {
			valid_input_detected = false;
			validChannel[n] = false;
		} else {
			scaledChannel[n] = scaleChannel(cmd.Channel[n], settings.ChannelMax[n],	settings.ChannelMin[n], settings.ChannelNeutral[n]);
			validChannel[n] = validInputRange(settings.ChannelMin[n], settings.ChannelMax[n], cmd.Channel[n], CONNECTION_OFFSET);
		}
	}

	// Check settings, if error raise alarm
	if (settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_ROLL] >= MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE ||
		settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_PITCH] >= MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE ||
		settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_YAW] >= MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE ||
		settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_THROTTLE] >= MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE ||
		// Check all channel mappings are valid
		cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_ROLL] == (uint16_t) PIOS_RCVR_INVALID ||
		cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_PITCH] == (uint16_t) PIOS_RCVR_INVALID ||
		cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_YAW] == (uint16_t) PIOS_RCVR_INVALID ||
		cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_THROTTLE] == (uint16_t) PIOS_RCVR_INVALID ||
		// Check the driver exists
		cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_ROLL] == (uint16_t) PIOS_RCVR_NODRIVER ||
		cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_PITCH] == (uint16_t) PIOS_RCVR_NODRIVER ||
		cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_YAW] == (uint16_t) PIOS_RCVR_NODRIVER ||
		cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_THROTTLE] == (uint16_t) PIOS_RCVR_NODRIVER ||
		// Check the FlightModeNumber is valid
		settings.FlightModeNumber < 1 || settings.FlightModeNumber > MANUALCONTROLSETTINGS_FLIGHTMODEPOSITION_NUMELEM ||
		// If we've got more than one possible valid FlightMode, we require a configured FlightMode channel
		((settings.FlightModeNumber > 1) && (settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_FLIGHTMODE] >= MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE)) ||
		// Whenever FlightMode channel is configured, it needs to be valid regardless of FlightModeNumber settings
		((settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_FLIGHTMODE] < MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE) && (
			cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_FLIGHTMODE] == (uint16_t) PIOS_RCVR_INVALID ||
			cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_FLIGHTMODE] == (uint16_t) PIOS_RCVR_NODRIVER ))) {

		set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_SETTINGS);

		cmd.Connected = MANUALCONTROLCOMMAND_CONNECTED_FALSE;
		ManualControlCommandSet(&cmd);

		// Need to do this here since we don't process armed status.  Since this shouldn't happen in flight (changed config) 
		// immediately disarm
		pending_control_event = CONTROL_EVENTS_DISARM;

		return -1;
	}

	// the block above validates the input configuration. this simply checks that the range is valid if flight mode is configured.
	bool flightmode_valid_input = settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_FLIGHTMODE] >= MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE ||
	    validChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_FLIGHTMODE];

	// because arming is optional we must determine if it is needed before checking it is valid
	bool arming_valid_input = settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_ARMING] >= MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE ||
	    validChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_ARMING];

	// decide if we have valid manual input or not
	valid_input_detected &= validChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_THROTTLE] &&
	    validChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_ROLL] &&
	    validChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_YAW] &&
	    validChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_PITCH] &&
	    flightmode_valid_input &&
	    arming_valid_input;

	// Implement hysteresis loop on connection status
	if (valid_input_detected && (++connected_count > 10)) {
		cmd.Connected = MANUALCONTROLCOMMAND_CONNECTED_TRUE;
		connected_count = 0;
		disconnected_count = 0;
	} else if (!valid_input_detected && (++disconnected_count > 10)) {
		cmd.Connected = MANUALCONTROLCOMMAND_CONNECTED_FALSE;
		connected_count = 0;
		disconnected_count = 0;
	}

	if (cmd.Connected == MANUALCONTROLCOMMAND_CONNECTED_FALSE) {
		// These values are not used but just put ManualControlCommand in a sane state.  When
		// Connected is false, then the failsafe submodule will be in control.

		cmd.Throttle = -1;
		cmd.Roll = 0;
		cmd.Yaw = 0;
		cmd.Pitch = 0;
		cmd.Collective = 0;

		set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_NORX);

	} else if (valid_input_detected) {
		set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_NONE);

		// Scale channels to -1 -> +1 range
		cmd.Roll           = scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_ROLL];
		cmd.Pitch          = scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_PITCH];
		cmd.Yaw            = scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_YAW];
		cmd.Throttle       = scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_THROTTLE];
		cmd.ArmSwitch      = scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_ARMING] > 0 ? 
		                     MANUALCONTROLCOMMAND_ARMSWITCH_ARMED : MANUALCONTROLCOMMAND_ARMSWITCH_DISARMED;
		flight_mode_value  = scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_FLIGHTMODE];

		// Apply deadband for Roll/Pitch/Yaw stick inputs
		if (settings.Deadband) {
			applyDeadband(&cmd.Roll, settings.Deadband);
			applyDeadband(&cmd.Pitch, settings.Deadband);
			applyDeadband(&cmd.Yaw, settings.Deadband);
		}

		if(cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_COLLECTIVE] != (uint16_t) PIOS_RCVR_INVALID &&
		   cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_COLLECTIVE] != (uint16_t) PIOS_RCVR_NODRIVER &&
		   cmd.Channel[MANUALCONTROLSETTINGS_CHANNELGROUPS_COLLECTIVE] != (uint16_t) PIOS_RCVR_TIMEOUT) {
			cmd.Collective = scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_COLLECTIVE];
		}
		   
		AccessoryDesiredData accessory;
		// Set Accessory 0
		if (settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_ACCESSORY0] != 
			MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE) {
			accessory.AccessoryVal = scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_ACCESSORY0];
			if(AccessoryDesiredInstSet(0, &accessory) != 0) //These are allocated later and that allocation might fail
				set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_ACCESSORY);
		}
		// Set Accessory 1
		if (settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_ACCESSORY1] != 
			MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE) {
			accessory.AccessoryVal = scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_ACCESSORY1];
			if(AccessoryDesiredInstSet(1, &accessory) != 0) //These are allocated later and that allocation might fail
				set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_ACCESSORY);
		}
		// Set Accessory 2
		if (settings.ChannelGroups[MANUALCONTROLSETTINGS_CHANNELGROUPS_ACCESSORY2] != 
			MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE) {
			accessory.AccessoryVal = scaledChannel[MANUALCONTROLSETTINGS_CHANNELGROUPS_ACCESSORY2];
			if(AccessoryDesiredInstSet(2, &accessory) != 0) //These are allocated later and that allocation might fail
				set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_ACCESSORY);
		}

	}

	// Process arming outside conditional so system will disarm when disconnected.  Notice this
	// is processed in the _update method instead of _select method so the state system is always
	// evalulated, even if not detected.
	process_transmitter_events(&cmd, &settings, valid_input_detected);
	
	// Update cmd object
	ManualControlCommandSet(&cmd);

#if defined(PIOS_INCLUDE_USB_RCTX)
	// Optionally make the hardware behave like a USB HID joystick
	if (pios_usb_rctx_id) {
		PIOS_USB_RCTX_Update(pios_usb_rctx_id,
				cmd.Channel,
				settings.ChannelMin,
				settings.ChannelMax,
				NELEMENTS(cmd.Channel));
	}
#endif	/* PIOS_INCLUDE_USB_RCTX */

	return 0;
}

/**
 * Select and use transmitter control
 * @param [in] reset_controller True if previously another controller was used
 */
int32_t transmitter_control_select(bool reset_controller)
{
	// Activate the flight mode corresponding to the switch position
	set_flight_mode();

	ManualControlCommandGet(&cmd);
	FlightStatusGet(&flightStatus);

	// Depending on the mode update the Stabilization or Actuator objects
	static uint8_t lastFlightMode = FLIGHTSTATUS_FLIGHTMODE_MANUAL;

	switch(flightStatus.FlightMode) {
	case FLIGHTSTATUS_FLIGHTMODE_MANUAL:
		update_actuator_desired(&cmd);
		break;
	case FLIGHTSTATUS_FLIGHTMODE_ACRO:
	case FLIGHTSTATUS_FLIGHTMODE_LEVELING:
	case FLIGHTSTATUS_FLIGHTMODE_VIRTUALBAR:
	case FLIGHTSTATUS_FLIGHTMODE_MWRATE:
	case FLIGHTSTATUS_FLIGHTMODE_HORIZON:
	case FLIGHTSTATUS_FLIGHTMODE_AXISLOCK:
	case FLIGHTSTATUS_FLIGHTMODE_STABILIZED1:
	case FLIGHTSTATUS_FLIGHTMODE_STABILIZED2:
	case FLIGHTSTATUS_FLIGHTMODE_STABILIZED3:
		update_stabilization_desired(&cmd, &settings);
		break;
	case FLIGHTSTATUS_FLIGHTMODE_AUTOTUNE:
		// Tuning takes settings directly from manualcontrolcommand.  No need to
		// call anything else.  This just avoids errors.
		break;
	case FLIGHTSTATUS_FLIGHTMODE_ALTITUDEHOLD:
		altitude_hold_desired(&cmd, lastFlightMode != flightStatus.FlightMode);
		break;
	case FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD:
		set_loiter_command(&cmd);
	case FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME:
		// The path planner module processes data here
		break;
	case FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER:
		break;
	default:
		set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_UNDEFINED);
		break;
	}
	lastFlightMode = flightStatus.FlightMode;

	return 0;
}

//! Get any control events and flush it
enum control_events transmitter_control_get_events()
{
	enum control_events to_return = pending_control_event;
	pending_control_event = CONTROL_EVENTS_NONE;
	return to_return;
}

//! Determine which of N positions the flight mode switch is in but do not set it
uint8_t transmitter_control_get_flight_mode()
{
	// Convert flightMode value into the switch position in the range [0..N-1]
	uint8_t pos = ((int16_t)(flight_mode_value * 256.0f) + 256) * settings.FlightModeNumber >> 9;
	if (pos >= settings.FlightModeNumber)
		pos = settings.FlightModeNumber - 1;

	return settings.FlightModePosition[pos];
}

//! Schedule the appropriate event to change the arm status
static void set_armed_if_changed(uint8_t new_arm) {
	uint8_t arm_status;
	FlightStatusArmedGet(&arm_status);
	if (new_arm != arm_status) {
		switch(new_arm) {
		case FLIGHTSTATUS_ARMED_DISARMED:
			pending_control_event = CONTROL_EVENTS_DISARM;
			break;
		case FLIGHTSTATUS_ARMED_ARMING:
			pending_control_event = CONTROL_EVENTS_ARMING;
			break;
		case FLIGHTSTATUS_ARMED_ARMED:
			pending_control_event = CONTROL_EVENTS_ARM;
			break;
		}
	}
}

/**
 * Check sticks to determine if they are in the arming position
 */
static bool arming_position(ManualControlCommandData * cmd, ManualControlSettingsData * settings) {

	bool lowThrottle = cmd->Throttle <= 0;

	switch(settings->Arming) {
	case MANUALCONTROLSETTINGS_ARMING_ALWAYSDISARMED:
		return false;
	case MANUALCONTROLSETTINGS_ARMING_ALWAYSARMED:
		return true;
	case MANUALCONTROLSETTINGS_ARMING_ROLLLEFT:
		return lowThrottle && cmd->Roll < -ARMED_THRESHOLD;
	case MANUALCONTROLSETTINGS_ARMING_ROLLRIGHT:
		return lowThrottle && cmd->Roll > ARMED_THRESHOLD;
	case MANUALCONTROLSETTINGS_ARMING_YAWLEFT:
		return lowThrottle && cmd->Yaw < -ARMED_THRESHOLD;
	case MANUALCONTROLSETTINGS_ARMING_YAWRIGHT:
		return lowThrottle && cmd->Yaw > ARMED_THRESHOLD;
	case MANUALCONTROLSETTINGS_ARMING_CORNERS:
		return lowThrottle && (
			(cmd->Yaw > ARMED_THRESHOLD || cmd->Yaw < -ARMED_THRESHOLD) &&
			(cmd->Roll > ARMED_THRESHOLD || cmd->Roll < -ARMED_THRESHOLD) ) &&
			(cmd->Pitch > ARMED_THRESHOLD);
			// Note that pulling pitch stick down corresponds to positive values
	case MANUALCONTROLSETTINGS_ARMING_SWITCH:
		return cmd->ArmSwitch == MANUALCONTROLCOMMAND_ARMSWITCH_ARMED;
	case MANUALCONTROLSETTINGS_ARMING_SWITCHTHROTTLE:
		return lowThrottle && cmd->ArmSwitch == MANUALCONTROLCOMMAND_ARMSWITCH_ARMED;
	default:
		return false;
	}
}

/**
 * Check sticks to determine if they are in the disarmed position
 */
static bool disarming_position(ManualControlCommandData * cmd, ManualControlSettingsData * settings) {

	bool lowThrottle = cmd->Throttle <= 0;

	switch(settings->Arming) {
	case MANUALCONTROLSETTINGS_ARMING_ALWAYSDISARMED:
		return true;
	case MANUALCONTROLSETTINGS_ARMING_ALWAYSARMED:
		return false;
	case MANUALCONTROLSETTINGS_ARMING_ROLLLEFT:
		return lowThrottle && cmd->Roll > ARMED_THRESHOLD;
	case MANUALCONTROLSETTINGS_ARMING_ROLLRIGHT:
		return lowThrottle && cmd->Roll < -ARMED_THRESHOLD;
	case MANUALCONTROLSETTINGS_ARMING_YAWLEFT:
		return lowThrottle && cmd->Yaw > ARMED_THRESHOLD;
	case MANUALCONTROLSETTINGS_ARMING_YAWRIGHT:
		return lowThrottle && cmd->Yaw < -ARMED_THRESHOLD;
	case MANUALCONTROLSETTINGS_ARMING_CORNERS:
		return lowThrottle && (
			(cmd->Yaw > ARMED_THRESHOLD || cmd->Yaw < -ARMED_THRESHOLD) &&
			(cmd->Roll > ARMED_THRESHOLD || cmd->Roll < -ARMED_THRESHOLD) );
	case MANUALCONTROLSETTINGS_ARMING_SWITCH:
	case MANUALCONTROLSETTINGS_ARMING_SWITCHTHROTTLE:
		return cmd->ArmSwitch != MANUALCONTROLCOMMAND_ARMSWITCH_ARMED;
	default:
		return false;
	}
}

/**
 * @brief Process the inputs and determine whether to arm or not
 * @param[in] cmd The manual control inputs
 * @param[in] settings Settings indicating the necessary position
 * @param[in] scaled The scaled channels, used for checking arming
 * @param[in] valid If the input data is valid (i.e. transmitter is transmitting)
 */
static void process_transmitter_events(ManualControlCommandData * cmd, ManualControlSettingsData * settings, bool valid)
{

	/* State machine to determine arming of disarming:
	  DISARMED: if invalid input, remain.
	            look for the arm signal to go true.
	              if a switch go straight to ARMED
	              else store time and go to ARMING
	  ARMING: if arm signal ends or invalid go to IDLE
	          if time expires go to ARMED_STILL_HOLDING
	  ARMED_STILL_HOLDING: if arm signal ends go to ARMED
	  ARMED: if invalid look for failsafe
	         look for disarm signal.
	           if switch go to DISARMED
	           else store time and go to DISARMING
	  DISARMING: if arm signal ends return to ARMED
	             if time expires to go DISARMED_STILL_HOLDING
	  DISARMED_STILL_HOLDING: wait for release
	*/

	enum arm_state {
		ARM_STATE_DISARMED,
		ARM_STATE_ARMING,
		ARM_STATE_ARMED_STILL_HOLDING,
		ARM_STATE_ARMED,
		ARM_STATE_DISARMING,
		ARM_STATE_DISARMED_STILL_HOLDING
	};
	static enum arm_state arm_state = ARM_STATE_DISARMED;
	static uint32_t armedDisarmStart;

	valid &= cmd->Connected == MANUALCONTROLCOMMAND_CONNECTED_TRUE;

  	switch(arm_state) {
	case ARM_STATE_DISARMED:
	{
		set_armed_if_changed(FLIGHTSTATUS_ARMED_DISARMED);

		bool arm = arming_position(cmd, settings) && valid;

		if (arm && (settings->Arming == MANUALCONTROLSETTINGS_ARMING_SWITCH ||
		       settings->Arming == MANUALCONTROLSETTINGS_ARMING_SWITCHTHROTTLE)) {
			arm_state = ARM_STATE_ARMED;
		} else if (arm) {
			armedDisarmStart = lastSysTime;
			arm_state = ARM_STATE_ARMING;
		}
	}
		break;

  	case ARM_STATE_ARMING:
  	{
  		set_armed_if_changed(FLIGHTSTATUS_ARMED_ARMING);

  		bool arm = arming_position(cmd, settings) && valid;
		uint16_t arm_time = (settings->ArmTime == MANUALCONTROLSETTINGS_ARMTIME_250) ? 250 : \
			(settings->ArmTime == MANUALCONTROLSETTINGS_ARMTIME_500) ? 500 : \
			(settings->ArmTime == MANUALCONTROLSETTINGS_ARMTIME_1000) ? 1000 : \
			(settings->ArmTime == MANUALCONTROLSETTINGS_ARMTIME_2000) ? 2000 : 1000;
		if (arm && timeDifferenceMs(armedDisarmStart, lastSysTime) > arm_time) {
			arm_state = ARM_STATE_ARMED_STILL_HOLDING;
		} else if (!arm) {
			arm_state = ARM_STATE_DISARMED;
		}
	}
		break;

	case ARM_STATE_ARMED_STILL_HOLDING:
	{
		set_armed_if_changed(FLIGHTSTATUS_ARMED_ARMED);

		bool arm = arming_position(cmd, settings) && valid;
		if (!arm) {
			arm_state = ARM_STATE_ARMED;
		}
	}
		break;

	case ARM_STATE_ARMED:
	{
		set_armed_if_changed(FLIGHTSTATUS_ARMED_ARMED);

		// Determine whether to disarm when throttle is low
		uint8_t flight_mode;
		FlightStatusFlightModeGet(&flight_mode);
		bool autonomous_mode = 
		                       flight_mode == FLIGHTSTATUS_FLIGHTMODE_POSITIONHOLD ||
		                       flight_mode == FLIGHTSTATUS_FLIGHTMODE_RETURNTOHOME ||
		                       flight_mode == FLIGHTSTATUS_FLIGHTMODE_PATHPLANNER  ||
		                       flight_mode == FLIGHTSTATUS_FLIGHTMODE_TABLETCONTROL;
		bool check_throttle = settings->ArmTimeoutAutonomous == MANUALCONTROLSETTINGS_ARMTIMEOUTAUTONOMOUS_ENABLED ||
			!autonomous_mode;
		bool low_throttle = cmd->Throttle < 0;

		// Check for arming timeout if transmitter invalid or throttle is low and checked
		if (!valid || (low_throttle && check_throttle)) {
			if ((settings->ArmedTimeout != 0) && (timeDifferenceMs(armedDisarmStart, lastSysTime) > settings->ArmedTimeout))
				arm_state = ARM_STATE_DISARMED;
		} else {
			armedDisarmStart = lastSysTime;
 		}

  		bool disarm = disarming_position(cmd, settings) && valid;
		if (disarm && (settings->Arming == MANUALCONTROLSETTINGS_ARMING_SWITCH ||
		       settings->Arming == MANUALCONTROLSETTINGS_ARMING_SWITCHTHROTTLE)) {
			arm_state = ARM_STATE_DISARMED;
		} else if (disarm) {
			armedDisarmStart = lastSysTime;
			arm_state = ARM_STATE_DISARMING;
		}
	}
  		break;

	case ARM_STATE_DISARMING:
	{
  		bool disarm = disarming_position(cmd, settings) && valid;
		uint16_t disarm_time = (settings->DisarmTime == MANUALCONTROLSETTINGS_DISARMTIME_250) ? 250 : \
			(settings->DisarmTime == MANUALCONTROLSETTINGS_DISARMTIME_500) ? 500 : \
			(settings->DisarmTime == MANUALCONTROLSETTINGS_DISARMTIME_1000) ? 1000 : \
			(settings->DisarmTime == MANUALCONTROLSETTINGS_DISARMTIME_2000) ? 2000 : 1000;
  		if (disarm && timeDifferenceMs(armedDisarmStart, lastSysTime) > disarm_time) {
			arm_state = ARM_STATE_DISARMED_STILL_HOLDING;
  		} else if (!disarm) {
  			arm_state = ARM_STATE_ARMED;
  		}
  	}
		break;

	case ARM_STATE_DISARMED_STILL_HOLDING:
	{
  		set_armed_if_changed(FLIGHTSTATUS_ARMED_DISARMED);

  		bool disarm = disarming_position(cmd, settings) && valid;
  		if (!disarm) {
  			arm_state = ARM_STATE_DISARMED;
  		}
  	}
  		break;
  	}

}


//! Determine which of N positions the flight mode switch is in and set flight mode accordingly
static void set_flight_mode()
{
	uint8_t new_mode = transmitter_control_get_flight_mode();

	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);

	if (flightStatus.FlightMode != new_mode) {
		flightStatus.FlightMode = new_mode;
		FlightStatusSet(&flightStatus);
	}
}


static void resetRcvrActivity(struct rcvr_activity_fsm * fsm)
{
	ReceiverActivityData data;
	bool updated = false;

	/* Clear all channel activity flags */
	ReceiverActivityGet(&data);
	if (data.ActiveGroup != RECEIVERACTIVITY_ACTIVEGROUP_NONE &&
		data.ActiveChannel != 255) {
		data.ActiveGroup = RECEIVERACTIVITY_ACTIVEGROUP_NONE;
		data.ActiveChannel = 255;
		updated = true;
	}
	if (updated) {
		ReceiverActivitySet(&data);
	}

	/* Reset the FSM state */
	fsm->group        = 0;
	fsm->sample_count = 0;
}

static void updateRcvrActivitySample(uintptr_t rcvr_id, uint16_t samples[], uint8_t max_channels) {
	for (uint8_t channel = 1; channel <= max_channels; channel++) {
		// Subtract 1 because channels are 1 indexed
		samples[channel - 1] = PIOS_RCVR_Read(rcvr_id, channel);
	}
}

static bool updateRcvrActivityCompare(uintptr_t rcvr_id, struct rcvr_activity_fsm * fsm)
{
	bool activity_updated = false;

	/* Compare the current value to the previous sampled value */
	for (uint8_t channel = 1;
	     channel <= RCVR_ACTIVITY_MONITOR_CHANNELS_PER_GROUP;
	     channel++) {
		uint16_t delta;
		uint16_t prev = fsm->prev[channel - 1];   // Subtract 1 because channels are 1 indexed
		uint16_t curr = PIOS_RCVR_Read(rcvr_id, channel); 
		if (curr > prev) {
			delta = curr - prev;
		} else {
			delta = prev - curr;
		}

		if (delta > RCVR_ACTIVITY_MONITOR_MIN_RANGE) {
			/* Mark this channel as active */
			ReceiverActivityActiveGroupOptions group;

			/* Don't assume manualcontrolsettings and receiveractivity are in the same order. */
			switch (fsm->group) {
			case MANUALCONTROLSETTINGS_CHANNELGROUPS_PWM: 
				group = RECEIVERACTIVITY_ACTIVEGROUP_PWM;
				break;
			case MANUALCONTROLSETTINGS_CHANNELGROUPS_PPM:
				group = RECEIVERACTIVITY_ACTIVEGROUP_PPM;
				break;
			case MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMMAINPORT:
				group = RECEIVERACTIVITY_ACTIVEGROUP_DSMMAINPORT;
				break;
			case MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMFLEXIPORT:
				group = RECEIVERACTIVITY_ACTIVEGROUP_DSMFLEXIPORT;
				break;
			case MANUALCONTROLSETTINGS_CHANNELGROUPS_DSMRCVRPORT:
				group = RECEIVERACTIVITY_ACTIVEGROUP_DSMRCVRPORT;
				break;
			case MANUALCONTROLSETTINGS_CHANNELGROUPS_SBUS:
				group = RECEIVERACTIVITY_ACTIVEGROUP_SBUS;
				break;
			case MANUALCONTROLSETTINGS_CHANNELGROUPS_RFM22B:
				group = RECEIVERACTIVITY_ACTIVEGROUP_RFM22B;
				break;
			case MANUALCONTROLSETTINGS_CHANNELGROUPS_OPENLRS:
				group = RECEIVERACTIVITY_ACTIVEGROUP_OPENLRS;
				break;
			case MANUALCONTROLSETTINGS_CHANNELGROUPS_GCS:
				group = RECEIVERACTIVITY_ACTIVEGROUP_GCS;
				break;
			case MANUALCONTROLSETTINGS_CHANNELGROUPS_HOTTSUM:
				group = RECEIVERACTIVITY_ACTIVEGROUP_HOTTSUM;
				break;
			default:
				set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_UNDEFINED);
				group = RECEIVERACTIVITY_ACTIVEGROUP_PWM;
				break;
			}

			ReceiverActivityActiveGroupSet((uint8_t*)&group);
			ReceiverActivityActiveChannelSet(&channel);
			activity_updated = true;
		}
	}
	return (activity_updated);
}

static bool updateRcvrActivity(struct rcvr_activity_fsm * fsm)
{
	bool activity_updated = false;

	if (fsm->group >= MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE) {
		/* We're out of range, reset things */
		resetRcvrActivity(fsm);
	}

	extern uintptr_t pios_rcvr_group_map[];
	if (!pios_rcvr_group_map[fsm->group]) {
		/* Unbound group, skip it */
		goto group_completed;
	}

	if (fsm->sample_count == 0) {
		/* Take a sample of each channel in this group */
		updateRcvrActivitySample(pios_rcvr_group_map[fsm->group],
					fsm->prev,
					NELEMENTS(fsm->prev));
		fsm->sample_count++;
		return (false);
	}

	/* Compare with previous sample */
	activity_updated = updateRcvrActivityCompare(pios_rcvr_group_map[fsm->group], fsm);

group_completed:
	/* Reset the sample counter */
	fsm->sample_count = 0;

	/* Find the next active group, but limit search so we can't loop forever here */
	for (uint8_t i = 0; i < MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE; i++) {
		/* Move to the next group */
		fsm->group++;
		if (fsm->group >= MANUALCONTROLSETTINGS_CHANNELGROUPS_NONE) {
			/* Wrap back to the first group */
			fsm->group = 0;
		}
		if (pios_rcvr_group_map[fsm->group]) {
			/* 
			 * Found an active group, take a sample here to avoid an
			 * extra 20ms delay in the main thread so we can speed up
			 * this algorithm.
			 */
			updateRcvrActivitySample(pios_rcvr_group_map[fsm->group],
						fsm->prev,
						NELEMENTS(fsm->prev));
			fsm->sample_count++;
			break;
		}
	}

	return (activity_updated);
}

//! In manual mode directly set actuator desired
static void update_actuator_desired(ManualControlCommandData * cmd)
{
	ActuatorDesiredData actuator;
	ActuatorDesiredGet(&actuator);
	actuator.Roll = cmd->Roll;
	actuator.Pitch = cmd->Pitch;
	actuator.Yaw = cmd->Yaw;
	actuator.Throttle = (cmd->Throttle < 0) ? -1 : cmd->Throttle;
	ActuatorDesiredSet(&actuator);
}

//! In stabilization mode, set stabilization desired
static void update_stabilization_desired(ManualControlCommandData * cmd, ManualControlSettingsData * settings)
{
	StabilizationDesiredData stabilization;
	StabilizationDesiredGet(&stabilization);

	StabilizationSettingsData stabSettings;
	StabilizationSettingsGet(&stabSettings);

	const uint8_t RATE_SETTINGS[3] = {  STABILIZATIONDESIRED_STABILIZATIONMODE_RATE,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_RATE,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK};
	const uint8_t ATTITUDE_SETTINGS[3] = {
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK};
	const uint8_t VIRTUALBAR_SETTINGS[3] = {
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_VIRTUALBAR,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_VIRTUALBAR,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK};
	const uint8_t MWRATE_SETTINGS[3] = {
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_MWRATE,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_MWRATE,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_MWRATE};
	const uint8_t HORIZON_SETTINGS[3] = {
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_HORIZON,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_HORIZON,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK};
	const uint8_t AXISLOCK_SETTINGS[3] = {
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK,
	                                    STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK};

	const uint8_t * stab_settings;
	FlightStatusData flightStatus;
	FlightStatusGet(&flightStatus);
	switch(flightStatus.FlightMode) {
		case FLIGHTSTATUS_FLIGHTMODE_ACRO:
			stab_settings = RATE_SETTINGS;
			break;
		case FLIGHTSTATUS_FLIGHTMODE_LEVELING:
			stab_settings = ATTITUDE_SETTINGS;
			break;
		case FLIGHTSTATUS_FLIGHTMODE_VIRTUALBAR:
			stab_settings = VIRTUALBAR_SETTINGS;
			break;
		case FLIGHTSTATUS_FLIGHTMODE_MWRATE:
			stab_settings = MWRATE_SETTINGS;
			break;
		case FLIGHTSTATUS_FLIGHTMODE_HORIZON:
			stab_settings = HORIZON_SETTINGS;
			break;
		case FLIGHTSTATUS_FLIGHTMODE_AXISLOCK:
			stab_settings = AXISLOCK_SETTINGS;
			break;
		case FLIGHTSTATUS_FLIGHTMODE_STABILIZED1:
			stab_settings = settings->Stabilization1Settings;
			break;
		case FLIGHTSTATUS_FLIGHTMODE_STABILIZED2:
			stab_settings = settings->Stabilization2Settings;
			break;
		case FLIGHTSTATUS_FLIGHTMODE_STABILIZED3:
			stab_settings = settings->Stabilization3Settings;
			break;
		default:
			{
				// Major error, this should not occur because only enter this block when one of these is true
				set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_UNDEFINED);
			}
			return;
	}

	// TOOD: Add assumption about order of stabilization desired and manual control stabilization mode fields having same order
	stabilization.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_ROLL]  = stab_settings[0];
	stabilization.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_PITCH] = stab_settings[1];
	stabilization.StabilizationMode[STABILIZATIONDESIRED_STABILIZATIONMODE_YAW]   = stab_settings[2];

	stabilization.Roll = (stab_settings[0] == STABILIZATIONDESIRED_STABILIZATIONMODE_NONE) ? cmd->Roll :
	     (stab_settings[0] == STABILIZATIONDESIRED_STABILIZATIONMODE_RATE) ? expo3(cmd->Roll, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_ROLL]) * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_ROLL] :
	     (stab_settings[0] == STABILIZATIONDESIRED_STABILIZATIONMODE_WEAKLEVELING) ? expo3(cmd->Roll, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_ROLL]) * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_ROLL]:
	     (stab_settings[0] == STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE) ? cmd->Roll * stabSettings.RollMax :
	     (stab_settings[0] == STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK) ? expo3(cmd->Roll, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_ROLL]) * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_ROLL] :
	     (stab_settings[0] == STABILIZATIONDESIRED_STABILIZATIONMODE_VIRTUALBAR) ? cmd->Roll :
	     (stab_settings[0] == STABILIZATIONDESIRED_STABILIZATIONMODE_HORIZON) ? expo3(cmd->Roll, stabSettings.HorizonExpo[STABILIZATIONSETTINGS_HORIZONEXPO_ROLL]) :
	     (stab_settings[0] == STABILIZATIONDESIRED_STABILIZATIONMODE_MWRATE) ? expo3(cmd->Roll, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_ROLL]) :
	     (stab_settings[0] == STABILIZATIONDESIRED_STABILIZATIONMODE_SYSTEMIDENT) ? cmd->Roll * stabSettings.RollMax :
	     (stab_settings[0] == STABILIZATIONDESIRED_STABILIZATIONMODE_COORDINATEDFLIGHT) ? cmd->Roll :
	     0; // this is an invalid mode

	stabilization.Pitch = (stab_settings[1] == STABILIZATIONDESIRED_STABILIZATIONMODE_NONE) ? cmd->Pitch :
	     (stab_settings[1] == STABILIZATIONDESIRED_STABILIZATIONMODE_RATE) ? expo3(cmd->Pitch, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_PITCH]) * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_PITCH] :
	     (stab_settings[1] == STABILIZATIONDESIRED_STABILIZATIONMODE_WEAKLEVELING) ? expo3(cmd->Pitch, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_PITCH]) * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_PITCH] :
	     (stab_settings[1] == STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE) ? cmd->Pitch * stabSettings.PitchMax :
	     (stab_settings[1] == STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK) ? expo3(cmd->Pitch, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_PITCH]) * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_PITCH] :
	     (stab_settings[1] == STABILIZATIONDESIRED_STABILIZATIONMODE_VIRTUALBAR) ? cmd->Pitch :
	     (stab_settings[1] == STABILIZATIONDESIRED_STABILIZATIONMODE_HORIZON) ? expo3(cmd->Pitch, stabSettings.HorizonExpo[STABILIZATIONSETTINGS_HORIZONEXPO_PITCH]):
	     (stab_settings[1] == STABILIZATIONDESIRED_STABILIZATIONMODE_MWRATE) ? expo3(cmd->Pitch, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_PITCH]) :
	     (stab_settings[1] == STABILIZATIONDESIRED_STABILIZATIONMODE_SYSTEMIDENT) ? cmd->Pitch * stabSettings.PitchMax :
	     (stab_settings[1] == STABILIZATIONDESIRED_STABILIZATIONMODE_COORDINATEDFLIGHT) ? cmd->Pitch :
	     0; // this is an invalid mode

	stabilization.Yaw = (stab_settings[2] == STABILIZATIONDESIRED_STABILIZATIONMODE_NONE) ? cmd->Yaw :
	     (stab_settings[2] == STABILIZATIONDESIRED_STABILIZATIONMODE_RATE) ? expo3(cmd->Yaw, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_YAW]) * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_YAW] :
	     (stab_settings[2] == STABILIZATIONDESIRED_STABILIZATIONMODE_WEAKLEVELING) ? expo3(cmd->Yaw, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_YAW]) * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_YAW] :
	     (stab_settings[2] == STABILIZATIONDESIRED_STABILIZATIONMODE_ATTITUDE) ? cmd->Yaw * stabSettings.YawMax :
	     (stab_settings[2] == STABILIZATIONDESIRED_STABILIZATIONMODE_AXISLOCK) ? expo3(cmd->Yaw, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_YAW]) * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_YAW] :
	     (stab_settings[2] == STABILIZATIONDESIRED_STABILIZATIONMODE_VIRTUALBAR) ? cmd->Yaw :
	     (stab_settings[2] == STABILIZATIONDESIRED_STABILIZATIONMODE_HORIZON) ? expo3(cmd->Yaw, stabSettings.HorizonExpo[STABILIZATIONSETTINGS_HORIZONEXPO_YAW]) :
	     (stab_settings[2] == STABILIZATIONDESIRED_STABILIZATIONMODE_MWRATE) ? expo3(cmd->Yaw, stabSettings.RateExpo[STABILIZATIONSETTINGS_RATEEXPO_YAW]) :
	     (stab_settings[2] == STABILIZATIONDESIRED_STABILIZATIONMODE_SYSTEMIDENT) ? cmd->Yaw * stabSettings.YawMax :
	     (stab_settings[2] == STABILIZATIONDESIRED_STABILIZATIONMODE_COORDINATEDFLIGHT) ? cmd->Yaw :
	     0; // this is an invalid mode

	stabilization.Throttle = (cmd->Throttle < 0) ? -1 : cmd->Throttle;
	StabilizationDesiredSet(&stabilization);
}

#if !defined(COPTERCONTROL) && !defined(GIMBAL)

/**
 * @brief Update the altitude desired to current altitude when
 * enabled and enable altitude mode for stabilization
 * @todo: Need compile flag to exclude this from copter control
 */
static void altitude_hold_desired(ManualControlCommandData * cmd, bool flightModeChanged)
{
	if (AltitudeHoldDesiredHandle() == NULL) {
		set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_ALTITUDEHOLD);
		return;
	}

	const float MIN_CLIMB_RATE = 0.01f;
	
	AltitudeHoldDesiredData altitudeHoldDesired;
	AltitudeHoldDesiredGet(&altitudeHoldDesired);

	StabilizationSettingsData stabSettings;
	StabilizationSettingsGet(&stabSettings);

	altitudeHoldDesired.Roll = cmd->Roll * stabSettings.RollMax;
	altitudeHoldDesired.Pitch = cmd->Pitch * stabSettings.PitchMax;
	altitudeHoldDesired.Yaw = cmd->Yaw * stabSettings.ManualRate[STABILIZATIONSETTINGS_MANUALRATE_YAW];
	
	float current_down;
	PositionActualDownGet(&current_down);

	if(flightModeChanged) {
		// Initialize at the current location. Note that this object uses the up is positive
		// convention.
		altitudeHoldDesired.Altitude = -current_down;
		altitudeHoldDesired.ClimbRate = 0;
	} else {
		uint8_t altitude_hold_expo, altitude_hold_maxrate, altitude_hold_deadband;
		AltitudeHoldSettingsMaxRateGet(&altitude_hold_maxrate);
		AltitudeHoldSettingsExpoGet(&altitude_hold_expo);
		AltitudeHoldSettingsDeadbandGet(&altitude_hold_deadband);

		const float DEADBAND_HIGH = 0.50f + altitude_hold_deadband * 0.01f;
		const float DEADBAND_LOW = 0.50f - altitude_hold_deadband * 0.01f;

		float climb_rate = 0.0f;
		if (cmd->Throttle > DEADBAND_HIGH) {
			climb_rate = expo3((cmd->Throttle - DEADBAND_HIGH) / (1.0f - DEADBAND_HIGH), altitude_hold_expo) *
		                         altitude_hold_maxrate;
		} else if (cmd->Throttle < DEADBAND_LOW && altitude_hold_maxrate > MIN_CLIMB_RATE) {
			climb_rate = ((cmd->Throttle < 0) ? DEADBAND_LOW : DEADBAND_LOW - cmd->Throttle) / DEADBAND_LOW;
			climb_rate = -expo3(climb_rate, altitude_hold_expo) * altitude_hold_maxrate;
		}

		// When throttle is negative tell the module that we are in landing mode
		altitudeHoldDesired.Land = (cmd->Throttle < 0) ? ALTITUDEHOLDDESIRED_LAND_TRUE : ALTITUDEHOLDDESIRED_LAND_FALSE;

		// If more than MIN_CLIMB_RATE enter vario mode
		if (fabsf(climb_rate) > MIN_CLIMB_RATE) {
			// Desired state is at the current location with the requested rate
			altitudeHoldDesired.Altitude = -current_down;
			altitudeHoldDesired.ClimbRate = climb_rate;
		} else {
			// Here we intentionally do not change the set point, it will
			// remain where the user released vario mode
			altitudeHoldDesired.ClimbRate = 0.0f;
		}
	}

	// Must always set since this contains the control signals
	AltitudeHoldDesiredSet(&altitudeHoldDesired);
}


static void set_loiter_command(ManualControlCommandData *cmd)
{
	const float CMD_THRESHOLD = 0.5f;
	const float MAX_SPEED     = 3.0f; // m/s

	LoiterCommandData loiterCommand;
	loiterCommand.Forward = (cmd->Pitch > CMD_THRESHOLD) ? cmd->Pitch - CMD_THRESHOLD :
	                        (cmd->Pitch < -CMD_THRESHOLD) ? cmd->Pitch + CMD_THRESHOLD :
	                        0;
	// Note the negative - forward pitch is negative
	loiterCommand.Forward *= -MAX_SPEED / (1.0f - CMD_THRESHOLD);

	loiterCommand.Right = (cmd->Roll > CMD_THRESHOLD) ? cmd->Roll - CMD_THRESHOLD :
	                      (cmd->Roll < -CMD_THRESHOLD) ? cmd->Roll + CMD_THRESHOLD :
	                      0;
	loiterCommand.Right *= MAX_SPEED / (1.0f - CMD_THRESHOLD);

	loiterCommand.Frame = LOITERCOMMAND_FRAME_BODY;

	LoiterCommandSet(&loiterCommand);
}

#else /* For boards that do not support navigation set error if these modes are selected */

static void altitude_hold_desired(ManualControlCommandData * cmd, bool flightModeChanged)
{
	set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_ALTITUDEHOLD);
}

static void set_loiter_command(ManualControlCommandData *cmd)
{
	set_manual_control_error(SYSTEMALARMS_MANUALCONTROL_PATHFOLLOWER);
}

#endif /* !defined(COPTERCONTROL) && !defined(GIMBAL) */

/**
 * Convert channel from servo pulse duration (microseconds) to scaled -1/+1 range.
 */
static float scaleChannel(int16_t value, int16_t max, int16_t min, int16_t neutral)
{
	float valueScaled;

	// Scale
	if ((max > min && value >= neutral) || (min > max && value <= neutral))
	{
		if (max != neutral)
			valueScaled = (float)(value - neutral) / (float)(max - neutral);
		else
			valueScaled = 0;
	}
	else
	{
		if (min != neutral)
			valueScaled = (float)(value - neutral) / (float)(neutral - min);
		else
			valueScaled = 0;
	}

	// Bound
	if (valueScaled >  1.0f) valueScaled =  1.0f;
	else
	if (valueScaled < -1.0f) valueScaled = -1.0f;

	return valueScaled;
}

static uint32_t timeDifferenceMs(uint32_t start_time, uint32_t end_time) {
	if (end_time >= start_time)
		return end_time - start_time;
	else
		return (uint32_t)PIOS_THREAD_TIMEOUT_MAX - start_time + end_time + 1;
}

/**
 * @brief Determine if the manual input value is within acceptable limits
 * @returns return TRUE if so, otherwise return FALSE
 */
bool validInputRange(int16_t min, int16_t max, uint16_t value, uint16_t offset)
{
	if (min > max)
	{
		int16_t tmp = min;
		min = max;
		max = tmp;
	}
	return (value >= min - offset && value <= max + offset);
}

/**
 * @brief Apply deadband to Roll/Pitch/Yaw channels
 */
static void applyDeadband(float *value, float deadband)
{
	if (fabsf(*value) < deadband)
		*value = 0.0f;
	else
		if (*value > 0.0f)
			*value -= deadband;
		else
			*value += deadband;
}

//! Update the manual control settings
static void manual_control_settings_updated(UAVObjEvent * ev)
{
	settings_updated = true;
}

/**
 * Set the error code and alarm state
 * @param[in] error code
 */
static void set_manual_control_error(SystemAlarmsManualControlOptions error_code)
{
	// Get the severity of the alarm given the error code
	SystemAlarmsAlarmOptions severity;
	switch (error_code) {
	case SYSTEMALARMS_MANUALCONTROL_NONE:
		severity = SYSTEMALARMS_ALARM_OK;
		break;
	case SYSTEMALARMS_MANUALCONTROL_NORX:
	case SYSTEMALARMS_MANUALCONTROL_ACCESSORY:
		severity = SYSTEMALARMS_ALARM_WARNING;
		break;
	case SYSTEMALARMS_MANUALCONTROL_SETTINGS:
		severity = SYSTEMALARMS_ALARM_CRITICAL;
		break;
	case SYSTEMALARMS_MANUALCONTROL_ALTITUDEHOLD:
		severity = SYSTEMALARMS_ALARM_ERROR;
		break;
	case SYSTEMALARMS_MANUALCONTROL_UNDEFINED:
	default:
		severity = SYSTEMALARMS_ALARM_CRITICAL;
		error_code = SYSTEMALARMS_MANUALCONTROL_UNDEFINED;
	}

	// Make sure not to set the error code if it didn't change
	SystemAlarmsManualControlOptions current_error_code;
	SystemAlarmsManualControlGet((uint8_t *) &current_error_code);
	if (current_error_code != error_code) {
		SystemAlarmsManualControlSet((uint8_t *) &error_code);
	}

	// AlarmSet checks only updates on toggle
	AlarmsSet(SYSTEMALARMS_ALARM_MANUALCONTROL, (uint8_t) severity);
}

/**
  * @}
  * @}
  */
