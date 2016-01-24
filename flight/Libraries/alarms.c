/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       alarms.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @brief      Library for setting and clearing system alarms
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
#include "alarms.h"
#include "pios_mutex.h"
#include "pios_reset.h"

// Private constants

// Private types

// Private variables
static struct pios_mutex *lock;

// Private functions
static int32_t hasSeverity(SystemAlarmsAlarmOptions severity);

/**
 * Initialize the alarms library
 */
int32_t AlarmsInitialize(void)
{
	SystemAlarmsInitialize();
	lock = PIOS_Mutex_Create();
	PIOS_Assert(lock != NULL);

	uint8_t reboot_reason = SYSTEMALARMS_REBOOTCAUSE_UNDEFINED;

	switch (PIOS_RESET_GetResetReason()) {
	case PIOS_RESET_FLAG_BROWNOUT: // Brownouts are not detected on STM32F1 or STM32F3
		reboot_reason = SYSTEMALARMS_REBOOTCAUSE_BROWNOUT;
		break;
	case PIOS_RESET_FLAG_PIN:
		reboot_reason = SYSTEMALARMS_REBOOTCAUSE_PINRESET;
		break;
	case PIOS_RESET_FLAG_POWERON:
		reboot_reason = SYSTEMALARMS_REBOOTCAUSE_POWERONRESET;
		break;
	case PIOS_RESET_FLAG_SOFTWARE:
		reboot_reason = SYSTEMALARMS_REBOOTCAUSE_SOFTWARERESET;
		break;
	case PIOS_RESET_FLAG_INDEPENDENT_WATCHDOG:
		reboot_reason = SYSTEMALARMS_REBOOTCAUSE_INDEPENDENTWATCHDOG;
		break;
	case PIOS_RESET_FLAG_WINDOW_WATCHDOG:
		reboot_reason = SYSTEMALARMS_REBOOTCAUSE_WINDOWWATCHDOG;
		break;
	case PIOS_RESET_FLAG_LOW_POWER:
		reboot_reason = SYSTEMALARMS_REBOOTCAUSE_LOWPOWER;
		break;
	case PIOS_RESET_FLAG_UNDEFINED:
		 reboot_reason = SYSTEMALARMS_REBOOTCAUSE_UNDEFINED;
		break;
	}

	SystemAlarmsRebootCauseSet(&reboot_reason);

	return 0;
}

/**
 * Set an alarm
 * @param alarm The system alarm to be modified
 * @param severity The alarm severity
 * @return 0 if success, -1 if an error
 */
int32_t AlarmsSet(SystemAlarmsAlarmElem alarm, SystemAlarmsAlarmOptions severity)
{
	SystemAlarmsData alarms;

	// Check that this is a valid alarm
	if (alarm >= SYSTEMALARMS_ALARM_NUMELEM)
	{
		return -1;
	}

	// Lock
	PIOS_Mutex_Lock(lock, PIOS_MUTEX_TIMEOUT_MAX);

    // Read alarm and update its severity only if it was changed
    SystemAlarmsGet(&alarms);
    if ( alarms.Alarm[alarm] != severity )
    {
    	alarms.Alarm[alarm] = severity;
    	SystemAlarmsSet(&alarms);
    }

    // Release lock
    PIOS_Mutex_Unlock(lock);
    return 0;

}

/**
 * Get an alarm
 * @param alarm The system alarm to be read
 * @return Alarm severity
 */
SystemAlarmsAlarmOptions AlarmsGet(SystemAlarmsAlarmElem alarm)
{
	SystemAlarmsData alarms;

	// Check that this is a valid alarm
	if (alarm >= SYSTEMALARMS_ALARM_NUMELEM)
	{
		return 0;
	}

    // Read alarm
    SystemAlarmsGet(&alarms);
    return alarms.Alarm[alarm];
}

/**
 * Set an alarm to it's default value
 * @param alarm The system alarm to be modified
 * @return 0 if success, -1 if an error
 */
int32_t AlarmsDefault(SystemAlarmsAlarmElem alarm)
{
	return AlarmsSet(alarm, SYSTEMALARMS_ALARM_DEFAULT);
}

/**
 * Default all alarms
 */
void AlarmsDefaultAll()
{
	uint32_t n;
    for (n = 0; n < SYSTEMALARMS_ALARM_NUMELEM; ++n)
    {
    	AlarmsDefault(n);
    }
}

/**
 * Clear an alarm
 * @param alarm The system alarm to be modified
 * @return 0 if success, -1 if an error
 */
int32_t AlarmsClear(SystemAlarmsAlarmElem alarm)
{
	return AlarmsSet(alarm, SYSTEMALARMS_ALARM_OK);
}

/**
 * Clear all alarms
 */
void AlarmsClearAll()
{
	uint32_t n;
    for (n = 0; n < SYSTEMALARMS_ALARM_NUMELEM; ++n)
    {
    	AlarmsClear(n);
    }
}

/**
 * Check if there are any alarms with the given or higher severity
 * @return 0 if no alarms are found, 1 if at least one alarm is found
 */
int32_t AlarmsHasWarnings()
{
	return hasSeverity(SYSTEMALARMS_ALARM_WARNING);
}

/**
 * Check if there are any alarms with error or higher severity
 * @return 0 if no alarms are found, 1 if at least one alarm is found
 */
int32_t AlarmsHasErrors()
{
	return hasSeverity(SYSTEMALARMS_ALARM_ERROR);
};

/**
 * Check if there are any alarms with critical or higher severity
 * @return 0 if no alarms are found, 1 if at least one alarm is found
 */
int32_t AlarmsHasCritical()
{
	return hasSeverity(SYSTEMALARMS_ALARM_CRITICAL);
};

/**
 * Check if there are any alarms with the given or higher severity
 * @return 0 if no alarms are found, 1 if at least one alarm is found
 */
static int32_t hasSeverity(SystemAlarmsAlarmOptions severity)
{
	SystemAlarmsData alarms;
	uint32_t n;

	// Lock
	PIOS_Mutex_Lock(lock, PIOS_MUTEX_TIMEOUT_MAX);

    // Read alarms
    SystemAlarmsGet(&alarms);

    // Go through alarms and check if any are of the given severity or higher
    for (n = 0; n < SYSTEMALARMS_ALARM_NUMELEM; ++n)
    {
    	if ( alarms.Alarm[n] >= severity)
    	{
    		PIOS_Mutex_Unlock(lock);
    		return 1;
    	}
    }

    // If this point is reached then no alarms found
	PIOS_Mutex_Unlock(lock);
    return 0;
}

static const char alarm_names[][9] = {
	[SYSTEMALARMS_ALARM_OUTOFMEMORY] = "MEMORY",
	[SYSTEMALARMS_ALARM_CPUOVERLOAD] = "CPU",
	[SYSTEMALARMS_ALARM_STACKOVERFLOW] = "STACK",
	[SYSTEMALARMS_ALARM_SYSTEMCONFIGURATION] = "CONFIG",
	[SYSTEMALARMS_ALARM_EVENTSYSTEM] = "EVENT",
	[SYSTEMALARMS_ALARM_TELEMETRY] = {0}, // ignored
	[SYSTEMALARMS_ALARM_MANUALCONTROL] = "MANUAL",
	[SYSTEMALARMS_ALARM_ACTUATOR] = "ACTUATOR",
	[SYSTEMALARMS_ALARM_ATTITUDE] = "ATTITUDE",
	[SYSTEMALARMS_ALARM_SENSORS] = "SENSORS",
	[SYSTEMALARMS_ALARM_STABILIZATION] = "STAB",
	[SYSTEMALARMS_ALARM_PATHFOLLOWER] = "PATH-F",
	[SYSTEMALARMS_ALARM_PATHPLANNER] = "PATH-P",
	[SYSTEMALARMS_ALARM_BATTERY] = "BATTERY",
	[SYSTEMALARMS_ALARM_FLIGHTTIME] = "TIME",
	[SYSTEMALARMS_ALARM_I2C] = "I2C",
	[SYSTEMALARMS_ALARM_GPS] = "GPS",
	[SYSTEMALARMS_ALARM_ALTITUDEHOLD] = "A-HOLD",
	[SYSTEMALARMS_ALARM_BOOTFAULT] = "BOOT",
	[SYSTEMALARMS_ALARM_GEOFENCE] = "GEOFENCE",
	[SYSTEMALARMS_ALARM_TEMPBARO] = "TEMPBARO",
	[SYSTEMALARMS_ALARM_GYROBIAS] = "GYROBIAS",
	[SYSTEMALARMS_ALARM_ADC] = "ADC",
};

// If someone adds a new alarm, we'd like it added to the array above.
DONT_BUILD_IF(NELEMENTS(alarm_names) != SYSTEMALARMS_ALARM_NUMELEM, AlarmArrayMismatch);

static const char config_error_names[][15] = {
	[SYSTEMALARMS_CONFIGERROR_STABILIZATION] = "CFG:STAB",
	[SYSTEMALARMS_CONFIGERROR_MULTIROTOR] = "CFG:MULTIROTOR",
	[SYSTEMALARMS_CONFIGERROR_AUTOTUNE] = "CFG:AUTOTUNE",
	[SYSTEMALARMS_CONFIGERROR_ALTITUDEHOLD] = "CFG:AH1",
	[SYSTEMALARMS_CONFIGERROR_POSITIONHOLD] = "CFG:POS-HOLD",
	[SYSTEMALARMS_CONFIGERROR_PATHPLANNER] = "CFG:PATHPLAN",
	[SYSTEMALARMS_CONFIGERROR_DUPLICATEPORTCFG] = "CFG:DUP PORT",
	[SYSTEMALARMS_CONFIGERROR_NAVFILTER] = "CFG:NAVFILTER",
	[SYSTEMALARMS_CONFIGERROR_UNSAFETOARM] = "CFG:UNSAFE",
	[SYSTEMALARMS_CONFIGERROR_UNDEFINED] = "CFG:UNDEF",
	[SYSTEMALARMS_CONFIGERROR_NONE] = {0},
};

// DONT_BUILD_IF(NELEMENTS(CONFIG_ERROR_NAMES) != SYSTEMALARMS_CONFIGERROR_NUMELEM, AlarmArrayMismatch);

static const char manual_control_names[][13] = {
	[SYSTEMALARMS_MANUALCONTROL_SETTINGS] = "MAN:SETTINGS",
	[SYSTEMALARMS_MANUALCONTROL_NORX] = "MAN:NO RX",
	[SYSTEMALARMS_MANUALCONTROL_ACCESSORY] = "MAN:ACC",
	[SYSTEMALARMS_MANUALCONTROL_ALTITUDEHOLD] = "MAN:A-HOLD",
	[SYSTEMALARMS_MANUALCONTROL_PATHFOLLOWER] = "MAN:PATH-F",
	[SYSTEMALARMS_MANUALCONTROL_UNDEFINED] = "MAN:UNDEF",
	[SYSTEMALARMS_MANUALCONTROL_NONE] = {0},
};

// DONT_BUILD_IF(NELEMENTS(MANUAL_CONTROL_NAMES) != SYSTEMALARMS_MANUALCONTROL_NUMELEM, AlarmArrayMismatch);

static const char boot_reason_names[][16] = {
	[SYSTEMALARMS_REBOOTCAUSE_BROWNOUT] = "BOOT:BROWNOUT",
	[SYSTEMALARMS_REBOOTCAUSE_PINRESET] = "BOOT:PIN RESET",
	[SYSTEMALARMS_REBOOTCAUSE_POWERONRESET] = "BOOT:PWR ON RST",
	[SYSTEMALARMS_REBOOTCAUSE_SOFTWARERESET] = "BOOT:SW RESET",
	[SYSTEMALARMS_REBOOTCAUSE_INDEPENDENTWATCHDOG] "BOOT:INDY WDOG",
	[SYSTEMALARMS_REBOOTCAUSE_WINDOWWATCHDOG] = "BOOT:WIN WDOG",
	[SYSTEMALARMS_REBOOTCAUSE_LOWPOWER] = "BOOT:LOW POWER",
	[SYSTEMALARMS_REBOOTCAUSE_UNDEFINED] = "BOOT:UNDEFINED",
};

#define LONGEST_MESSAGE 17

DONT_BUILD_IF((LONGEST_MESSAGE <= sizeof(*config_error_names)
	       || LONGEST_MESSAGE <= sizeof(*manual_control_names)
	       || LONGEST_MESSAGE <= sizeof(*boot_reason_names)),
	      InsufficientBufferage);

const char *AlarmBootReason(uint8_t reason) {
	if (reason >= NELEMENTS(boot_reason_names)) {
		return (const char*)&boot_reason_names[SYSTEMALARMS_REBOOTCAUSE_UNDEFINED];
	}
	return (const char*)&boot_reason_names[reason];
}

int32_t AlarmString(SystemAlarmsData *alarm, char *buf, size_t buflen, bool blink, uint8_t *state) {
	*state = SYSTEMALARMS_ALARM_OK;
	buf[0] = '\0';
	int pos = 0;

	// TODO(dustin): sort the alarms by severity.  Alarm messages
	// will get truncated.  We want the most urgent stuff to show
	// up first, having warnings show up only if there's space.
	for (int i = 0; i < SYSTEMALARMS_ALARM_NUMELEM; i++) {
		if (((alarm->Alarm[i] == SYSTEMALARMS_ALARM_WARNING) ||
		     (alarm->Alarm[i] == SYSTEMALARMS_ALARM_ERROR) ||
		     (alarm->Alarm[i] == SYSTEMALARMS_ALARM_CRITICAL))) {

			if (!alarm_names[i][0]) {
				// zero-length alarm names indicate the alarm is
				// explicitly ignored
				continue;
			}

			// Returned state is the worst state.
			if (alarm->Alarm[i] > *state) {
				*state = alarm->Alarm[i];
			}

			char current_msg[LONGEST_MESSAGE+1] = {0};
			switch (i) {
			case SYSTEMALARMS_ALARM_SYSTEMCONFIGURATION:
				strncpy(current_msg,
					(const char*)config_error_names[alarm->ConfigError],
					sizeof(*config_error_names));
				current_msg[sizeof(*config_error_names)] = '\0';
				break;
			case SYSTEMALARMS_ALARM_MANUALCONTROL:
				strncpy(current_msg,
					(const char*)manual_control_names[alarm->ManualControl],
					sizeof(*manual_control_names));
				current_msg[sizeof(*manual_control_names)] = '\0';
				break;
			default:
				strncpy(current_msg, (const char*)alarm_names[i], sizeof(*alarm_names));
				current_msg[sizeof(*alarm_names)] = '\0';
			}

			int this_len = strlen(current_msg);

			if (pos + this_len + 2 >= buflen) {
				break;
			}

			if ((alarm->Alarm[i] != SYSTEMALARMS_ALARM_WARNING) && !blink) {
				this_len += 1;
				while (this_len > 0) {
					buf[pos++] = ' ';
					this_len--;
				}
				continue;
			}

			memcpy(&buf[pos], current_msg, this_len);
			pos += this_len;
			buf[pos++] = ' ';

		}
	}

	buf[pos] = '\0';
	return pos;
}

/**
 * @}
 */

