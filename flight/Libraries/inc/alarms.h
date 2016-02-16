/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       alarms.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      Include file of the alarm library
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
#ifndef ALARMS_H
#define ALARMS_H

#include "systemalarms.h"
#define SYSTEMALARMS_ALARM_DEFAULT SYSTEMALARMS_ALARM_UNINITIALISED

int32_t AlarmsInitialize(void);
int32_t AlarmsSet(SystemAlarmsAlarmElem alarm, SystemAlarmsAlarmOptions severity);
SystemAlarmsAlarmOptions AlarmsGet(SystemAlarmsAlarmElem alarm);
int32_t AlarmsDefault(SystemAlarmsAlarmElem alarm);
void AlarmsDefaultAll();
int32_t AlarmsClear(SystemAlarmsAlarmElem alarm);
void AlarmsClearAll();
int32_t AlarmsHasWarnings();
int32_t AlarmsHasErrors();
int32_t AlarmsHasCritical();
/** Produce a string indicating what alarms are currently firing.
 * @param[alarm] the current alarm state (from SystemAlarmsGet).
 * @param[buf] where the alarm string should be written.
 * @param[buflen] how many bytes may be safely written into buf.
 * @param[blink] if true, alarms are replaced with spaces.
 * @param[state] output variable indicating the most severe alarm found.
 * The output variable 'state' is a value from the SystemAlarms alarm enum
 * (e.g., SYSTEMALARMS_ALARM_WARNING).
 * @returns The number of bytes written to the buffer.
 */
int32_t AlarmString(SystemAlarmsData *alarm, char *buf, size_t buflen,
		    bool blink, uint8_t *state);
const char *AlarmBootReason(uint8_t reason);

#endif // ALARMS_H

/**
 * @}
 */
