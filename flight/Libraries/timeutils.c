/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       timeutils.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @brief      Time conversion functions
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

#include "timeutils.h"

/**
 * Convert UNIX timestamp to date/time
 * Based on: http://stackoverflow.com/questions/21593692/convert-unix-timestamp-to-date-without-system-libs
 * NOTE: May not handle leap-seconds correctly
 */
void date_from_timestamp(uint32_t timestamp, DateTimeT *date_time)
{
	uint32_t minutes, hours;
	uint16_t days, year, dayOfWeek;
	uint8_t month;

	/* calculate minutes */
	minutes  = timestamp / 60;
	timestamp -= minutes * 60;
	/* calculate hours */
	hours    = minutes / 60;
	minutes -= hours   * 60;
	/* calculate days */
	days     = hours   / 24;
	hours   -= days    * 24;

	/* Unix time starts in 1970 on a Thursday */
	year      = 1970;
	dayOfWeek = 4;

	while(1)
	{
		uint8_t leapYear = (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
		uint16_t daysInYear = leapYear ? 366 : 365;
		if (days >= daysInYear)
		{
			dayOfWeek += leapYear ? 2 : 1;
			days -= daysInYear;
			if (dayOfWeek >= 7)
				dayOfWeek -= 7;
			++year;
		}
		else
		{
			dayOfWeek  += days;
			dayOfWeek  %= 7;

			/* calculate the month and day */
			static const uint8_t daysInMonth[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
			for(month = 0; month < 12; ++month)
			{
				uint8_t dim = daysInMonth[month];

				/* add a day to feburary if this is a leap year */
				if (month == 1 && leapYear)
					++dim;

				if (days >= dim)
					days -= dim;
				else
					break;
			}
			break;
		}
	}

	date_time->sec  = timestamp;
	date_time->min  = minutes;
	date_time->hour = hours;
	date_time->mday = days + 1;
	date_time->mon = month;
	date_time->year = year - 1900;
	date_time->wday = dayOfWeek;
}
