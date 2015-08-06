/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       timeutils.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 *             Free Software Foundation, Inc. (C) 1991-2015
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

/* Test for leap year. Nonzero if YEAR is a leap year (every 4 years,
 except every 100th isn't, and every 400th is). */
# define __isleap(year) \
 ((year) % 4 == 0 && ((year) % 100 != 0 || (year) % 400 == 0))

/* How many days come before each month (0-12).  */
static const uint16_t __mon_yday[2][13] = {
	/* Normal years.  */
	{0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365},
	/* Leap years.  */
	{0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366}
};

#define	SECS_PER_HOUR	(60 * 60)
#define	SECS_PER_DAY	(SECS_PER_HOUR * 24)

/**
 * Convert UNIX timestamp to date/time
 * Based on code from GNU C Library
 */
void date_from_timestamp(uint32_t timestamp, DateTimeT *date_time)
{
	uint16_t y;
	int16_t days;
	uint32_t rem;

	days = timestamp / SECS_PER_DAY;
	rem = timestamp % SECS_PER_DAY;
	while (rem >= SECS_PER_DAY)
	{
		rem -= SECS_PER_DAY;
		++days;
	}
	date_time->hour = rem / SECS_PER_HOUR;
	rem %= SECS_PER_HOUR;
	date_time->min = rem / 60;
	date_time->sec = rem % 60;
	/* January 1, 1970 was a Thursday.  */
	date_time->wday = (4 + days) % 7;
	if (date_time->wday < 0){
		date_time->wday += 7;
	}
	y = 1970;

#define DIV(a, b) ((a) / (b) - ((a) % (b) < 0))
#define LEAPS_THRU_END_OF(y) (DIV (y, 4) - DIV (y, 100) + DIV (y, 400))

	while (days < 0 || days >= (__isleap (y) ? 366 : 365))
	{
		/* Guess a corrected year, assuming 365 days per year.  */
		uint16_t yg = y + days / 365 - (days % 365 < 0);

		/* Adjust DAYS and Y to match the guessed year.  */
		days -= ((yg - y) * 365
				 + LEAPS_THRU_END_OF (yg - 1)
				 - LEAPS_THRU_END_OF (y - 1));
		y = yg;
	}
	date_time->year = y - 1900;
	if (date_time->year != y - 1900)
	{
		/* The year cannot be represented due to overflow.  */
		return;
	}
	const uint16_t *ip = __mon_yday[__isleap(y)];
	for (y = 11; days < (uint16_t)ip[y]; --y){
		continue;
	}
	days -= ip[y];
	date_time->mon = y;
	date_time->mday = days + 1;
}
