/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       timeutils.h
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

#ifndef _TIMEUTILS_H
#define _TIMEUTILS_H

#include <stdint.h>

typedef struct {
	uint8_t sec;  // seconds after the minute - [0, 60]
	uint8_t min;  // minutes after the hour - [0, 59]
	uint8_t hour; // hours since midnight - [0, 23]
	uint8_t mday; // day of the month - [1, 31]
	uint8_t mon;  // months since January - [0, 11]
	uint8_t wday; // days since Sunday - [0, 6]
	uint8_t year; // years since 1900
} __attribute__((packed)) DateTimeT;

void date_from_timestamp(uint32_t timestamp, DateTimeT *date_time);


#endif /* _TIMEUTILS_H */
