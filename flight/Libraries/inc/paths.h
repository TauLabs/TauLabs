/**
 ******************************************************************************
 * @addtogroup TauLabsLibraries Tau Labs Libraries
 * @{
 *
 * @file       paths.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @brief      Path calculation library with common API
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

#ifndef PATHS_H_
#define PATHS_H_

#include "pios.h"
#include "openpilot.h"
#include "pathdesired.h"

struct path_status {
	float fractional_progress;
	float error;
	float correction_direction[2];
	float path_direction[2];
};

void path_progress(const PathDesiredData *pathDesired, const float * cur_point, struct path_status * status);

#endif /* PATHS_H_ */

/**
 * @}
 */