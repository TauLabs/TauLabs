/**
 ******************************************************************************
 * @file       pathfilet.cpp
 * @author     Tau Labs, http://github.com/TauLabs, Copyright (C) 2012-2013.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Path Planner Pluggin
 * @{
 * @brief Abstact algorithm that can be run on a path
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

#include <algorithms/pathfilet.h>

PathFilet::PathFilet(QObject *parent) : IPathAlgorithm(parent)
{
    // TODO: move into the constructor and come from the UI
    radius = 5;
}

/**
 * Verify the path is valid to run through this algorithm
 * @param[in] model the flight model to validate
 * @param[out] err an error message for the user for invalid paths
 * @return true for valid path, false for invalid
 */
bool PathFilet::verifyPath(FlightDataModel *model, QString &err)
{
    Q_UNUSED(model);
    Q_UNUSED(err);

    return true;
}

/**
 * Process the flight path according to the algorithm
 * @param[in] original the flight model to process
 * @param[out] new the resulting flight model
 * @return true for success, false for failure
 */
bool PathFilet::processPath(FlightDataModel *model)
{
    Q_UNUSED(model);

    return true;
}

