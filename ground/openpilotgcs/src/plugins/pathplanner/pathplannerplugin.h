/**
 ******************************************************************************
 * @file       pathplannerplugin.h
 * @author     The PhoenixPilot Team, Copyright (C) 2012.
 * @addtogroup Path Planner GCS Plugins
 * @{
 * @addtogroup PathPlannerGadgetPlugin Path Planner Gadget Plugin
 * @{
 * @brief A gadget to edit a list of waypoints
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

#ifndef PathPlannerPLUGIN_H_
#define PathPlannerPLUGIN_H_

#include <extensionsystem/iplugin.h>
#include "flightdatamodel.h"
#include "modeluavoproxy.h"

class PathPlannerGadgetFactory;

class PathPlannerPlugin : public ExtensionSystem::IPlugin
{
public:
    PathPlannerPlugin();
   ~PathPlannerPlugin();

   void extensionsInitialized();
   bool initialize(const QStringList & arguments, QString * errorString);
   void shutdown();
private:
   PathPlannerGadgetFactory *mf;
   FlightDataModel          *dataModel;
};
#endif /* PathPlannerPLUGIN_H_ */
