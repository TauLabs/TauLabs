###############################################################################
# @file       Makefile
# @author     The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2012.
# @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
# @addtogroup 
# @{
# @addtogroup 
# @{
# @brief Makefile to build UAVObject code for quanton flight control.
###############################################################################
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#

# These are the UAVObjects supposed to be build as part of the OpenPilot target
# (all architectures)

UAVOBJSRCFILENAMES =
UAVOBJSRCFILENAMES += acceldesired
UAVOBJSRCFILENAMES += accessorydesired
UAVOBJSRCFILENAMES += actuatorcommand
UAVOBJSRCFILENAMES += actuatordesired
UAVOBJSRCFILENAMES += actuatorsettings
UAVOBJSRCFILENAMES += attitudesettings
UAVOBJSRCFILENAMES += inssettings
UAVOBJSRCFILENAMES += attitudeactual
UAVOBJSRCFILENAMES += gyros
UAVOBJSRCFILENAMES += gyrosbias
UAVOBJSRCFILENAMES += sensorsettings
UAVOBJSRCFILENAMES += accels
UAVOBJSRCFILENAMES += magnetometer
UAVOBJSRCFILENAMES += magbias
UAVOBJSRCFILENAMES += baroaltitude
UAVOBJSRCFILENAMES += baroairspeed
UAVOBJSRCFILENAMES += airspeedsettings
UAVOBJSRCFILENAMES += airspeedactual
UAVOBJSRCFILENAMES += fixedwingairspeeds
UAVOBJSRCFILENAMES += fixedwingpathfollowersettings
UAVOBJSRCFILENAMES += fixedwingpathfollowerstatus
UAVOBJSRCFILENAMES += flightbatterysettings
UAVOBJSRCFILENAMES += firmwareiapobj
UAVOBJSRCFILENAMES += flightbatterystate
UAVOBJSRCFILENAMES += flightplancontrol
UAVOBJSRCFILENAMES += flightplansettings
UAVOBJSRCFILENAMES += flightplanstatus
UAVOBJSRCFILENAMES += flighttelemetrystats
UAVOBJSRCFILENAMES += gcstelemetrystats
UAVOBJSRCFILENAMES += gcsreceiver
UAVOBJSRCFILENAMES += geofencesettings
UAVOBJSRCFILENAMES += gpsposition
UAVOBJSRCFILENAMES += gpssatellites
UAVOBJSRCFILENAMES += gpstime
UAVOBJSRCFILENAMES += gpsvelocity
UAVOBJSRCFILENAMES += groundpathfollowersettings
UAVOBJSRCFILENAMES += loggingsettings
UAVOBJSRCFILENAMES += loggingstats
UAVOBJSRCFILENAMES += loitercommand
UAVOBJSRCFILENAMES += vtolpathfollowersettings
UAVOBJSRCFILENAMES += vtolpathfollowerstatus
UAVOBJSRCFILENAMES += homelocation
UAVOBJSRCFILENAMES += insstate
UAVOBJSRCFILENAMES += manualcontrolcommand
UAVOBJSRCFILENAMES += manualcontrolsettings
UAVOBJSRCFILENAMES += mixersettings
UAVOBJSRCFILENAMES += mixerstatus
UAVOBJSRCFILENAMES += mwratesettings
UAVOBJSRCFILENAMES += nedaccel
UAVOBJSRCFILENAMES += nedposition
UAVOBJSRCFILENAMES += objectpersistence
UAVOBJSRCFILENAMES += overosyncstats
UAVOBJSRCFILENAMES += overosyncsettings
UAVOBJSRCFILENAMES += pathdesired
UAVOBJSRCFILENAMES += pathplannersettings
UAVOBJSRCFILENAMES += pathstatus
UAVOBJSRCFILENAMES += poilocation
UAVOBJSRCFILENAMES += positionactual
UAVOBJSRCFILENAMES += ratedesired
UAVOBJSRCFILENAMES += sonaraltitude
UAVOBJSRCFILENAMES += stabilizationdesired
UAVOBJSRCFILENAMES += stabilizationsettings
UAVOBJSRCFILENAMES += stateestimation
UAVOBJSRCFILENAMES += systemalarms
UAVOBJSRCFILENAMES += systemident
UAVOBJSRCFILENAMES += systemsettings
UAVOBJSRCFILENAMES += systemstats
UAVOBJSRCFILENAMES += tabletinfo
UAVOBJSRCFILENAMES += taskinfo
UAVOBJSRCFILENAMES += velocityactual
UAVOBJSRCFILENAMES += velocitydesired
UAVOBJSRCFILENAMES += watchdogstatus
UAVOBJSRCFILENAMES += flightstatus
UAVOBJSRCFILENAMES += hwquanton
UAVOBJSRCFILENAMES += modulesettings
UAVOBJSRCFILENAMES += receiveractivity
UAVOBJSRCFILENAMES += cameradesired
UAVOBJSRCFILENAMES += camerastabsettings
UAVOBJSRCFILENAMES += altitudeholdsettings
UAVOBJSRCFILENAMES += altitudeholddesired
UAVOBJSRCFILENAMES += altitudeholdstate
UAVOBJSRCFILENAMES += waypoint
UAVOBJSRCFILENAMES += waypointactive
UAVOBJSRCFILENAMES += ubloxinfo
UAVOBJSRCFILENAMES += hottsettings
UAVOBJSRCFILENAMES += picocsettings
UAVOBJSRCFILENAMES += picocstatus

UAVOBJSRCFILENAMES += txpidsettings

UAVOBJSRCFILENAMES += vibrationanalysissettings
UAVOBJSRCFILENAMES += vibrationanalysisoutput
UAVOBJSRCFILENAMES += trimangles
UAVOBJSRCFILENAMES += trimanglessettings

UAVOBJSRCFILENAMES += sessionmanaging

UAVOBJSRC = $(foreach UAVOBJSRCFILE,$(UAVOBJSRCFILENAMES),$(OPUAVSYNTHDIR)/$(UAVOBJSRCFILE).c )
UAVOBJDEFINE = $(foreach UAVOBJSRCFILE,$(UAVOBJSRCFILENAMES),-DUAVOBJ_INIT_$(UAVOBJSRCFILE) )
