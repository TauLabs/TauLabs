#####
# Project: Sparky2
#
# Makefile for Sparky2 UAVObject code
#
# Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
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
#####

# These are the UAVObjects supposed to be build as part of the OpenPilot target
# (all architectures)

UAVOBJSRCFILENAMES =
UAVOBJSRCFILENAMES += acceldesired
UAVOBJSRCFILENAMES += inssettings
UAVOBJSRCFILENAMES += gyrosbias
UAVOBJSRCFILENAMES += magnetometer
UAVOBJSRCFILENAMES += magbias
UAVOBJSRCFILENAMES += baroairspeed
UAVOBJSRCFILENAMES += airspeedsettings
UAVOBJSRCFILENAMES += fixedwingpathfollowersettings
UAVOBJSRCFILENAMES += flightplancontrol
UAVOBJSRCFILENAMES += flightplansettings
UAVOBJSRCFILENAMES += flightplanstatus
UAVOBJSRCFILENAMES += flightstats
UAVOBJSRCFILENAMES += flightstatssettings
UAVOBJSRCFILENAMES += geofencesettings
UAVOBJSRCFILENAMES += gpssatellites
UAVOBJSRCFILENAMES += gpstime
UAVOBJSRCFILENAMES += groundpathfollowersettings
UAVOBJSRCFILENAMES += loggingsettings
UAVOBJSRCFILENAMES += loggingstats
UAVOBJSRCFILENAMES += loitercommand
UAVOBJSRCFILENAMES += vtolpathfollowersettings
UAVOBJSRCFILENAMES += vtolpathfollowerstatus
UAVOBJSRCFILENAMES += insstate
UAVOBJSRCFILENAMES += nedaccel
UAVOBJSRCFILENAMES += nedposition
UAVOBJSRCFILENAMES += overosyncstats
UAVOBJSRCFILENAMES += overosyncsettings
UAVOBJSRCFILENAMES += pathplannersettings
UAVOBJSRCFILENAMES += rfm22breceiver
UAVOBJSRCFILENAMES += sonaraltitude
UAVOBJSRCFILENAMES += stateestimation
UAVOBJSRCFILENAMES += rfm22bstatus
UAVOBJSRCFILENAMES += openlrs
UAVOBJSRCFILENAMES += openlrsstatus
UAVOBJSRCFILENAMES += velocitydesired
UAVOBJSRCFILENAMES += hwsparky2
UAVOBJSRCFILENAMES += altitudeholdsettings
UAVOBJSRCFILENAMES += altitudeholdstate
UAVOBJSRCFILENAMES += altitudeholddesired
UAVOBJSRCFILENAMES += waypoint
UAVOBJSRCFILENAMES += waypointactive
UAVOBJSRCFILENAMES += hottsettings
UAVOBJSRCFILENAMES += picocsettings
UAVOBJSRCFILENAMES += picocstatus

UAVOBJSRCFILENAMES += vibrationanalysissettings
UAVOBJSRCFILENAMES += vibrationanalysisoutput

