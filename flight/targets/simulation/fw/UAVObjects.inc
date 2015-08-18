#####
# Project: OpenPilot
#
# Makefile for OpenPilot UAVObject code
#
# The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2011.
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
UAVOBJSRCFILENAMES += geofencesettings
UAVOBJSRCFILENAMES += groundpathfollowersettings
UAVOBJSRCFILENAMES += overosyncstats
UAVOBJSRCFILENAMES += overosyncsettings
UAVOBJSRCFILENAMES += altitudeholdstate
UAVOBJSRCFILENAMES += hottsettings
UAVOBJSRCFILENAMES += picocsettings
UAVOBJSRCFILENAMES += picocstatus


# Include this in the simulator so that we can edit a representative inherited
# obj for dev test.
UAVOBJSRCFILENAMES += hwsparky

UAVOBJSRCFILENAMES += attitudesimulated
