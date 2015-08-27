###############################################################################
# @file       Makefile
# @author     The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2011.
# @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2015
# @addtogroup 
# @{
# @addtogroup 
# @{
# @brief Makefile to build UAVObject code for RevoMini board.
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

# These are the UAVObjects supposed to be build as part of the OpenPilot target
# (all architectures)

UAVOBJSRCFILENAMES =
UAVOBJSRCFILENAMES += acceldesired
UAVOBJSRCFILENAMES += geofencesettings
UAVOBJSRCFILENAMES += groundpathfollowersettings
UAVOBJSRCFILENAMES += rfm22breceiver
UAVOBJSRCFILENAMES += hwrevomini
UAVOBJSRCFILENAMES += altitudeholdstate
UAVOBJSRCFILENAMES += hottsettings
UAVOBJSRCFILENAMES += picocsettings
UAVOBJSRCFILENAMES += picocstatus



#Support for radio module on RM
UAVOBJSRCFILENAMES += rfm22bstatus
UAVOBJSRCFILENAMES += openlrs
UAVOBJSRCFILENAMES += openlrsstatus
