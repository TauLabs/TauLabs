###############################################################################
# @file       UAVObjects.inc
# @author     The OpenPilot Team, http://www.openpilot.org, Copyright (C) 2011.
# @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
# @addtogroup 
# @{
# @addtogroup 
# @{
# @brief Makefile includes for UAVObject source files
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
UAVOBJSRCFILENAMES += accels
UAVOBJSRCFILENAMES += firmwareiapobj
UAVOBJSRCFILENAMES += flighttelemetrystats
UAVOBJSRCFILENAMES += gcstelemetrystats
UAVOBJSRCFILENAMES += i2cstats
UAVOBJSRCFILENAMES += nedaccel
UAVOBJSRCFILENAMES += objectpersistence
UAVOBJSRCFILENAMES += overosyncstats
UAVOBJSRCFILENAMES += systemalarms
UAVOBJSRCFILENAMES += systemsettings
UAVOBJSRCFILENAMES += systemstats
UAVOBJSRCFILENAMES += taskinfo
UAVOBJSRCFILENAMES += watchdogstatus
UAVOBJSRCFILENAMES += flightstatus
UAVOBJSRCFILENAMES += hwsettings
UAVOBJSRCFILENAMES += receiveractivity
UAVOBJSRCFILENAMES += gcsreceiver
UAVOBJSRCFILENAMES += faultsettings

UAVOBJSRC = $(foreach UAVOBJSRCFILE,$(UAVOBJSRCFILENAMES),$(OPUAVSYNTHDIR)/$(UAVOBJSRCFILE).c )
UAVOBJDEFINE = $(foreach UAVOBJSRCFILE,$(UAVOBJSRCFILENAMES),-DUAVOBJ_INIT_$(UAVOBJSRCFILE) )
