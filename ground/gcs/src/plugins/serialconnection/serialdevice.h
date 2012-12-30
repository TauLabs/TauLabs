/**
 ******************************************************************************
 *
 * @file       serialdevice.h
 * @author     The PhoenixPilot Team, Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief The Core GCS plugin
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
#ifndef SERIALDEVICE_H
#define SERIALDEVICE_H

#include <coreplugin/idevice.h>


/**
 * @brief The SerialDevice class
 *
 *  No added features on top of IDevice.
 */

class SerialDevice : public Core::IDevice
{
public:
    SerialDevice();
};

#endif // SERIALDEVICE_H
