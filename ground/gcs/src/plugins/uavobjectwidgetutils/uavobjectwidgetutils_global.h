/**
 ******************************************************************************
 *
 * @file       uavobjectwidgetutils_global.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVObjectWidgetUtils Plugin
 * @{
 * @brief Utility plugin for UAVObject to Widget relation management
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

#ifndef UAVOBJECTWIDGETUTILS_GLOBAL_H
#define UAVOBJECTWIDGETUTILS_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(UAVOBJECTWIDGETUTILS_LIBRARY)
#  define UAVOBJECTWIDGETUTILS_EXPORT Q_DECL_EXPORT
#else
#  define UAVOBJECTWIDGETUTILS_EXPORT Q_DECL_IMPORT
#endif

#endif
