/**
 ******************************************************************************
 *
 * @file       uavobjects_global.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup UAVTalkPlugin UAVTalk Plugin
 * @{
 * @brief The UAVTalk protocol plugin
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

#ifndef UAVTALKRELAY_GLOBAL_H
#define UAVTALKRELAY_GLOBAL_H

#include <QtCore/qglobal.h>
#include <QObject>
#include <coreplugin/iconfigurableplugin.h>

#if defined(UAVTALKRELAY_LIBRARY)
#  define UAVTALKRELAY_EXPORT Q_DECL_EXPORT
#else
#  define UAVTALKRELAY_EXPORT Q_DECL_IMPORT
#endif

class UAVTALKRELAY_EXPORT UavTalkRelayComon:public QObject
{
    Q_OBJECT
public:
    typedef enum {ReadOnly,WriteOnly,ReadWrite,None} accessType;
};

#endif // UAVTALKRELAY_GLOBAL_H
