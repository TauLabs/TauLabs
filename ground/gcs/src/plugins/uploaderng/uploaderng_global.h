/**
 ******************************************************************************
 *
 * @file       uploaderng_global.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @see        The GNU Public License (GPL) Version 3
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup
 * @{
 * @brief
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

#ifndef UPLOADERNG_GLOBAL_H
#define UPLOADERNG_GLOBAL_H

#include <QtCore/qglobal.h>
namespace uploaderng
{
typedef enum { WAITING_CONNECT, WAITING_DISCONNECT, FAILURE, FAILURE_FILENOTFOUND, LOADING_FW, SUCCESS, DISCONNECTED, BOOTING, HALTING, RESCUING, BL_FROM_HALT, BL_FROM_RESCUE, CONNECTED_TO_TELEMETRY, UPLOADING_FW, UPLOADING_DESC, DOWNLOADING_PARTITION, UPLOADING_PARTITION, DOWNLOADING_PARTITION_BUNDLE, UPLOADING_PARTITION_BUNDLE} UploaderStatus;
}
#if defined(UPLOADERNG_LIBRARY)
#  define UPLOADERNG_EXPORT Q_DECL_EXPORT
#else
#  define UPLOADERNG_EXPORT Q_DECL_IMPORT
#endif

#endif
