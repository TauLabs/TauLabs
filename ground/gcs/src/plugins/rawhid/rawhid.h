/**
 ******************************************************************************
 *
 * @file       rawhid.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup RawHIDPlugin Raw HID Plugin
 * @{
 * @brief Impliments a HID USB connection to the flight hardware as a QIODevice
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

#ifndef RAWHID_H
#define RAWHID_H

#include "rawhid_global.h"

#include <QThread>
#include <QIODevice>
#include <QMutex>
#include <QByteArray>
#include <coreplugin/iconnection.h>

#include "hidapi/hidapi.h"
#include "usbmonitor.h"
#include "usbdevice.h"

//helper classes
class RawHIDReadThread;
class RawHIDWriteThread;

/**
*   The actual IO device that will be used to communicate
*   with the board.
*/
class RAWHID_EXPORT RawHID : public QIODevice
{
	Q_OBJECT

    friend class RawHIDReadThread;
    friend class RawHIDWriteThread;

public:
    RawHID();
    RawHID(USBDevice *deviceName);
    virtual ~RawHID();

    virtual bool open(OpenMode mode);
    virtual void close();
    virtual bool isSequential() const;

protected:
    virtual qint64 readData(char *data, qint64 maxSize);
    virtual qint64 writeData(const char *data, qint64 maxSize);
    virtual qint64 bytesAvailable() const;
    virtual qint64 bytesToWrite() const;

    USBDevice *m_deviceInfo;
    hid_device *m_handle;

    RawHIDReadThread *m_readThread;
    RawHIDWriteThread *m_writeThread;

	QMutex *m_mutex;
};

#endif // RAWHID_H
