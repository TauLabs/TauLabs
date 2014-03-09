/**
 ******************************************************************************
 *
 * @file       port.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Uploader Serial and USB Uploader Plugin
 * @{
 * @brief The USB and Serial protocol uploader plugin
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
#ifndef PORT_H
#define PORT_H
#include <qglobal.h>
#include <QtSerialPort/QSerialPort>
#include <QtSerialPort/QSerialPortInfo>
#include <QTime>
#include <QDebug>
#include "common.h"

class port
{
public:
    enum portstatus{open,closed,error};
    qint16 pfSerialRead(void);                 // function to read a character from the serial input stream
    void pfSerialWrite( quint8 );              // function to write a byte to be sent out the serial port
    quint32 pfGetTime(void);
    quint8		retryCount;						// how many times have we tried to transmit the 'send' packet
    quint8 	maxRetryCount;					// max. times to try to transmit the 'send' packet
    quint16 	max_retry;                             	// Maximum number of retrys for a single transmit.
    qint32 	timeoutLen;						// how long to wait for each retry to succeed
    quint32		timeout;						// current timeout. when 'time' reaches this point we have timed out
    quint8 	txSeqNo; 						// current 'send' packet sequence number
    quint16 	rxBufPos;						//  current buffer position in the receive packet
    quint16	rxBufLen;						// number of 'data' bytes in the buffer
    quint8 	rxSeqNo;						// current 'receive' packet number
    quint16 	rxBufSize;						// size of the receive buffer.
    quint16 	txBufSize;						// size of the transmit buffer.
    quint8		*txBuf;							// transmit buffer. REquired to store a copy of packet data in case a retry is needed.
    quint8		*rxBuf;							// receive buffer. Used to store data as a packet is received.
    quint16    sendSynch;      				// flag to indicate that we should send a synchronize packet to the host
    // this is required when switching from the application to the bootloader
    // and vice-versa. This fixes the firwmare download timeout.
    // when this flag is set to true, the next time we send a packet we will first                                                                                 // send a synchronize packet.
    ReceiveState	InputState;
    decodeState_	DecodeState;
    quint16		SendState;
    quint16		crc;
    quint32		RxError;
    quint32		TxError;
    quint16		flags;
    port(QString name);
    ~port();
    portstatus status();
private:
    portstatus mstatus;
    QTime timer;
    QSerialPort *sport;
};

#endif // PORT_H
