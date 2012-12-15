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
#include <stdint.h>
#include <qextserialport/src/qextserialport.h>
#include <QTime>
#include <QDebug>
#include "common.h"

class port
{
public:
    enum portstatus{open,closed,error};
    virtual int16_t pfSerialRead(void);			// function to read a character from the serial input stream
    virtual void pfSerialWrite( uint8_t );	// function to write a byte to be sent out the serial port
    virtual uint32_t pfGetTime(void);
    uint8_t		retryCount;						// how many times have we tried to transmit the 'send' packet
    uint8_t 	maxRetryCount;					// max. times to try to transmit the 'send' packet
    uint16_t 	max_retry;                             	// Maximum number of retrys for a single transmit.
    int32_t 	timeoutLen;						// how long to wait for each retry to succeed
    uint32_t		timeout;						// current timeout. when 'time' reaches this point we have timed out
    uint8_t 	txSeqNo; 						// current 'send' packet sequence number
    uint16_t 	rxBufPos;						//  current buffer position in the receive packet
    uint16_t	rxBufLen;						// number of 'data' bytes in the buffer
    uint8_t 	rxSeqNo;						// current 'receive' packet number
    uint16_t 	rxBufSize;						// size of the receive buffer.
    uint16_t 	txBufSize;						// size of the transmit buffer.
    uint8_t		*txBuf;							// transmit buffer. REquired to store a copy of packet data in case a retry is needed.
    uint8_t		*rxBuf;							// receive buffer. Used to store data as a packet is received.
    uint16_t    sendSynch;      				// flag to indicate that we should send a synchronize packet to the host
    // this is required when switching from the application to the bootloader
    // and vice-versa. This fixes the firwmare download timeout.
    // when this flag is set to true, the next time we send a packet we will first                                                                                 // send a synchronize packet.
    ReceiveState	InputState;
    decodeState_	DecodeState;
    uint16_t		SendState;
    uint16_t		crc;
    uint32_t		RxError;
    uint32_t		TxError;
    uint16_t		flags;
    port(PortSettings settings,QString name);
    ~port();
    portstatus status();
private:
    portstatus mstatus;
    QTime timer;
    QextSerialPort *sport;
};

#endif // PORT_H
