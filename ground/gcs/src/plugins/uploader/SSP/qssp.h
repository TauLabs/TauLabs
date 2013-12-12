/**
 ******************************************************************************
 *
 * @file       qssp.h
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
#ifndef QSSP_H
#define QSSP_H
#include <qglobal.h>
#include "port.h"
#include "common.h"
/** LOCAL DEFINITIONS **/
#ifndef TRUE
#define TRUE	1
#endif

#ifndef FALSE
#define FALSE	0
#endif

#define SSP_TX_IDLE       	0   // not expecting a ACK packet (no current transmissions in progress)
#define SSP_TX_WAITING    	1   // waiting for a valid ACK to arrive
#define SSP_TX_TIMEOUT    	2   // failed to receive a valid ACK in the timeout period, after retrying.
#define SSP_TX_ACKED      	3   // valid ACK received before timeout period.
#define SSP_TX_BUFOVERRUN 	4   // amount of data to send execeds the transmission buffer sizeof
#define SSP_TX_BUSY       	5   // Attempted to start a transmission while a transmission was already in  progress.
//#define SSP_TX_FAIL    - failure...

#define	SSP_RX_IDLE			0
#define SSP_RX_RECEIVING 	1
#define SSP_RX_COMPLETE		2

// types of packet that can be received
#define SSP_RX_DATA       	5
#define SSP_RX_ACK        	6
#define SSP_RX_SYNCH      	7


typedef struct
{
    quint8 	*pbuff;
    quint16 	length;
    quint16 	crc;
    quint8 	seqNo;
} Packet_t;

typedef struct {

    quint8 	*rxBuf;                             	// Buffer used to store rcv data
    quint16 	rxBufSize;                         	// rcv buffer size.
    quint8 	*txBuf;                            	// Length of data in buffer
    quint16 	txBufSize;                        	// CRC for data in Packet buff
    quint16 	max_retry;                             	// Maximum number of retrys for a single transmit.
    qint32 	timeoutLen;                          	//  how long to wait for each retry to succeed
    // function returns time in number of seconds that has elapsed from a given reference point
}	PortConfig_t;





/** Public Data **/




/** EXTERNAL FUNCTIONS **/

class qssp
{
private:
    port * thisport;
    decodeState_ DecodeState_t;
    /** PRIVATE FUNCTIONS **/
    //static void   	sf_SendSynchPacket( Port_t *thisport );
    quint16 sf_crc16( quint16 crc, quint8 data );
    void   	sf_write_byte(quint8 c );
    void   	sf_SetSendTimeout();
    quint16 sf_CheckTimeout();
    qint16 	sf_DecodeState(quint8 c );
    qint16 	sf_ReceiveState(quint8 c );

    void   	sf_SendPacket();
    void   	sf_SendAckPacket(quint8 seqNumber);
    void     sf_MakePacket( quint8 *buf, const quint8 * pdata, quint16 length, quint8 seqNo );
    qint16 	sf_ReceivePacket();
    quint16 ssp_SendDataBlock(quint8 *data, quint16 length );
    bool debug;
public:
    /** PUBLIC FUNCTIONS **/
     virtual void pfCallBack( quint8 *, quint16);	// call back function that is called when a full packet has been received
    qint16     ssp_ReceiveProcess();
    qint16 	ssp_SendProcess();
    quint16    ssp_SendString(char *str );
    qint16     ssp_SendData( const quint8 * data,const quint16 length );
    void        ssp_Init( const PortConfig_t* const info);
    qint16		ssp_ReceiveByte( );
    quint16 	ssp_Synchronise(  );
    qssp(port * info,bool debug);
};

#endif // QSSP_H
