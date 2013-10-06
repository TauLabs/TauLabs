/*******************************************************************
 *
 *	NAME: ssp.h
 *
 *
 *******************************************************************/
#ifndef SSP_H
#define SSP_H
/** INCLUDE FILES **/
#include <qglobal.h>

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

typedef enum decodeState_ {
	decode_len1_e = 0,
	decode_seqNo_e,
	decode_data_e,
	decode_crc1_e,
	decode_crc2_e,
	decode_idle_e
} DecodeState_t;

typedef enum ReceiveState {
	state_escaped_e = 0,
	state_unescaped_e
} ReceiveState_t;

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
	void		(*pfCallBack)( quint8 *, quint16);	// call back function that is called when a full packet has been received
	qint16		(*pfSerialRead)(void);					// function to call to read a byte from serial hardware
	void		(*pfSerialWrite)( quint8 );			// function used to write a byte to serial hardware for transmission
	quint32	(*pfGetTime)(void);						// function returns time in number of seconds that has elapsed from a given reference point
}	PortConfig_t;

typedef struct Port_tag {
	void		(*pfCallBack)( quint8 *, quint16);	// call back function that is called when a full packet has been received
	qint16		(*pfSerialRead)(void);			// function to read a character from the serial input stream
	void		(*pfSerialWrite)( quint8 );	// function to write a byte to be sent out the serial port
	quint32	(*pfGetTime)(void);				// function returns time in number of seconds that has elapsed from a given reference point
	quint8		retryCount;						// how many times have we tried to transmit the 'send' packet
	quint8 	maxRetryCount;					// max. times to try to transmit the 'send' packet
	qint32 	timeoutLen;						// how long to wait for each retry to succeed
	qint32		timeout;						// current timeout. when 'time' reaches this point we have timed out
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
											// when this flag is set to true, the next time we send a packet we will first
											// send a synchronize packet.
	ReceiveState_t	InputState;
	DecodeState_t	DecodeState;
	quint16		SendState;
	quint16		crc;
	quint32		RxError;
	quint32		TxError;
	quint16		flags;
} Port_t;



/** Public Data **/

/** PUBLIC FUNCTIONS **/
qint16     ssp_ReceiveProcess( Port_t *thisport );
qint16 	ssp_SendProcess( Port_t *thisport );
quint16    ssp_SendString( Port_t *thisport, char *str );
qint16     ssp_SendData(Port_t *thisport, const quint8 * data,const quint16 length );
void        ssp_Init( Port_t *thisport, const PortConfig_t* const info);
qint16		ssp_ReceiveByte(Port_t *thisport );
quint16 	ssp_Synchronise( Port_t *thisport );


/** EXTERNAL FUNCTIONS **/

#endif
