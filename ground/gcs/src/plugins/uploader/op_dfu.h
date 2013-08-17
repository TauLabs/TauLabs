/**
 ******************************************************************************
 *
 * @file       op_dfu.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Uploader Uploader Plugin
 * @{
 * @brief The uploader plugin
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

#ifndef OP_DFU_H
#define OP_DFU_H

#include <QByteArray>
#include <rawhid/pjrc_rawhid.h>
#include <rawhid/usbmonitor.h>
#include <rawhid/usbsignalfilter.h>
#include <QDebug>
#include <QFile>
#include <QThread>
#include <QMutex>
#include <QMutexLocker>
#include <QMetaType>
#include <QCryptographicHash>
#include <QList>
#include <QVariant>
#include <iostream>
#include <QtSerialPort/QSerialPort>
#include <QtSerialPort/QSerialPortInfo>
#include <QTime>
#include <QTimer>
#include <coreplugin/icore.h>
#include <coreplugin/boardmanager.h>
#include "SSP/qssp.h"
#include "SSP/port.h"
#include "SSP/qsspt.h"

using namespace std;
#define BUF_LEN 64
#define BL_CAP_EXTENSION_MAGIC 0x3456
#define MAX_PACKET_DATA_LEN	255
#define MAX_PACKET_BUF_SIZE	(1+1+MAX_PACKET_DATA_LEN+2)

namespace OP_DFU {

    enum TransferTypes
    {
        FW,
        Descript
    };

    enum CompareType
    {
        crccompare,
        bytetobytecompare
    };

    enum Status
    {
        DFUidle,//0
        uploading,//1
        wrong_packet_received,//2
        too_many_packets,//3
        too_few_packets,//4
        Last_operation_Success,//5
        downloading,//6
        idle,//7
        Last_operation_failed,//8
        uploadingStarting,//9
        outsideDevCapabilities,//10
        CRC_Fail,//11
        failed_jump,//12
        abort//13
    };

    enum Actions
    {
        actionProgram,
        actionProgramAndVerify,
        actionDownload,
        actionCompareAll,
        actionCompareCrc,
        actionListDevs,
        actionStatusReq,
        actionReset,
        actionJump
    };

    enum Commands
    {
        Reserved,//0
        Req_Capabilities,//1
        Rep_Capabilities,//2
        EnterDFU,//3
        JumpFW,//4
        Reset,//5
        Abort_Operation,//6
        Upload,//7
        Op_END,//8
        Download_Req,//9
        Download,//10
        Status_Request,//11
        Status_Rep,//12
        Wipe_Partition//13
    };

    enum eBoardType
    {
        eBoardUnkwn = 0,
        eBoardMainbrd = 1,
        eBoardINS,
        eBoardPip = 3,
        eBoardCC = 4,
        eBoardRevo = 9,
    };

    struct device
    {
            quint16 ID;
            quint32 FW_CRC;
            quint8 BL_Version;
            int SizeOfDesc;
            quint32 SizeOfCode;
            bool Readable;
            bool Writable;
            QVector<quint32> PartitionSizes;
    };


    class DFUObject : public QThread
    {
        Q_OBJECT;

        public:
        static quint32 CRCFromQBArray(QByteArray array, quint32 Size);
        //DFUObject(bool debug);
        DFUObject(bool debug,bool use_serial,QString port);

        ~DFUObject();

        // Service commands:
        bool enterDFU(int const &devNumber);
        bool findDevices();
        int JumpToApp(bool);
        int ResetDevice(void);
        OP_DFU::Status StatusRequest();
        bool EndOperation();
        int AbortOperation(void);
        bool ready() { return mready; }

        // Upload (send to device) commands
        OP_DFU::Status UploadDescription(QVariant description);
        bool UploadPartition(const QString &sfile, const bool &verify, int device, int partition, int size);

        // Download (get from device) commands:
        // DownloadDescription is synchronous
        QString DownloadDescription(int const & numberOfChars);
        QByteArray DownloadDescriptionAsBA(int const & numberOfChars);
        // Asynchronous firmware download: initiates fw download,
        // and a downloadFinished signal is emitted when download
        // if finished:
        bool DownloadFirmware(QByteArray *byteArray, int device);

        // Comparison functions (is this needed?)
        OP_DFU::Status CompareFirmware(const QString &sfile, const CompareType &type,int device);

        bool SaveByteArrayToFile(QString const & file,QByteArray const &array);



        // Variables:
        QList<device> devices;
        int numberOfDevices;
        int send_delay;
        bool use_delay;

        // Helper functions:
        QString StatusToString(OP_DFU::Status  const & status);
        static quint32 CRC32WideFast(quint32 Crc, quint32 Size, quint32 *Buffer);
        OP_DFU::eBoardType GetBoardType(int boardNum);

        bool DownloadPartition(QByteArray *firmwareArray, int device, int partition,int size);
        bool WipePartition(int partition);

    signals:
       void progressUpdated(int);
       void downloadFinished(bool);
       void uploadFinished(OP_DFU::Status);
       void operationProgress(QString status);

    private:
       // Generic variables:
        bool debug;
        bool use_serial;
        bool mready;
        int RWFlags;
        qsspt * serialhandle;
        int sendData(void*,int);
        int receiveData(void * data,int size);
        quint8	sspTxBuf[MAX_PACKET_BUF_SIZE];
        quint8	sspRxBuf[MAX_PACKET_BUF_SIZE];
        port * info;


        // USB Bootloader:
        pjrc_rawhid hidHandle;
        int setStartBit(int command){ return command|0x20; }

        void CopyWords(char * source, char* destination, int count);
        void printProgBar( int const & percent,QString const& label);
        bool StartUpload(qint32  const &numberOfBytes, const int &type, quint32 crc);
        bool UploadData(qint32 const & numberOfPackets,QByteArray  & data);

        // Thread management:
        // Same as startDownload except that we store in an external array:
        bool StartDownloadT(QByteArray *fw, qint32 const & numberOfBytes, const int &type);
        OP_DFU::Status UploadPartitionT(const QString &sfile, const bool &verify,int device,int partition);
        QMutex mutex;
        OP_DFU::Commands requestedOperation;
        qint32 requestSize;
        int requestTransferType;
        QByteArray *requestStorage;
        QString requestFilename;
        bool requestVerify;
        int requestDevice;
        int partition_size;

    protected:
       void run();// Executes the upload or download operations

    };

}

Q_DECLARE_METATYPE(OP_DFU::Status)


#endif // OP_DFU_H
