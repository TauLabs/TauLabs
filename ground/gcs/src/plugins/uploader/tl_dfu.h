/**
 ******************************************************************************
 *
 * @file       tl_dfu.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Uploader Uploader Plugin
 * @{
 * @brief Low level bootloader protocol functions
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

#ifndef TL_DFU_H
#define TL_DFU_H

#include <rawhid/hidapi/hidapi.h>
#include <rawhid/usbsignalfilter.h>
#include <QDebug>
#include <QFile>
#include <QThread>
#include <QTimer>
#include "bl_messages.h"

using namespace std;
#define BUF_LEN 64
#define BL_CAP_EXTENSION_MAGIC 0x3456

namespace tl_dfu {

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
    abort,//13
    not_in_dfu
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
    int HW_Rev;
    bool CapExt;
};

class DFUObject : public QThread
{
    Q_OBJECT

    typedef struct messagePackets
    {
        quint32 numberOfPackets;
        quint8 lastPacketCount;
        int pad;
    } messagePackets;

public:
    static quint32 CRCFromQBArray(QByteArray array, quint32 Size);
    DFUObject();
    ~DFUObject();

    // Service commands:
    int JumpToApp(bool);
    int ResetDevice(void);
    bool OpenBootloaderComs(USBPortInfo port);
    void CloseBootloaderComs();

    // Partition operations:
    bool UploadPartitionThreaded(QByteArray &sourceArray, dfu_partition_label partition, int size);
    bool DownloadPartitionThreaded(QByteArray *firmwareArray, dfu_partition_label partition, int size);
    bool WipePartition(dfu_partition_label partition);
    QByteArray DownloadDescriptionAsByteArray(int const & numberOfChars);

public slots:
    device findCapabilities();
    QString partitionStringFromLabel(dfu_partition_label label);

signals:
    void downloadFinished(bool);
    void uploadFinished(tl_dfu::Status);
    void operationProgress(QString status, int progress);

private:
    bool DownloadPartition(QByteArray *fw, qint32 const & numberOfBytes, const dfu_partition_label &partition);
    tl_dfu::Status UploadPartition(QByteArray &sfile, dfu_partition_label partition);

    // Helper functions:
    QString StatusToString(tl_dfu::Status  const & status);
    static quint32 CRC32WideFast(quint32 Crc, quint32 Size, quint32 *Buffer);
    void CopyWords(char * source, char* destination, int count);
    messagePackets CalculatePadding(quint32 numberOfBytes);

    // Service commands:
    bool EnterDFU();
    tl_dfu::Status StatusRequest();
    bool EndOperation();
    int AbortOperation(void);

    // USB coms:
    int SendData(bl_messages);
    int ReceiveData(bl_messages &data);
    hid_device *m_hidHandle;

    bool StartUpload(qint32  const &numberOfBytes, const dfu_partition_label &label, quint32 crc);
    bool UploadData(qint32 const & numberOfPackets,QByteArray  & data);

    typedef struct ThreadJobStruc
    {
        enum Actions
        {
            Download,
            Upload
        };
        qint32 requestSize;
        dfu_partition_label requestTransferType;
        QByteArray *requestStorage;
        quint32 partition_size;
        Actions requestedOperation;
    } ThreadJobStruc;
    ThreadJobStruc threadJob;

    bool open;
protected:
    void run();// Executes the upload or download operations
};
}

Q_DECLARE_METATYPE(tl_dfu::Status)

#endif // TL_DFU_H
