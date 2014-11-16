/**
 ******************************************************************************
 *
 * @file       tl_dfu.cpp
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

#include "tl_dfu.h"

#include <QApplication>

#define TL_DFU_DEBUG
#ifdef TL_DFU_DEBUG
#define TL_DFU_QXTLOG_DEBUG(...) qDebug()<<__VA_ARGS__
#else  // TL_DFU_DEBUG
#define TL_DFU_QXTLOG_DEBUG(...)
#endif	// TL_DFU_DEBUG

using namespace tl_dfu;

DFUObject::DFUObject() : open(false)
{
    qRegisterMetaType<tl_dfu::Status>("TL_DFU::Status");
}

DFUObject::~DFUObject()
{
    if (open)
        hid_close(m_hidHandle);
}

/**
  Tells the mainboard to enter DFU Mode.
  */
bool DFUObject::EnterDFU()
{
    bl_messages message;
    message.flags_command = BL_MSG_ENTER_DFU;
    message.v.enter_dfu.device_number = 0;
    int result = SendData(message);
    if(result < 1)
        return false;
    TL_DFU_QXTLOG_DEBUG(QString("EnterDFU:%0 bytes sent").arg(result));
    return true;
}

/**
  Tells the board to get ready for an upload. It will in particular
  erase the memory to make room for the data. You will have to query
  its status to wait until erase is done before doing the actual upload.
  @param numberOfByte number of bytes of the transfer
  @param label partition where the data will be uploaded to
  @param crc crc value of the data to be uploaded
  @returns result of the requested operation
  */
bool DFUObject::StartUpload(qint32 const & numberOfBytes, dfu_partition_label const & label,quint32 crc)
{
    messagePackets msg = CalculatePadding(numberOfBytes);
    bl_messages message;
    message.flags_command = BL_MSG_WRITE_START;
    message.v.xfer_start.expected_crc = ntohl(crc);
    message.v.xfer_start.packets_in_transfer = ntohl(msg.numberOfPackets);
    message.v.xfer_start.words_in_last_packet = msg.lastPacketCount;
    message.v.xfer_start.label = label;

    TL_DFU_QXTLOG_DEBUG(QString("Number of packets:%0 Size of last packet:%1").arg(msg.numberOfPackets).arg(msg.lastPacketCount));

    int result = SendData(message);
    QEventLoop m_eventloop;
    QTimer::singleShot(500,&m_eventloop, SLOT(quit()));
    m_eventloop.exec();
    TL_DFU_QXTLOG_DEBUG(QString("%0 bytes sent").arg(result));
    if(result > 0)
        return true;
    return false;
}


/**
  Does the actual data upload to the board. Needs to be called once the
  board is ready to accept data following a StartUpload command, and it is erased.
  @param numberOfBytes number of bytes to transfer
  @param data data to transfer
  @returns result of the requested operation
  */
bool DFUObject::UploadData(qint32 const & numberOfBytes, QByteArray  & data)
{
    messagePackets msg = CalculatePadding(numberOfBytes);
    TL_DFU_DEBUG(QString("Start Uploading:%0 4byte packets").arg(msg.numberOfPackets));
    bl_messages message;
    message.flags_command = BL_MSG_WRITE_CONT;
    int packetsize;
    float percentage;
    int laspercentage = 0;
    for(quint32 packetcount = 0; packetcount < msg.numberOfPackets; ++packetcount)
    {
        percentage = (float)(packetcount + 1) / msg.numberOfPackets * 100;
        if(laspercentage != (int)percentage)
            emit operationProgress("", percentage);
        laspercentage=(int)percentage;
        if(packetcount == msg.numberOfPackets)
            packetsize = msg.lastPacketCount;
        else
            packetsize = 14;
        message.v.xfer_cont.current_packet_number = ntohl(packetcount);
        char *pointer = data.data();
        pointer = pointer + 4 * 14 * packetcount;
        CopyWords(pointer, (char*)message.v.xfer_cont.data, packetsize *4);
        int result = SendData(message);
        if(result < 1)
            return false;
    }
    return true;
}

/**
  Downloads the description string for the current device.
  You have to call enterDFU before calling this function.
*/
QByteArray DFUObject::DownloadDescriptionAsByteArray(int const & numberOfChars)
{
    QByteArray array;
    DownloadPartition(&array, numberOfChars, DFU_PARTITION_DESC);
    return array;
}

/**
  Starts a partition download
  @param firmwareArray: pointer to the location where we should store the firmware
  @param partition: the partition to download
  @param size: the number of bytes to transfer
  */
bool DFUObject::DownloadPartitionThreaded(QByteArray *firmwareArray, dfu_partition_label partition, int size)
{
    qDebug()<<"DOWNLOAD PARTITION- SETTING DOWNLOAD CONFIG PARTITION="<<partition<<" SIZE:"<<size;
    if (isRunning())
        return false;
    threadJob.requestedOperation = ThreadJobStruc::Download;
    threadJob.requestSize = size;
    threadJob.requestTransferType = partition;
    threadJob.requestStorage = firmwareArray;
    start();
    return true;
}

/**
  Wipes a partition
  @param partition number of the partition to wipe
  */
bool DFUObject::WipePartition(dfu_partition_label partition)
{
    bl_messages message;
    message.flags_command = BL_MSG_WIPE_PARTITION;
    message.v.wipe_partition.label = partition;
    int result = SendData(message);
    return (result > 0);
}

/**
   Runs the upload or download operations.
  */
void DFUObject::run()
{
    bool downloadResult;
    tl_dfu::Status uploadStatus;
    switch (threadJob.requestedOperation) {
    case ThreadJobStruc::Download:
        qDebug()<<"DOWNLOAD THREAD STARTED";
        downloadResult = DownloadPartition(threadJob.requestStorage, threadJob.requestSize, threadJob.requestTransferType);
        qDebug()<<"DOWNLOAD FINISHED";
        emit downloadFinished(downloadResult);
        break;
    case ThreadJobStruc::Upload:
        uploadStatus = UploadPartition(*threadJob.requestStorage,threadJob.requestTransferType);
        qDebug()<<"UPLOAD FINISHED";
        emit uploadFinished(uploadStatus);
        break;
    default:
        break;
    }
    return;
}

/**
  Synchronously downloads a partition from the board
  @param fw array to store the downloaded partition
  @param numberOfByte number of bytes to download
  @param partition partition from where to download the data
  @returns result of the requested operation
  */
bool DFUObject::DownloadPartition(QByteArray *fw, qint32 const & numberOfBytes, dfu_partition_label const & partition)
{
    EnterDFU();
    emit operationProgress(QString("%0 partition download").arg(partitionStringFromLabel(partition)), -1);
    messagePackets msg = CalculatePadding(numberOfBytes);
    bl_messages message;
    message.flags_command = BL_MSG_READ_START;
    message.v.xfer_start.packets_in_transfer = ntohl(msg.numberOfPackets);
    message.v.xfer_start.words_in_last_packet = msg.lastPacketCount;
    message.v.xfer_start.label = partition;

    int result = SendData(message);

    TL_DFU_QXTLOG_DEBUG(QString("StartDownload %0 Last Packet Size %1 Bytes sent %2").arg(msg.numberOfPackets).arg(msg.lastPacketCount).arg(result));
    float percentage;
    int laspercentage = 0;

    // Now get those packets:
    for(quint32 x = 0;x < msg.numberOfPackets; ++x)
    {
        int size;
        percentage = (float)(x + 1) / msg.numberOfPackets * 100;
        if(laspercentage != (int)percentage)
            operationProgress("",(int)percentage);
        laspercentage=(int)percentage;

        result = ReceiveData(message);

        if(message.flags_command != BL_MSG_READ_CONT)
        {
            TL_DFU_QXTLOG_DEBUG("Message different from BL_MSG_READ_CONT received while downloading partition");
            return false;
        }
        if(ntohl(message.v.xfer_cont.current_packet_number) != x)
        {
            TL_DFU_QXTLOG_DEBUG(QString("Wrong packet number received while downloading partition %0").arg(ntohl(message.v.xfer_cont.current_packet_number)));
            return false;
        }
        if(x == msg.numberOfPackets - 1)
            size = msg.lastPacketCount * 4;
        else
            size = 14 * 4;
        fw->append((char*)message.v.xfer_cont.data, size);
    }
    qDebug()<<"STATUS="<<StatusRequest();
    return true;
}


/**
  Resets the device
  */
int DFUObject::ResetDevice(void)
{
    bl_messages message;
    message.flags_command = BL_MSG_RESET;
    return SendData(message);
}

/**
  Aborts the current operation
  */
int DFUObject::AbortOperation(void)
{
    bl_messages message;
    message.flags_command = BL_MSG_OP_ABORT;
    return SendData(message);
}

/**
  Calculates the padding used for the transfer
  Each packet can contain up to 14 32bit values
  If the entire transfer is not a multiple of 4x14byte than the last packet will contain the rest of the division
  and the size of this last packet must be transmitted on the start of the transmission
  */
DFUObject::messagePackets DFUObject::CalculatePadding(quint32 numberOfBytes)
{
    messagePackets msg;

    msg.numberOfPackets = numberOfBytes / 4 / 14;
    msg.pad = (numberOfBytes - msg.numberOfPackets * 4 * 14) / 4;
    if(msg.pad == 0) {
        msg.lastPacketCount = 14;
    }
    else {
        ++msg.numberOfPackets;
        msg.lastPacketCount = msg.pad;
    }
    return msg;
}

/**
  Starts the firmware (leaves bootloader and boots the main software)
  */
int DFUObject::JumpToApp(bool safeboot)
{
    bl_messages message;
    message.flags_command = BL_MSG_JUMP_FW;
    if(safeboot)
        message.v.jump_fw.safe_word = 0x5afe;
    else
        message.v.jump_fw.safe_word = 0x0000;
    return SendData(message);
}

/**
  Requests the current bootloader status
  */
tl_dfu::Status DFUObject::StatusRequest()
{
    bl_messages message;
    message.flags_command = BL_MSG_STATUS_REQ;
    int result = SendData(message);

    TL_DFU_QXTLOG_DEBUG(QString("StatusRequest:%0 bytes sent").arg(result));
    result = ReceiveData(message);
    qDebug()<< result;//TODO CHECK LENGHT
    TL_DFU_QXTLOG_DEBUG(QString("StatusRequest:%0 bytes received").arg(result));
    if(message.flags_command == BL_MSG_STATUS_REP)
    {
        TL_DFU_QXTLOG_DEBUG(QString("Status:%0").arg(message.v.status_rep.current_state));
        return (tl_dfu::Status)message.v.status_rep.current_state;
    }
    else
        return tl_dfu::not_in_dfu;
}

/**
  Ask the bootloader for the current device characteristics
  */
device DFUObject::findCapabilities()
{
    device currentDevice;
    TL_DFU_QXTLOG_DEBUG("FINDDEVICES BEGIN");
    bl_messages message;
    message.flags_command = BL_MSG_CAP_REQ;
    message.v.cap_req.device_number = 1;
    TL_DFU_QXTLOG_DEBUG(QString("FINDDEVICES SENDING CAPABILITIES REQUEST BUFFER_SIZE:%0").arg(BUF_LEN));
    int result = SendData(message);
    TL_DFU_QXTLOG_DEBUG(QString("FINDDEVICES CAPABILITIES REQUEST BYTES_SENT:%0").arg(result));
    if (result < 1)
        return currentDevice;

    result = ReceiveData(message);
    TL_DFU_QXTLOG_DEBUG(QString("FINDDEVICES CAPABILITIES ANSWER BYTES_RECEIVED:%0").arg(result));
    if ((result < 1) || (message.flags_command != BL_MSG_CAP_REP))
        return currentDevice;

    currentDevice.BL_Version = message.v.cap_rep_specific.bl_version;
    currentDevice.HW_Rev = message.v.cap_rep_specific.board_rev;
    if(message.v.cap_rep_specific.cap_extension_magic == BL_CAP_EXTENSION_MAGIC)
        currentDevice.CapExt = true;
    else
        currentDevice.CapExt = false;
    currentDevice.SizeOfDesc = message.v.cap_rep_specific.desc_size;
    currentDevice.ID = ntohs(message.v.cap_rep_specific.device_id);
    message.v.cap_rep_specific.device_number = 1;
    currentDevice.FW_CRC = ntohl(message.v.cap_rep_specific.fw_crc);
    currentDevice.SizeOfCode = ntohl(message.v.cap_rep_specific.fw_size);
    if(currentDevice.CapExt)
    {
        for(int partition = 0;partition < 10;++partition)
        {
            currentDevice.PartitionSizes.append(ntohl(message.v.cap_rep_specific.partition_sizes[partition]));
        }
    }
    {
        TL_DFU_QXTLOG_DEBUG(QString("Device ID=%0").arg(currentDevice.ID));
        TL_DFU_QXTLOG_DEBUG(QString("Device SizeOfCode=%0").arg(currentDevice.SizeOfCode));
        TL_DFU_QXTLOG_DEBUG(QString("Device SizeOfDesc=%0").arg(currentDevice.SizeOfDesc));
        TL_DFU_QXTLOG_DEBUG(QString("BL Version=%0").arg(currentDevice.BL_Version));
        TL_DFU_QXTLOG_DEBUG(QString("FW CRC=%0").arg(currentDevice.FW_CRC));
        if(currentDevice.PartitionSizes.size() > 0)
        {
            for(int partition = 0;partition < 10;++partition)
                TL_DFU_QXTLOG_DEBUG(QString("Partition %0 Size %1").arg(partition).arg(currentDevice.PartitionSizes.at(partition)));
            currentDevice.PartitionSizes.resize(currentDevice.PartitionSizes.indexOf(0));
        }
        else
            TL_DFU_QXTLOG_DEBUG("No partition found, probably using old bootloader");
    }
    return currentDevice;
}

/**
  Opens bootloader coms to a given USB port
  @param port USB port to use
  @returns operation success
  */
bool DFUObject::OpenBootloaderComs(USBPortInfo port)
{
    // If device was unplugged the previous coms are
    // not closed. We must close it before openning
    // a new one.
    if (open)
        CloseBootloaderComs();

    QEventLoop m_eventloop;
    QTimer::singleShot(200,&m_eventloop, SLOT(quit()));
    m_eventloop.exec();
    hid_init();
    m_hidHandle = hid_open(port.vendorID, port.productID, NULL);
    if ( m_hidHandle )
    {
        QTimer::singleShot(200,&m_eventloop, SLOT(quit()));
        m_eventloop.exec();
        AbortOperation();
        if(!EnterDFU())
        {
            TL_DFU_QXTLOG_DEBUG(QString("Could not process enterDFU command"));
            hid_close(m_hidHandle);
            return false;
        }
        if(StatusRequest() != tl_dfu::DFUidle)
        {
            TL_DFU_QXTLOG_DEBUG(QString("Status different that DFUidle after enterDFU command"));
            hid_close(m_hidHandle);
            return false;
        }

        open = true;
        return true;
    } else
    {
        TL_DFU_QXTLOG_DEBUG(QString("Could not open USB port"));
        hid_close(m_hidHandle);
        return false;
    }
    return false;
}

/**
  Close bootloader coms
  */
void DFUObject::CloseBootloaderComs()
{
    hid_close(m_hidHandle);
    m_hidHandle = NULL;
    open = false;
}

/**
  @brief :Ends the current operation
  */
bool DFUObject::EndOperation()
{
    bl_messages message;
    message.flags_command = BL_MSG_OP_END;
    int result = SendData(message);
    TL_DFU_QXTLOG_DEBUG(QString("%0 bytes sent").arg(result));
    if(result > 0)
        return true;
    return false;
}

/**
  Asynchronously uploads a partition to the board
  @param sourceArray array containing the data to upload
  @param partition destination partition
  @param size size of the data to upload
  @returns status of the board after upload
  */
bool DFUObject::UploadPartitionThreaded(QByteArray &sourceArray,dfu_partition_label partition,int size)
{
    if (isRunning())
        return false;
    threadJob.requestedOperation = ThreadJobStruc::Upload;
    threadJob.requestTransferType = partition;
    threadJob.requestStorage = &sourceArray;
    threadJob.partition_size = size;
    start();
    return true;
}

/**
  Synchronously uploads a partition to the board
  @param sourceArray array containing the data to upload
  @param partition destination partition
  @returns status of the board aftet upload
  */
tl_dfu::Status DFUObject::UploadPartition(QByteArray &sourceArray, dfu_partition_label partition)
{
    tl_dfu::Status ret;

    TL_DFU_QXTLOG_DEBUG("Starting Firmware Upload...");
    emit operationProgress(QString("Starting upload"), -1);
    TL_DFU_QXTLOG_DEBUG(QString("Bytes Loaded=%0").arg(sourceArray.length()));
    if(sourceArray.length() %4 != 0)
    {
        int pad=sourceArray.length() / 4;
        ++pad;
        pad = pad * 4;
        pad = pad - sourceArray.length();
        sourceArray.append( QByteArray(pad, 255) );
    }

    if( threadJob.partition_size < (quint32)sourceArray.length() )
    {
        TL_DFU_QXTLOG_DEBUG("ERROR array to big for device");
        return tl_dfu::abort;;
    }

    quint32 crc = DFUObject::CRCFromQBArray(sourceArray, threadJob.partition_size);
    TL_DFU_QXTLOG_DEBUG( QString("NEW FIRMWARE CRC=%0").arg(crc));

    if( !StartUpload( sourceArray.length(), partition, crc) )
    {
        ret = StatusRequest();
        TL_DFU_QXTLOG_DEBUG("StartUpload failed");
        TL_DFU_QXTLOG_DEBUG(QString("StartUpload returned:").arg(StatusToString(ret)));
        return ret;
    }
    emit operationProgress(QString("Erasing, please wait..."), -1);

    TL_DFU_QXTLOG_DEBUG( "Erasing memory");
    if( StatusRequest() == tl_dfu::abort)
    {
        TL_DFU_QXTLOG_DEBUG( "returning TL_DFU::abort");
        return tl_dfu::abort;
    }
    ret = StatusRequest();
    TL_DFU_QXTLOG_DEBUG(QString("Erase returned:%0").arg(StatusToString(ret)));

    if(ret != tl_dfu::uploading)
        return ret;

    emit operationProgress(QString(tr("Uploading %0")).arg(partitionStringFromLabel(partition)), -1);

    if( !UploadData(sourceArray.length(),sourceArray) )
    {
        ret = StatusRequest();
        TL_DFU_QXTLOG_DEBUG("Upload failed (upload data)");
        TL_DFU_QXTLOG_DEBUG(QString("UploadData returned:").arg(StatusToString(ret)));

        return ret;
    }
    if( !EndOperation() )
    {
        ret = StatusRequest();
        TL_DFU_QXTLOG_DEBUG("Upload failed (EndOperation)");
        TL_DFU_QXTLOG_DEBUG(QString("EndOperation returned:").arg(StatusToString(ret)));

        return ret;
    }
    ret = StatusRequest();
    if(ret != tl_dfu::Last_operation_Success)
        return ret;

    TL_DFU_QXTLOG_DEBUG(QString("Status=%0").arg(StatusToString(ret)));
    TL_DFU_QXTLOG_DEBUG("Firmware Uploading succeeded");
    return ret;
}

/**
  Copies one array into another inverting endianess
  @param source source array
  @param destination destination array
  @param count number of byte to copy
  */
void DFUObject::CopyWords(char *source, char *destination, int count)
{
    for (int x = 0;x < count;x = x + 4)
    {
        *(destination + x) = source[x + 3];
        *(destination + x + 1) = source[x + 2];
        *(destination + x + 2) = source[x + 1];
        *(destination + x + 3) = source[x + 0];
    }
}

/**
  Utility function
  Calculates the CRC value of an array
  */
quint32 DFUObject::CRC32WideFast(quint32 Crc, quint32 Size, quint32 *Buffer)
{
    while(Size--)
    {
        static const quint32 CrcTable[16] = { // Nibble lookup table for 0x04C11DB7 polynomial
                                              0x00000000,0x04C11DB7,0x09823B6E,0x0D4326D9,0x130476DC,0x17C56B6B,0x1A864DB2,0x1E475005,
                                              0x2608EDB8,0x22C9F00F,0x2F8AD6D6,0x2B4BCB61,0x350C9B64,0x31CD86D3,0x3C8EA00A,0x384FBDBD };

        Crc = Crc ^ *((quint32 *)Buffer); // Apply all 32-bits

        Buffer += 1;

        // Process 32-bits, 4 at a time, or 8 rounds

        Crc = (Crc << 4) ^ CrcTable[Crc >> 28]; // Assumes 32-bit reg, masking index to 4-bits
        Crc = (Crc << 4) ^ CrcTable[Crc >> 28]; //  0x04C11DB7 Polynomial used in STM32
        Crc = (Crc << 4) ^ CrcTable[Crc >> 28];
        Crc = (Crc << 4) ^ CrcTable[Crc >> 28];
        Crc = (Crc << 4) ^ CrcTable[Crc >> 28];
        Crc = (Crc << 4) ^ CrcTable[Crc >> 28];
        Crc = (Crc << 4) ^ CrcTable[Crc >> 28];
        Crc = (Crc << 4) ^ CrcTable[Crc >> 28];
    }
    return(Crc);
}

/**
  Utility function
  Calculates the CRC value of an array after padding it to the format used with the bootloader
  */
quint32 DFUObject::CRCFromQBArray(QByteArray array, quint32 Size)
{
    // If array is not an 32-bit word aligned file then
    // pad out the end to make it so like the firmware
    // expects
    if(array.length() % 4 != 0)
    {
        int pad = array.length() / 4;
        ++pad;
        pad = pad * 4;
        pad = pad - array.length();
        array.append(QByteArray(pad,255));
    }

    // If the size is greater than the provided code then
    // pad the end with 0xFF
    if ((int) Size > array.length()) {
        qDebug() << "Padding";
        quint32 pad = Size - array.length();
        array.append( QByteArray(pad, 255) );
    }

    int maxSize = (array.length() > Size) ? Size : array.length();
    quint32 t[Size / 4];
    for(int x = 0; x < maxSize / 4;x++)
    {
        quint32 aux = 0;
        aux = (char)array[x * 4 + 3] & 0xFF;
        aux = aux << 8;
        aux += (char)array[x * 4 + 2] & 0xFF;
        aux = aux << 8;
        aux += (char)array[x * 4 + 1] & 0xFF;
        aux=aux << 8;
        aux += (char)array[x * 4 + 0] & 0xFF;
        t[x] = aux;
    }
    return DFUObject::CRC32WideFast(0xFFFFFFFF, Size / 4, (quint32*)t);
}

/**
  Sends a message to the currently used USB port
  @param data data to write to port
  @returns actual bytes written
  */
int DFUObject::SendData(bl_messages data)
{
    char array[sizeof(bl_messages) + 1];
    array[0] = 0x02;
    memcpy(array + 1, &data, sizeof(bl_messages));
    return hid_write(m_hidHandle, (unsigned char *) array, BUF_LEN);
}

/**
  Receives a message from the currently used USB port
  @param data variable where the received data will be stored
  @return actual bytes read
  */
int DFUObject::ReceiveData(bl_messages &data)
{
    char array[sizeof(bl_messages) + 1];
    int received = hid_read_timeout(m_hidHandle, (unsigned char *) array, BUF_LEN, 10000);
    memcpy(&data, array + 1, sizeof(bl_messages));
    return received;
}

/**
  Converts a partition enum label to string
  @param label partition label enum value
  @return partition string representation
  */
QString DFUObject::partitionStringFromLabel(dfu_partition_label label)
{
    switch (label) {
    case DFU_PARTITION_BL:
        return QString(tr("bootloader"));
        break;
    case DFU_PARTITION_DESC:
        return QString(tr("description"));
        break;
    case DFU_PARTITION_FW:
        return QString(tr("firmware"));
        break;
    case DFU_PARTITION_SETTINGS:
        return QString(tr("settings"));
        break;
    case DFU_PARTITION_WAYPOINTS:
        return QString(tr("waypoints"));
        break;
    default:
        return QString::number(label);
        break;
    }
}

/**
  Converts a DFU status enum value to a string
  @param status: status to convert
  @return string representation of the status
  */
QString DFUObject::StatusToString(tl_dfu::Status const & status)
{
    switch(status)
    {
    case DFUidle:
        return "DFU Idle";
    case uploading:
        return "Uploading";
    case wrong_packet_received:
        return "Wrong packet received";
    case too_many_packets:
        return "Too many packets received";
    case too_few_packets:
        return "Too few packets received";
    case Last_operation_Success:
        return "Last operation success";
    case downloading:
        return "Downloading";
    case idle:
        return "Idle";
    case Last_operation_failed:
        return "Last operation failed";
    case outsideDevCapabilities:
        return "Outside Device Capabilities";
    case CRC_Fail:
        return "CRC check FAILED";
    case failed_jump:
        return "Jump to user FW failed";
    case abort:
        return "Abort";
    case uploadingStarting:
        return "Upload Starting";
    default:
        return "unknown";
    }
}
