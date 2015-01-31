/**
 ******************************************************************************
 *
 * @file       uploadergadgetwidget.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup  Uploader Uploader Plugin
 * @{
 * @brief The Tau Labs uploader plugin main widget
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

#include <QFileDialog>
#include <QMessageBox>
#include <QDesktopServices>

#include "uploadergadgetwidget.h"
#include "firmwareiapobj.h"
#include "fileutils.h"
#include "coreplugin/icore.h"
#include "rawhid/rawhidplugin.h"

using namespace uploader;

/**
 * @brief Class constructor, sets signal to slot connections,
 * creates actions, creates utility classes instances, etc
 */
UploaderGadgetWidget::UploaderGadgetWidget(QWidget *parent):QWidget(parent),
    telemetryConnected(false), iapPresent(false), iapUpdated(false)
{
    m_widget = new Ui_UploaderWidget();
    m_widget->setupUi(this);
    setUploaderStatus(uploader::DISCONNECTED);
    m_widget->partitionBrowserTW->setSelectionMode(QAbstractItemView::SingleSelection);
    m_widget->partitionBrowserTW->setSelectionBehavior(QAbstractItemView::SelectRows);

    //Setup partition manager actions
    QAction *action = new QAction("Save to file",this);
    connect(action, SIGNAL(triggered()), this, SLOT(onPartitionSave()));
    m_widget->partitionBrowserTW->addAction(action);
    action = new QAction("Flash from file",this);
    connect(action, SIGNAL(triggered()), this, SLOT(onPartitionFlash()));
    m_widget->partitionBrowserTW->addAction(action);
    action = new QAction("Erase",this);
    connect(action, SIGNAL(triggered()), this, SLOT(onPartitionErase()));
    m_widget->partitionBrowserTW->addAction(action);
    action = new QAction("Save partitions bundle",this);
    connect(action, SIGNAL(triggered()), this, SLOT(onPartitionsBundleSave()));
    m_widget->partitionBrowserTW->addAction(action);
    action = new QAction("Flash partitions bundle",this);
    connect(action, SIGNAL(triggered()), this, SLOT(onPartitionsBundleFlash()));
    m_widget->partitionBrowserTW->addAction(action);
    m_widget->partitionBrowserTW->setContextMenuPolicy(Qt::ActionsContextMenu);

    //Clear widgets to defaults
    FirmwareOnDeviceClear(true);
    FirmwareLoadedClear(true);
    PartitionBrowserClear();
    DeviceInformationClear();

    pm = ExtensionSystem::PluginManager::instance();
    telMngr = pm->getObject<TelemetryManager>();
    utilMngr = pm->getObject<UAVObjectUtilManager>();

    UAVObjectManager *obm = pm->getObject<UAVObjectManager>();
    connect(telMngr, SIGNAL(connected()), this, SLOT(onAutopilotConnect()));
    connect(telMngr, SIGNAL(disconnected()), this, SLOT(onAutopilotDisconnect()));
    firmwareIap = FirmwareIAPObj::GetInstance(obm);

    connect(firmwareIap, SIGNAL(presentOnHardwareChanged(UAVDataObject*)), this, SLOT(onIAPPresentChanged(UAVDataObject*)));
    connect(firmwareIap, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(onIAPUpdated()), Qt::UniqueConnection);

    //Connect button signals to slots
    connect(m_widget->openButton, SIGNAL(clicked()), this, SLOT(onLoadFirmwareButtonClick()));
    connect(m_widget->haltButton, SIGNAL(clicked()), this, SLOT(onHaltButtonClick()));
    connect(m_widget->rescueButton, SIGNAL(clicked()), this, SLOT(onRescueButtonClick()));
    connect(m_widget->flashButton, SIGNAL(clicked()), this, SLOT(onFlashButtonClick()));
    connect(m_widget->bootButton, SIGNAL(clicked()), this, SLOT(onBootButtonClick()));
    connect(m_widget->safeBootButton, SIGNAL(clicked()), this, SLOT(onBootButtonClick()));
    connect(m_widget->resetButton, SIGNAL(clicked()), this, SLOT(onResetButtonClick()));
    connect(m_widget->pbHelp, SIGNAL(clicked()),this,SLOT(openHelp()));
    Core::BoardManager* brdMgr = Core::ICore::instance()->boardManager();

    //Setup usb discovery signals for boards in bl state
    usbFilterBL = new USBSignalFilter(brdMgr->getKnownVendorIDs(),-1,-1,USBMonitor::Bootloader);
    connect(usbFilterBL, SIGNAL(deviceRemoved()), this, SLOT(onBootloaderRemoved()));

    conMngr = Core::ICore::instance()->connectionManager();
    connect(conMngr, SIGNAL(availableDevicesChanged(QLinkedList<Core::DevListItem>)), this, SLOT(onAvailableDevicesChanged(QLinkedList<Core::DevListItem>)));
    // Check if a board is already in bootloader state when the GCS starts
    foreach (int i, brdMgr->getKnownVendorIDs()) {
        if(USBMonitor::instance()->availableDevices(i, -1, -1, USBMonitor::Bootloader).length() > 0)
        {
            setUploaderStatus(uploader::RESCUING);
            onBootloaderDetected();
            break;
        }
    }

    bootTimeoutTimer.setInterval(3000);
    bootTimeoutTimer.setSingleShot(true);
    connect(&bootTimeoutTimer, SIGNAL(timeout()), this, SLOT(onBootingTimout()));
}

/**
 * @brief Class destructor, nothing needs to be destructed ATM
 */
UploaderGadgetWidget::~UploaderGadgetWidget()
{

}

/**
 * @brief Configure the board to use an receiver input type on a port number
 * @param type the type of receiver to use
 * @param port_num which input port to configure (board specific numbering)
 * @return true if successfully configured or false otherwise
 */

/**
 * @brief Hides or unhides the firmware on device information set
 * when hidden it presents an information not available string
 * @param clear set to true to hide information and false to unhide
 */
void UploaderGadgetWidget::FirmwareOnDeviceClear(bool clear)
{
    QRegExp rx("*OD_lbl");
    rx.setPatternSyntax(QRegExp::Wildcard);
    foreach(QLabel *l, findChildren<QLabel *>(rx))
    {
        if(clear)
            l->clear();
        l->setVisible(!clear);
    }
    m_widget->firmwareOnDevicePic->setVisible(!clear);
    m_widget->noInfoOnDevice->setVisible(clear);
    m_widget->onDeviceGB->setEnabled(!clear);
}

/**
 * @brief Hides or unhides the firmware loaded (from file) information set
 * when hidden it presents an information not available string
 * @param clear set to true to hide information and false to unhide
 */
void UploaderGadgetWidget::FirmwareLoadedClear(bool clear)
{
    QRegExp rx("*LD_lbl");
    rx.setPatternSyntax(QRegExp::Wildcard);
    foreach(QLabel *l, findChildren<QLabel *>(rx))
    {
        if(clear)
            l->clear();
        l->setVisible(!clear);
    }
    m_widget->firmwareLoadedPic->setVisible(!clear);
    m_widget->noFirmwareLoaded->setVisible(clear);
    m_widget->loadedGB->setEnabled(!clear);
    if(clear)
        m_widget->userDefined_LD_lbl->clear();
    m_widget->userDefined_LD_lbl->setVisible(!clear);
}

/**
 * @brief Clears and disables the partition browser table
 */
void UploaderGadgetWidget::PartitionBrowserClear()
{
    m_widget->partitionBrowserTW->clearContents();
    m_widget->partitionBrowserGB->setEnabled(false);
}

void UploaderGadgetWidget::DeviceInformationClear()
{
    m_widget->deviceInformationMainLayout->setVisible(false);
    m_widget->deviceInformationNoInfo->setVisible(true);
    m_widget->boardName_lbl->setVisible(false);
    currentBoard.board = NULL;
}

/**
 * @brief Fills out the device information widgets
 * @param board structure describing the connected hardware
 */
void UploaderGadgetWidget::DeviceInformationUpdate(deviceInfo board)
{
    if(!board.board)
        return;
    currentBoard = board;
    m_widget->boardName_lbl->setText(board.board->boardDescription());
    m_widget->devID_lbl->setText(QString::number(board.board->getBoardType(), 16));
    m_widget->hwRev_lbl->setText(board.hw_revision);
    m_widget->blVer_lbl->setText(board.bl_version);
    m_widget->maxCode_lbl->setText(board.max_code_size);
    m_widget->baroCap_lbl->setVisible(board.board->queryCapabilities(Core::IBoardType::BOARD_CAPABILITIES_BAROS));
    m_widget->accCap_lbl->setVisible(board.board->queryCapabilities(Core::IBoardType::BOARD_CAPABILITIES_ACCELS));
    m_widget->gyroCap_lbl->setVisible(board.board->queryCapabilities(Core::IBoardType::BOARD_CAPABILITIES_GYROS));
    m_widget->magCap_lbl->setVisible(board.board->queryCapabilities(Core::IBoardType::BOARD_CAPABILITIES_MAGS));
    m_widget->radioCap_lbl->setVisible(board.board->queryCapabilities(Core::IBoardType::BOARD_CAPABILITIES_RADIO));
    m_widget->deviceInformationMainLayout->setVisible(true);
    m_widget->deviceInformationNoInfo->setVisible(false);
    m_widget->boardPic->setPixmap(board.board->getBoardPicture().scaled(200, 200, Qt::KeepAspectRatio));
    FirmwareLoadedUpdate(loadedFile);
}

/**
 * @brief Fills out the firmware on device information widgets
 * @param firmware structure describing the firmware
 * @param crc firmware crc
 */
void UploaderGadgetWidget::FirmwareOnDeviceUpdate(deviceDescriptorStruct firmware, QString crc)
{
    m_widget->builtForOD_lbl->setText(Core::IBoardType::getBoardNameFromID(firmware.boardID()));
    m_widget->crcOD_lbl->setText(crc);
    m_widget->gitHashOD_lbl->setText(firmware.gitHash);
    m_widget->firmwareDateOD_lbl->setText(firmware.gitDate);
    m_widget->firmwareTagOD_lbl->setText(firmware.gitTag);
    m_widget->uavosSHA_OD_lbl->setText(firmware.uavoHash.toHex().toUpper());
    m_widget->userDefined_OD_lbl->setText(firmware.userDefined);
    if(firmware.certified)
    {
        QPixmap pix = QPixmap(QString(":uploader/images/application-certificate.svg"));
        m_widget->firmwareOnDevicePic->setPixmap(pix);
        m_widget->firmwareOnDevicePic->setToolTip(tr("Tagged officially released firmware build"));
    }
    else
    {
        QPixmap pix = QPixmap(QString(":uploader/images/warning.svg"));
        m_widget->firmwareOnDevicePic->setPixmap(pix);
        m_widget->firmwareOnDevicePic->setToolTip(tr("Custom Firmware Build"));
    }
    FirmwareOnDeviceClear(false);
}

/**
 * @brief Fills out the loaded firmware (from file) information widgets
 * @param firmwareArray array containing the loaded firmware
 */
void UploaderGadgetWidget::FirmwareLoadedUpdate(QByteArray firmwareArray)
{
    if(firmwareArray.isEmpty())
        return;
    deviceDescriptorStruct firmware;
    if(!utilMngr->descriptionToStructure(firmwareArray.right(100), firmware))
    {
        setStatusInfo(tr("Error parsing file metadata"), uploader::STATUSICON_FAIL);
        QPixmap pix = QPixmap(QString(":uploader/images/error.svg"));
        m_widget->firmwareLoadedPic->setPixmap(pix);
        m_widget->firmwareLoadedPic->setToolTip(tr("Error unrecognized file format, proceed at your own risk"));
        m_widget->gitHashLD_lbl->setText(tr("Error unrecognized file format, proceed at your own risk"));
        FirmwareLoadedClear(false);
        m_widget->userDefined_LD_lbl->setVisible(false);
        return;
    }
    m_widget->builtForLD_lbl->setText(Core::IBoardType::getBoardNameFromID(firmware.boardID()));
    bool ok;
    currentBoard.max_code_size.toLong(&ok);
    if(!ok)
        m_widget->crcLD_lbl->setText("Not Available");
    else if (firmwareArray.length() > currentBoard.max_code_size.toLong()) {
        m_widget->crcLD_lbl->setText("Not Available");
    } else {
        quint32 crc = dfu.CRCFromQBArray(firmwareArray, currentBoard.max_code_size.toLong());
        m_widget->crcLD_lbl->setText(QString::number(crc));
    }
    m_widget->gitHashLD_lbl->setText(firmware.gitHash);
    m_widget->firmwareDateLD_lbl->setText(firmware.gitDate);
    m_widget->firmwareTagLD_lbl->setText(firmware.gitTag);
    m_widget->uavosSHA_LD_lbl->setText(firmware.uavoHash.toHex().toUpper());
    m_widget->userDefined_LD_lbl->setText(firmware.userDefined);
    if(currentBoard.board && (currentBoard.board->getBoardType() != firmware.boardType))
    {
        QPixmap pix = QPixmap(QString(":uploader/images/error.svg"));
        m_widget->firmwareLoadedPic->setPixmap(pix);
        m_widget->firmwareLoadedPic->setToolTip(tr("Error firmware loaded firmware doesn't match connected board"));
    }
    else if(firmware.certified)
    {
        QPixmap pix = QPixmap(QString(":uploader/images/application-certificate.svg"));
        m_widget->firmwareLoadedPic->setPixmap(pix);
        m_widget->firmwareLoadedPic->setToolTip(tr("Tagged officially released firmware build"));
    }
    else
    {
        QPixmap pix = QPixmap(QString(":uploader/images/warning.svg"));
        m_widget->firmwareLoadedPic->setPixmap(pix);
        m_widget->firmwareLoadedPic->setToolTip(tr("Custom Firmware Build"));
    }
    FirmwareLoadedClear(false);
}

/**
 * @brief slot called by the telemetryManager when the autopilot connects
 */
void UploaderGadgetWidget::onAutopilotConnect()
{
    telemetryConnected = true;
    bootTimeoutTimer.stop();
    CheckAutopilotReady();
}

/**
 * @brief slot called by the telemetryManager when the autopilot disconnects
 */
void UploaderGadgetWidget::onAutopilotDisconnect()
{
    FirmwareOnDeviceClear(true);
    DeviceInformationClear();
    PartitionBrowserClear();
    telemetryConnected = false;
    iapPresent = false;
    iapUpdated = false;
    if( (getUploaderStatus() == uploader::RESCUING) || (getUploaderStatus() == uploader::HALTING) || (getUploaderStatus() == uploader::BOOTING))
        return;
    setUploaderStatus(uploader::DISCONNECTED);
    setStatusInfo(tr("Telemetry disconnected"), uploader::STATUSICON_INFO);
}

/**
 * @brief Called when the autopilot is ready to be queried for information
 * regarding running firmware and hardware specs
 */
void UploaderGadgetWidget::onAutopilotReady()
{
    setUploaderStatus(uploader::CONNECTED_TO_TELEMETRY);
    setStatusInfo(tr("Telemetry connected"), uploader::STATUSICON_INFO);
    deviceInfo board;
    board.bl_version = tr("Not available");
    board.board = utilMngr->getBoardType();
    board.cpu_serial = utilMngr->getBoardCPUSerial().toHex();
    board.hw_revision = QString::number(utilMngr->getBoardRevision());
    board.max_code_size = tr("Not available");
    DeviceInformationUpdate(board);
    deviceDescriptorStruct device;
    if(utilMngr->getBoardDescriptionStruct(device))
        FirmwareOnDeviceUpdate(device, QString::number(utilMngr->getFirmwareCRC(), 16));
}

/**
 * @brief slot called by the iap object when the hardware reports it
 * has it
 */
void UploaderGadgetWidget::onIAPPresentChanged(UAVDataObject *obj)
{
    iapPresent = obj->getIsPresentOnHardware();
    if(obj->getIsPresentOnHardware())
        CheckAutopilotReady();
}

/**
 * @brief slot called by the iap object when it is updated
 */
void UploaderGadgetWidget::onIAPUpdated()
{
    if( (getUploaderStatus() == uploader::RESCUING) || (getUploaderStatus() == uploader::HALTING) )
        return;
    iapUpdated = true;
    CheckAutopilotReady();
}

/**
 * @brief slot called when the user presses the load firmware button
 * It loads the file and calls the information display functions
 */
void UploaderGadgetWidget::onLoadFirmwareButtonClick()
{
    QString board;
    if(currentBoard.board)
        board = currentBoard.board->shortName().toLower();
    QString filename = LoadFirmwareFileDialog(board);
    if(filename.isEmpty())
    {
        setStatusInfo(tr("Invalid Filename"), uploader::STATUSICON_FAIL);
        return;
    }
    QFile file(filename);
    if(!file.open(QIODevice::ReadOnly))
    {
        setStatusInfo(tr("Could not open file"), uploader::STATUSICON_FAIL);
        return;
    }
    loadedFile = file.readAll();

    FirmwareLoadedClear(true);
    FirmwareLoadedUpdate(loadedFile);
    setUploaderStatus(getUploaderStatus());
}

/**
 * @brief slot called when the user presses the flash firmware button
 * It start a non blocking firmware upload
 */
void UploaderGadgetWidget::onFlashButtonClick()
{
    setStatusInfo("",uploader::STATUSICON_RUNNING);
    previousStatus = uploaderStatus;
    setUploaderStatus(uploader::UPLOADING_FW);
    connect(&dfu, SIGNAL(operationProgress(QString,int)), this, SLOT(onStatusUpdate(QString, int)));
    connect(&dfu, SIGNAL(uploadFinished(tl_dfu::Status)), this, SLOT(onUploadFinish(tl_dfu::Status)));
    dfu.UploadPartitionThreaded(loadedFile, DFU_PARTITION_FW, currentBoard.max_code_size.toInt());
}

/**
 * @brief slot called when the user presses the rescue button
 * It enables the bootloader detection and starts a detection timer
 */
void UploaderGadgetWidget::onRescueButtonClick()
{
    conMngr->suspendPolling();
    setUploaderStatus(uploader::RESCUING);
    setStatusInfo(tr("Please connect the board with USB with no external power applied"), uploader::STATUSICON_INFO);
    onRescueTimer(true);
}

/**
 * @brief slot called by the USBSignalFilter when it detects a board in bootloader state
 * Fills out partition info, device info, and firmware info
 */
void UploaderGadgetWidget::onBootloaderDetected()
{
    Core::BoardManager* brdMgr = Core::ICore::instance()->boardManager();
    QList<USBPortInfo> devices;
    foreach(int vendorID, brdMgr->getKnownVendorIDs()) {
        devices.append(USBMonitor::instance()->availableDevices(vendorID,-1,-1,USBMonitor::Bootloader));
    }
    if(devices.length() > 1)
    {
        setStatusInfo(tr("More than one device was detected in bootloader state"), uploader::STATUSICON_INFO);
        return;
    }
    else if(devices.length() == 0)
    {
        setStatusInfo("No devices in bootloader state detected", uploader::STATUSICON_FAIL);
        return;
    }
    if(dfu.OpenBootloaderComs(devices.first()))
    {
        tl_dfu::device dev = dfu.findCapabilities();

        //Bootloader has new cap extensions, query partitions and fill out browser
        if(dev.CapExt)
        {
            QTableWidgetItem *label;
            QTableWidgetItem *size;
            int index = 0;
            QString name;
            m_widget->partitionBrowserGB->setEnabled(true);
            m_widget->partitionBrowserTW->setRowCount(dev.PartitionSizes.length());
            foreach (int i, dev.PartitionSizes) {
                switch (index) {
                case DFU_PARTITION_FW:
                    name = "Firmware";
                    break;
                case DFU_PARTITION_DESC:
                    name = "Metadata";
                    break;
                case DFU_PARTITION_BL:
                    name = "Bootloader";
                    break;
                case DFU_PARTITION_SETTINGS:
                    name = "Settings";
                    break;
                case DFU_PARTITION_WAYPOINTS:
                    name = "Waypoints";
                    break;
                case DFU_PARTITION_LOG:
                    name = "Log";
                    break;
                case DFU_PARTITION_OTA:
                    name = "OTA";
                    break;
                default:
                    name = QString::number(index);
                    break;
                }
                label = new QTableWidgetItem(name);
                size = new QTableWidgetItem(QString::number(i));
                size->setTextAlignment(Qt::AlignRight);
                m_widget->partitionBrowserTW->setItem(index, 0, label);
                m_widget->partitionBrowserTW->setItem(index, 1, size);
                ++index;
            }
        }
        deviceInfo info;
        QList <Core::IBoardType *> boards = pm->getObjects<Core::IBoardType>();
        foreach (Core::IBoardType *board, boards) {
            if (board->getBoardType() == (dev.ID>>8))
            {
                info.board = board;
                break;
            }
        }
        usbFilterBL->disconnect(this, SLOT(onBootloaderDetected()));
        info.bl_version = QString::number(dev.BL_Version, 16);
        info.cpu_serial = "Not Available";
        info.hw_revision = QString::number(dev.HW_Rev);
        info.max_code_size = QString::number(dev.SizeOfCode);
        QByteArray description = dfu.DownloadDescriptionAsByteArray(dev.SizeOfDesc);
        deviceDescriptorStruct descStructure;
        if(!UAVObjectUtilManager::descriptionToStructure(description, descStructure))
            setStatusInfo(tr("Could not parse firmware metadata"), uploader::STATUSICON_INFO);
        else
        {
            FirmwareOnDeviceUpdate(descStructure, QString::number((dev.FW_CRC)));
        }
        DeviceInformationUpdate(info);
        switch (uploaderStatus) {
        case uploader::HALTING:
            setUploaderStatus(uploader::BL_FROM_HALT);
            break;
        case uploader::RESCUING:
            setUploaderStatus(uploader::BL_FROM_RESCUE);
            break;
        default:
            break;
        }
        setStatusInfo(tr("Connection to bootloader successful"), uploader::STATUSICON_OK);
        emit bootloaderDetected();
    }
    else
    {
        setStatusInfo(tr("Could not open coms with the bootloader"), uploader::STATUSICON_FAIL);
        setUploaderStatus((uploader::DISCONNECTED));
        conMngr->resumePolling();
    }
}

/**
 * @brief slot called by the USBSignalFilter when it detects a board in bootloader state as been removed
 */
void UploaderGadgetWidget::onBootloaderRemoved()
{
    if( (getUploaderStatus() == uploader::RESCUING) || (getUploaderStatus() == uploader::HALTING) )
        return;
    conMngr->resumePolling();
    setStatusInfo(tr("Bootloader disconnection detected"), uploader::STATUSICON_INFO);
    DeviceInformationClear();
    FirmwareOnDeviceClear(true);
    PartitionBrowserClear();
    if(getUploaderStatus() != uploader::BOOTING)
        setUploaderStatus(uploader::DISCONNECTED);
    dfu.disconnect();
}

/**
 * @brief slot called by a timer created by itself or by the onRescueButton
 * @param start true if the rescue timer is to be started
 */
void UploaderGadgetWidget::onRescueTimer(bool start)
{
    static int progress;
    static QTimer timer;
    if(sender() == usbFilterBL)
    {
        timer.stop();
        m_widget->progressBar->setValue(0);
        disconnect(usbFilterBL, SIGNAL(deviceDiscovered()), this, SLOT(onBootloaderDetected()));
        disconnect(usbFilterBL, SIGNAL(deviceDiscovered()), this, SLOT(onRescueTimer()));
        rescueFinish(true);
        return;
    }
    if(start)
    {
        setStatusInfo(tr("Trying to detect bootloader"), uploader::STATUSICON_INFO);
        progress = 100;
        connect(&timer, SIGNAL(timeout()), this, SLOT(onRescueTimer()),Qt::UniqueConnection);
        timer.start(200);
        connect(usbFilterBL, SIGNAL(deviceDiscovered()), this, SLOT(onBootloaderDetected()), Qt::UniqueConnection);
        connect(usbFilterBL, SIGNAL(deviceDiscovered()), this, SLOT(onRescueTimer()), Qt::UniqueConnection);
        emit rescueTimer(0);
    }
    else
    {
        progress -= 1;
        emit rescueTimer(100 - progress);
    }
    m_widget->progressBar->setValue(progress);
    if(progress == 0)
    {
        disconnect(usbFilterBL, SIGNAL(deviceDiscovered()), this, SLOT(onBootloaderDetected()));
        disconnect(usbFilterBL, SIGNAL(deviceDiscovered()), this, SLOT(onRescueTimer()));
        timer.disconnect();
        setStatusInfo(tr("Failed to detect bootloader"), uploader::STATUSICON_FAIL);
        setUploaderStatus(uploader::DISCONNECTED);
        emit rescueFinish(false);
        conMngr->resumePolling();
    }
}

/**
 * @brief slot called by the DFUObject to update operations status
 * @param text text corresponding to the status
 * @param progress 0 to 100 corresponding the the current operation progress
 */
void UploaderGadgetWidget::onStatusUpdate(QString text, int progress)
{
    if(!text.isEmpty())
        setStatusInfo(text, uploader::STATUSICON_RUNNING);
    if(progress != -1)
        m_widget->progressBar->setValue(progress);
    if( (getUploaderStatus() == uploader::UPLOADING_FW) || (getUploaderStatus() == uploader::UPLOADING_DESC) )
    {
        emit uploadProgress(getUploaderStatus(), progress);
    }
}

/**
 * @brief slot called the DFUObject when an upload operation finishes
 * @param stat upload result
 */
void UploaderGadgetWidget::onUploadFinish(Status stat)
{
    switch (uploaderStatus) {
    case uploader::UPLOADING_FW:
        if(stat == Last_operation_Success)
        {
            setStatusInfo(tr("Firmware upload success"), uploader::STATUSICON_OK);
            if (loadedFile.right(100).startsWith("TlFw") || loadedFile.right(100).startsWith("OpFw")) {
                tempArray.clear();
                tempArray.append(loadedFile.right(100));
                tempArray.chop(20);
                QString user("                    ");
                user = user.replace(0, m_widget->userDefined_LD_lbl->text().length(), m_widget->userDefined_LD_lbl->text());
                tempArray.append(user.toLatin1());
                setStatusInfo(tr("Starting firmware metadata upload"), uploader::STATUSICON_INFO);
                dfu.UploadPartitionThreaded(tempArray, DFU_PARTITION_DESC, 100);
                setUploaderStatus(uploader::UPLOADING_DESC);
            }
            else
            {
                setUploaderStatus(previousStatus);
                dfu.disconnect();
            }
        }
        else
        {
            setUploaderStatus(previousStatus);
            setStatusInfo(tr("Firmware upload failed"), uploader::STATUSICON_FAIL);
            dfu.disconnect();
            lastUploadResult = false;
            uploadFinish(false);
        }
        break;
    case uploader::UPLOADING_DESC:
        if(stat == Last_operation_Success)
        {
            setStatusInfo(tr("Firmware and firmware metadata upload success"), uploader::STATUSICON_OK);
            lastUploadResult = true;
            emit uploadFinish(true);
        }
        else
        {
            setStatusInfo(tr("Firmware metadata upload failed"), uploader::STATUSICON_FAIL);
            lastUploadResult = false;
            uploadFinish(false);
        }
        dfu.disconnect();
        setUploaderStatus(previousStatus);
        break;
    case uploader::UPLOADING_PARTITION:
        if(stat == Last_operation_Success)
            setStatusInfo(tr("Partition upload success"), uploader::STATUSICON_OK);
        else
            setStatusInfo(tr("Partition upload failed"), uploader::STATUSICON_FAIL);
        dfu.disconnect();
        setUploaderStatus(previousStatus);
        break;
    case uploader::UPLOADING_PARTITION_BUNDLE:
    {
        bool result = false;
        if(stat == Last_operation_Success)
        {
            setStatusInfo(tr("Partition upload success"), uploader::STATUSICON_OK);
            result = true;
        }
        else
            setStatusInfo(tr("Partition upload failed"), uploader::STATUSICON_FAIL);
        ProcessPartitionBundleFlash(result);
    }
        break;
    default:
        break;
    }
}

/**
 * @brief slot called the DFUObject when a download operation finishes
 * @param result true if the download was successfull
 */
void UploaderGadgetWidget::onDownloadFinish(bool result)
{
    switch (uploaderStatus) {
    case uploader::DOWNLOADING_PARTITION_BUNDLE:
        ProcessPartitionBundleSave(result);
        break;
    case uploader::DOWNLOADING_PARTITION:
        dfu.disconnect();
        if(result)
        {
            setStatusInfo(tr("Partition download success"), uploader::STATUSICON_OK);
            QString filename = QFileDialog::getSaveFileName(this, tr("Save File"),QDir::homePath(),"*.bin");
            if(filename.isEmpty())
            {
                setStatusInfo(tr("Error, empty filename"), uploader::STATUSICON_FAIL);
                setUploaderStatus(previousStatus);
                return;
            }

            if(!filename.endsWith(".bin",Qt::CaseInsensitive))
                filename.append(".bin");
            QFile file(filename);
            if(file.open(QIODevice::WriteOnly))
            {
                file.write(tempArray);
                file.close();
                setStatusInfo(tr("Partition written to file"), uploader::STATUSICON_OK);
            }
            else
                setStatusInfo(tr("Error could not open file for save"), uploader::STATUSICON_FAIL);
        }
        else
            setStatusInfo(tr("Partition download failed"), uploader::STATUSICON_FAIL);
        setUploaderStatus(previousStatus);
        break;
    default:
        break;
    }

}

/**
 * @brief slot called when the user clicks save on the partition browser
 */
void UploaderGadgetWidget::onPartitionSave()
{
    if(!CheckInBootloaderState())
        return;
    int index = m_widget->partitionBrowserTW->selectedItems().first()->row();
    int size = m_widget->partitionBrowserTW->item(index,1)->text().toInt();

    setStatusInfo("",uploader::STATUSICON_RUNNING);
    previousStatus = uploaderStatus;
    setUploaderStatus(uploader::DOWNLOADING_PARTITION);
    connect(&dfu, SIGNAL(operationProgress(QString,int)), this, SLOT(onStatusUpdate(QString, int)));
    connect(&dfu, SIGNAL(downloadFinished(bool)), this, SLOT(onDownloadFinish(bool)));
    tempArray.clear();
    dfu.DownloadPartitionThreaded(&tempArray, (dfu_partition_label)index, size);
}

/**
 * @brief slot called when the user clicks flash on the partition browser
 */
void UploaderGadgetWidget::onPartitionFlash()
{
    if(!CheckInBootloaderState())
        return;
    QString filename = QFileDialog::getOpenFileName(this, tr("Open File"),QDir::homePath(),"*.bin");
    if(filename.isEmpty())
    {
        setStatusInfo(tr("Error, empty filename"), uploader::STATUSICON_FAIL);
        return;
    }
    QFile file(filename);
    if(!file.open(QIODevice::ReadOnly))
    {
        setStatusInfo(tr("Could not open file for read"), uploader::STATUSICON_FAIL);
        return;
    }
    tempArray.clear();
    tempArray = file.readAll();

    int index = m_widget->partitionBrowserTW->selectedItems().first()->row();
    int size = m_widget->partitionBrowserTW->item(index,1)->text().toInt();

    if(tempArray.length() > size)
    {
        setStatusInfo(tr("File bigger than partition size"), uploader::STATUSICON_FAIL);
    }
    if(QMessageBox::warning(this, tr("Warning"), tr("Are you sure you want to flash the selected partition?"),QMessageBox::Yes, QMessageBox::No) != QMessageBox::Yes)
        return;
    setStatusInfo("",uploader::STATUSICON_RUNNING);
    previousStatus = uploaderStatus;
    setUploaderStatus(uploader::UPLOADING_PARTITION);
    connect(&dfu, SIGNAL(operationProgress(QString,int)), this, SLOT(onStatusUpdate(QString, int)));
    connect(&dfu, SIGNAL(uploadFinished(tl_dfu::Status)), this, SLOT(onUploadFinish(tl_dfu::Status)));
    dfu.UploadPartitionThreaded(tempArray, (dfu_partition_label)index, size);
}

/**
 * @brief slot called when the user clicks erase on the partition browser
 */
void UploaderGadgetWidget::onPartitionErase()
{
    int index = m_widget->partitionBrowserTW->selectedItems().first()->row();
    if(QMessageBox::warning(this, tr("Warning"), tr("Are you sure you want erase the selected partition?"),QMessageBox::Yes, QMessageBox::No) != QMessageBox::Yes)
        return;
    if(dfu.WipePartition((dfu_partition_label)index))
        setStatusInfo(tr("Partition erased"), uploader::STATUSICON_OK);
    else
        setStatusInfo(tr("Partition erasure failed"), uploader::STATUSICON_FAIL);
}

/**
 * @brief slot called when the user clicks bundle save on the partition browser
 * This creates a zip file with all the partitions binaries
 */
void UploaderGadgetWidget::onPartitionsBundleSave()
{
    if(!CheckInBootloaderState())
        return;
    int count = m_widget->partitionBrowserTW->rowCount();

    setStatusInfo("",uploader::STATUSICON_RUNNING);
    previousStatus = uploaderStatus;
    setUploaderStatus(uploader::DOWNLOADING_PARTITION_BUNDLE);
    connect(&dfu, SIGNAL(operationProgress(QString,int)), this, SLOT(onStatusUpdate(QString, int)));
    connect(&dfu, SIGNAL(downloadFinished(bool)), this, SLOT(onDownloadFinish(bool)));
    ProcessPartitionBundleSave(true, count);
}

/**
 * @brief slot called when the user clicks bundle save on the partition browser
 * This opens a zip file of users choice extracts it and flashes the binaries included
 * to the correspondent partitions
 */
void UploaderGadgetWidget::onPartitionsBundleFlash()
{
    if(!CheckInBootloaderState())
        return;
    setStatusInfo("",uploader::STATUSICON_RUNNING);
    ProcessPartitionBundleFlash(true, true);
}

/**
 * @brief slot called when the user clicks the boot button
 * Atempts to Boot the board and starts a timeout timer
 */
void UploaderGadgetWidget::onBootButtonClick()
{
    if(!CheckInBootloaderState())
        return;
    setUploaderStatus(uploader::BOOTING);
    bootTimeoutTimer.start();
    bool safeboot = (sender() == m_widget->safeBootButton);
    dfu.JumpToApp(safeboot);
    dfu.CloseBootloaderComs();
}

/**
 * @brief slot called by the boot timeout timer
 */
void UploaderGadgetWidget::onBootingTimout()
{
    if(getUploaderStatus() == uploader::BOOTING)
        setUploaderStatus(uploader::DISCONNECTED);
}

/**
 * @brief Processes the partition bundle flashing, this gets called everytime
 * one of the partitions gets flashed
 * @param result result of last partition save
 * @param start true if the partition bundle flash is to be started from the first partition
 */
void UploaderGadgetWidget::ProcessPartitionBundleFlash(bool result, bool start)
{
    static QList<partitionStruc> arrayList;
    static QList<int> failedUploads;
    static int lastPartition;
    if(start)
    {
        lastPartition = -1;
        arrayList.clear();
        failedUploads.clear();
        QString fileName = QFileDialog::getOpenFileName(this,
                                                        tr("Select bundle file"),
                                                        QDir::homePath(),
                                                        tr("Bundle File (*.zip)"));
        QDir dir = QDir::temp();
        if(!dir.mkdir("tlbundleextract"))
        {
            setStatusInfo(tr("Error could not create temporary directory"), uploader::STATUSICON_FAIL);
            return;
        }
        dir.cd("tlbundleextract");
        if(!FileUtils::extractAll(fileName, dir))
        {
            setStatusInfo(tr("Error could not create temporary directory"), uploader::STATUSICON_FAIL);
            return;
        }
        foreach (QFileInfo fileInfo, dir.entryInfoList(QDir::Files, QDir::Name)) {
            QFile file(fileInfo.absoluteFilePath());
            if(!file.open(QIODevice::ReadOnly))
            {
                setStatusInfo(tr("Error could not open temporary files"), uploader::STATUSICON_FAIL);
                return;
            }
            partitionStruc p;
            p.partitionData = file.readAll();
            p.partitionNumber = fileInfo.fileName().remove(".bin").toInt();
            arrayList.append(p);
            file.close();
        }
        foreach (partitionStruc p, arrayList) {
            qDebug()<<p.partitionNumber<<"<<"<<p.partitionData.length();
        }
        FileUtils::removeDir(dir.absolutePath());
        previousStatus = uploaderStatus;
        setUploaderStatus(uploader::UPLOADING_PARTITION_BUNDLE);
        connect(&dfu, SIGNAL(operationProgress(QString,int)), this, SLOT(onStatusUpdate(QString, int)));
        connect(&dfu, SIGNAL(uploadFinished(tl_dfu::Status)), this, SLOT(onUploadFinish(tl_dfu::Status)));
    }
    if(!arrayList.isEmpty())
    {
        if( (lastPartition != -1) && !result )
            failedUploads.append(lastPartition);
        partitionStruc p = arrayList.first();
        lastPartition = p.partitionNumber;
        tempArray = p.partitionData;
        arrayList.removeFirst();
        setStatusInfo(QString(tr("Uploading %0 partition")).arg(dfu.partitionStringFromLabel((dfu_partition_label)p.partitionNumber)), uploader::STATUSICON_RUNNING);
        dfu.UploadPartitionThreaded(tempArray, dfu_partition_label(p.partitionNumber), tempArray.length());
    }
    else
    {
        dfu.disconnect();
        setUploaderStatus(previousStatus);
        if(failedUploads.length() == 0)
            setStatusInfo(tr("Partitions bundle written to flash"), uploader::STATUSICON_OK);
        else
        {
            QString failed;
            foreach (int i, failedUploads) {
                failed.append(dfu.partitionStringFromLabel((dfu_partition_label)i)+ ", ");
            }
            failed = failed.left(failed.length() -2);
            setStatusInfo(QString(tr("The following partitions failed to be flashed:%0").arg(failed)), uploader::STATUSICON_FAIL);
        }
    }
}

/**
 * @brief Processes the partition bundle saving, this gets called everytime
 * one of the partitions gets downloaded
 * @param result result of last partition download
 * @param count number of partitions to save
 */
void UploaderGadgetWidget::ProcessPartitionBundleSave(bool result, int count)
{
    static QList<QByteArray> arrayList;
    static QList<int> failedSaves;
    static int m_count = 0;
    static int m_current_partition;
    if(count != -1)
    {
        m_count = count;
        m_current_partition = 0;
    }
    else
    {
        if(!result)
            failedSaves.append(m_count);
        arrayList.append(tempArray);
    }
    if(m_current_partition == m_count)
    {
        dfu.disconnect();
        setUploaderStatus(previousStatus);
        QDir dir = QDir::temp();
        if(!dir.mkdir("tlbundle"))
        {
            setStatusInfo(tr("Error could not create temporary directory"), uploader::STATUSICON_FAIL);
            return;
        }
        dir.cd("tlbundle");
        for(int x = 0;x < m_count; ++x)
        {
            if(!failedSaves.contains(x))
            {
                QFile file(dir.absolutePath() + QDir::separator() + QString::number(x) + ".bin");
                if(!file.open(QIODevice::WriteOnly))
                {
                    setStatusInfo(tr("Error could not save temporary file"), uploader::STATUSICON_FAIL);
                    return;
                }
                QByteArray array = arrayList.at(x);
                file.write(array);
                file.close();
            }
        }
        QString filename = QFileDialog::getSaveFileName(this, tr("Save File"),QDir::homePath(),"*.zip");
        if(filename.isEmpty())
        {
            setStatusInfo(tr("Error, empty filename"), uploader::STATUSICON_FAIL);
            return;
        }

        if(!filename.endsWith(".zip",Qt::CaseInsensitive))
            filename.append(".zip");
        if(FileUtils::archive(filename, dir, "tlbundle", ""))
        {
            if(failedSaves.length() == 0)
                setStatusInfo(tr("Partitions bundle written to file"), uploader::STATUSICON_OK);
            else
            {
                QString failed;
                foreach (int i, failedSaves) {
                    failed.append(dfu.partitionStringFromLabel((dfu_partition_label)i)+ ", ");
                }
                failed = failed.left(failed.length() -2);
                setStatusInfo(QString(tr("The following partitions failed to load:%0").arg(failed)), uploader::STATUSICON_FAIL);
            }
        }
        else
            setStatusInfo(tr("Error could not open file for save"), uploader::STATUSICON_FAIL);
        FileUtils::removeDir(dir.absolutePath());
        arrayList.clear();
        failedSaves.clear();
        return;
    }
    int size = m_widget->partitionBrowserTW->item(m_current_partition, 1)->text().toInt();
    tempArray.clear();
    dfu.DownloadPartitionThreaded(&tempArray, (dfu_partition_label)m_current_partition, size);
    ++m_current_partition;
}

/**
 * @brief Checks if all conditions are met for the autopilot to be ready for required operations
 */
void UploaderGadgetWidget::CheckAutopilotReady()
{
    if(telemetryConnected && iapPresent && iapUpdated)
    {
        onAutopilotReady();
        utilMngr->versionMatchCheck();
    }
}

/**
 * @brief Check if the board is in bootloader state
 * @return true if in bootloader state
 */
bool UploaderGadgetWidget::CheckInBootloaderState()
{
    if( (uploaderStatus != uploader::BL_FROM_HALT) && (uploaderStatus != uploader::BL_FROM_RESCUE) )
    {
        setStatusInfo(tr("Cannot perform the selected operation if not in bootloader state"), uploader::STATUSICON_FAIL);
        return false;
    }
    else
        return true;
}

/**
 *Opens an open file dialog pointing to the file which corresponds to the connected board if possible
 */
QString UploaderGadgetWidget::LoadFirmwareFileDialog(QString boardName)
{
    QFileDialog::Options options;
    QString selectedFilter;
    QString fwDirectoryStr;
    QDir fwDirectory;
    boardName = boardName.toLower();
    //Format filename for file chooser
#ifdef Q_OS_WIN
    fwDirectoryStr = QCoreApplication::applicationDirPath();
    fwDirectory = QDir(fwDirectoryStr);
    fwDirectory.cdUp();
    fwDirectory.cd("firmware");
    fwDirectoryStr = fwDirectory.absolutePath();
#elif defined Q_OS_LINUX
    fwDirectoryStr = QCoreApplication::applicationDirPath();
    fwDirectory = QDir(fwDirectoryStr);
    fwDirectory.cd("../../..");
    fwDirectoryStr = fwDirectory.absolutePath();
    fwDirectoryStr = fwDirectoryStr + "/fw_" + boardName +"/fw_"+ boardName +".tlfw";
#elif defined Q_OS_MAC
    fwDirectoryStr = QCoreApplication::applicationDirPath();
    fwDirectory = QDir(fwDirectoryStr);
    fwDirectory.cd("../../../../../..");
    fwDirectoryStr = fwDirectory.absolutePath();
    fwDirectoryStr = fwDirectoryStr+"/fw_" + boardName +"/fw_"+ boardName +".tlfw";
#endif
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Select firmware file"),
                                                    fwDirectoryStr,
                                                    tr("Firmware Files (*.opfw *.tlfw *.bin)"),
                                                    &selectedFilter,
                                                    options);
    return fileName;
}

/**
 * @brief Updates the status message
 * @param str string containing the text which describes the current status
 * @param ic icon which describes the current status
 */
void UploaderGadgetWidget::setStatusInfo(QString str, uploader::StatusIcon ic)
{
    QPixmap px;
    m_widget->statusLabel->setText(str);
    switch (ic) {
    case uploader::STATUSICON_RUNNING:
        px.load(QString(":/uploader/images/system-run.svg"));
        break;
    case uploader::STATUSICON_OK:
        px.load(QString(":/uploader/images/dialog-apply.svg"));
        break;
    case uploader::STATUSICON_FAIL:
        px.load(QString(":/uploader/images/process-stop.svg"));
        break;
    default:
        px.load(QString(":/uploader/images/gtk-info.svg"));
    }
    m_widget->statusPic->setPixmap(px);
}

/**
 * @brief slot called when the user clicks reset
 */
void UploaderGadgetWidget::onResetButtonClick()
{

    lastConnectedTelemetryDevice = conMngr->getCurrentDevice().device.data()->getName();
    if(!telMngr->isConnected() || !firmwareIap->getIsPresentOnHardware())
        return;
    previousStatus = getUploaderStatus();
    setUploaderStatus(uploader::HALTING);
    QEventLoop loop;
    QTimer timeout;
    timeout.setSingleShot(true);
    firmwareIap->setBoardRevision(0);
    firmwareIap->setBoardType(0);
    connect(&timeout, SIGNAL(timeout()), &loop, SLOT(quit()));
    connect(firmwareIap,SIGNAL(transactionCompleted(UAVObject*,bool)), &loop, SLOT(quit()));
    int magicValue = 1122;
    int magicStep = 1111;
    for(int i = 0; i < 3; ++i)
    {
        //Firmware IAP module specifies that the timing between iap commands must be
        //between 500 and 5000ms
        timeout.start(600);
        loop.exec();
        firmwareIap->setCommand(magicValue);
        magicValue += magicStep;
        if(magicValue == 3344)
            magicValue = 4455;
        setStatusInfo(QString(tr("Sending IAP Step %0").arg(i + 1)), uploader::STATUSICON_INFO);
        firmwareIap->updated();
        timeout.start(1000);
        loop.exec();
        if(!timeout.isActive())
        {
            setStatusInfo(QString(tr("Sending IAP Step %0 failed").arg(i + 1)), uploader::STATUSICON_FAIL);
            setUploaderStatus(previousStatus);
            return;
        }
        timeout.stop();
    }
    setUploaderStatus(uploader::BOOTING);
    conMngr->disconnectDevice();
    timeout.start(200);
    loop.exec();
    conMngr->suspendPolling();
    bootTimeoutTimer.start();
    return;
}

/**
 * @brief slot by connectionManager when new devices arrive
 * Used to reconnect the boards after booting if autoconnect is disabled
 */
void UploaderGadgetWidget::onAvailableDevicesChanged(QLinkedList<Core::DevListItem> devList)
{
    if(conMngr->getAutoconnect() || conMngr->isConnected() || lastConnectedTelemetryDevice.isEmpty())
        return;
    foreach (Core::DevListItem item, devList) {
        if(item.device.data()->getName() == lastConnectedTelemetryDevice)
        {
            conMngr->connectDevice(item);
        }
    }
}

/**
 * @brief slot called when the user clicks halt
 */
void UploaderGadgetWidget::onHaltButtonClick()
{
    lastConnectedTelemetryDevice = conMngr->getCurrentDevice().device.data()->getName();
    if(!telMngr->isConnected() || !firmwareIap->getIsPresentOnHardware())
        return;
    previousStatus = getUploaderStatus();
    setUploaderStatus(uploader::HALTING);
    QEventLoop loop;
    QTimer timeout;
    timeout.setSingleShot(true);
    firmwareIap->setBoardRevision(0);
    firmwareIap->setBoardType(0);
    connect(&timeout, SIGNAL(timeout()), &loop, SLOT(quit()));
    connect(firmwareIap,SIGNAL(transactionCompleted(UAVObject*,bool)), &loop, SLOT(quit()));
    int magicValue = 1122;
    int magicStep = 1111;
    for(int i = 0; i < 3; ++i)
    {
        //Firmware IAP module specifies that the timing between iap commands must be
        //between 500 and 5000ms
        timeout.start(600);
        loop.exec();
        firmwareIap->setCommand(magicValue);
        magicValue += magicStep;
        setStatusInfo(QString(tr("Sending IAP Step %0").arg(i + 1)), uploader::STATUSICON_INFO);
        firmwareIap->updated();
        timeout.start(1000);
        loop.exec();
        if(!timeout.isActive())
        {
            setStatusInfo(QString(tr("Sending IAP Step %0 failed").arg(i + 1)), uploader::STATUSICON_FAIL);
            setUploaderStatus(previousStatus);
            return;
        }
        timeout.stop();
    }
    conMngr->disconnectDevice();
    timeout.start(200);
    loop.exec();
    conMngr->suspendPolling();
    onRescueTimer(true);
    return;
}

/**
 * @brief Returns the current uploader status
 */
uploader::UploaderStatus UploaderGadgetWidget::getUploaderStatus() const
{
    return uploaderStatus;
}

/**
 * @brief Sets the current uploader status
 * Enables, disables, hides, unhides widgets according to the new status
 * @param value value of the new status
 */
void UploaderGadgetWidget::setUploaderStatus(const uploader::UploaderStatus &value)
{
    qDebug()<< "STATUS="<<value;
    uploaderStatus = value;
    switch (uploaderStatus) {
    case uploader::DISCONNECTED:
        m_widget->rescueButton->setVisible(true);
        m_widget->openButton->setVisible(true);
        m_widget->haltButton->setVisible(false);
        m_widget->bootButton->setVisible(false);
        m_widget->safeBootButton->setVisible(false);
        m_widget->resetButton->setVisible(false);
        m_widget->flashButton->setVisible(false);

        m_widget->rescueButton->setEnabled(true);
        m_widget->openButton->setEnabled(true);
        m_widget->haltButton->setEnabled(true);
        m_widget->bootButton->setEnabled(true);
        m_widget->safeBootButton->setEnabled(true);
        m_widget->resetButton->setEnabled(true);
        m_widget->flashButton->setEnabled(true);
        m_widget->partitionBrowserTW->setContextMenuPolicy(Qt::NoContextMenu);
        break;
    case uploader::HALTING:
        m_widget->rescueButton->setVisible(false);
        m_widget->openButton->setVisible(true);
        m_widget->haltButton->setVisible(true);
        m_widget->bootButton->setVisible(false);
        m_widget->safeBootButton->setVisible(false);
        m_widget->resetButton->setVisible(false);
        m_widget->flashButton->setVisible(false);

        m_widget->rescueButton->setEnabled(true);
        m_widget->openButton->setEnabled(true);
        m_widget->haltButton->setEnabled(false);
        m_widget->bootButton->setEnabled(true);
        m_widget->safeBootButton->setEnabled(true);
        m_widget->resetButton->setEnabled(true);
        m_widget->flashButton->setEnabled(true);
        m_widget->partitionBrowserTW->setContextMenuPolicy(Qt::NoContextMenu);
        break;
    case uploader::RESCUING:
        m_widget->rescueButton->setVisible(true);
        m_widget->openButton->setVisible(true);
        m_widget->haltButton->setVisible(false);
        m_widget->bootButton->setVisible(false);
        m_widget->safeBootButton->setVisible(false);
        m_widget->resetButton->setVisible(false);
        m_widget->flashButton->setVisible(false);

        m_widget->rescueButton->setEnabled(false);
        m_widget->openButton->setEnabled(true);
        m_widget->haltButton->setEnabled(true);
        m_widget->bootButton->setEnabled(true);
        m_widget->safeBootButton->setEnabled(true);
        m_widget->resetButton->setEnabled(true);
        m_widget->flashButton->setEnabled(true);
        m_widget->partitionBrowserTW->setContextMenuPolicy(Qt::NoContextMenu);
        break;
    case uploader::BL_FROM_HALT:
    case uploader::BL_FROM_RESCUE:
        m_widget->rescueButton->setVisible(false);
        m_widget->openButton->setVisible(true);
        m_widget->haltButton->setVisible(false);
        m_widget->bootButton->setVisible(true);
        m_widget->safeBootButton->setVisible(true);
        m_widget->resetButton->setVisible(false);
        m_widget->flashButton->setVisible(true);


        m_widget->rescueButton->setEnabled(true);
        m_widget->openButton->setEnabled(true);
        m_widget->haltButton->setEnabled(true);
        m_widget->bootButton->setEnabled(true);
        m_widget->safeBootButton->setEnabled(true);
        m_widget->resetButton->setEnabled(true);
        if(!loadedFile.isEmpty())
            m_widget->flashButton->setEnabled(true);
        else
            m_widget->flashButton->setEnabled(false);
        m_widget->partitionBrowserTW->setContextMenuPolicy(Qt::ActionsContextMenu);
        break;
    case uploader::CONNECTED_TO_TELEMETRY:
        m_widget->rescueButton->setVisible(false);
        m_widget->openButton->setVisible(true);
        if(conMngr->getCurrentDevice().connection->shortName() == "USB")
            m_widget->haltButton->setVisible(true);
        else
            m_widget->haltButton->setVisible(false);
        m_widget->bootButton->setVisible(false);
        m_widget->safeBootButton->setVisible(false);
        m_widget->resetButton->setVisible(true);
        m_widget->flashButton->setVisible(false);

        m_widget->rescueButton->setEnabled(true);
        m_widget->openButton->setEnabled(true);
        m_widget->haltButton->setEnabled(true);
        m_widget->bootButton->setEnabled(true);
        m_widget->safeBootButton->setEnabled(true);
        m_widget->resetButton->setEnabled(true);
        m_widget->flashButton->setEnabled(true);
        m_widget->partitionBrowserTW->setContextMenuPolicy(Qt::NoContextMenu);
        break;
    case uploader::UPLOADING_FW:
    case uploader::UPLOADING_DESC:
    case uploader::DOWNLOADING_PARTITION:
    case uploader::UPLOADING_PARTITION:
    case uploader::DOWNLOADING_PARTITION_BUNDLE:
    case uploader::UPLOADING_PARTITION_BUNDLE:
        m_widget->rescueButton->setVisible(false);
        m_widget->openButton->setVisible(true);
        m_widget->haltButton->setVisible(false);
        m_widget->bootButton->setVisible(true);
        m_widget->safeBootButton->setVisible(true);
        m_widget->resetButton->setVisible(false);
        m_widget->flashButton->setVisible(true);

        m_widget->rescueButton->setEnabled(true);
        m_widget->openButton->setEnabled(false);
        m_widget->haltButton->setEnabled(true);
        m_widget->bootButton->setEnabled(false);
        m_widget->safeBootButton->setEnabled(false);
        m_widget->resetButton->setEnabled(false);
        m_widget->flashButton->setEnabled(false);
        m_widget->partitionBrowserTW->setContextMenuPolicy(Qt::NoContextMenu);
        break;
    case uploader::BOOTING:
        m_widget->rescueButton->setVisible(false);
        m_widget->openButton->setVisible(true);
        m_widget->haltButton->setVisible(false);
        m_widget->bootButton->setVisible(true);
        m_widget->safeBootButton->setVisible(true);
        m_widget->resetButton->setVisible(false);
        m_widget->flashButton->setVisible(true);

        m_widget->rescueButton->setEnabled(false);
        m_widget->openButton->setEnabled(false);
        m_widget->haltButton->setEnabled(false);
        m_widget->bootButton->setEnabled(false);
        m_widget->safeBootButton->setEnabled(false);
        m_widget->resetButton->setEnabled(false);
        m_widget->flashButton->setEnabled(false);
        m_widget->partitionBrowserTW->setContextMenuPolicy(Qt::NoContextMenu);
    default:
        break;
    }
}

//! Check that the resources for auto updating are compiled in
bool UploaderGadgetWidget::autoUpdateCapable()
{
    return QDir(":/build").exists();
}

/**
 * @brief Slot indirecly called by the wizzard plugin to autoupdate the board with the
 * embedded firmware
 * @return true if the update was successful
 */
bool UploaderGadgetWidget::autoUpdate()
{
    Core::BoardManager* brdMgr = Core::ICore::instance()->boardManager();
    QTimer timer;
    timer.setSingleShot(true);
    QEventLoop loop;
    bool stillLooking = true;
    while(stillLooking)
    {
        stillLooking = false;
        foreach(int vendorID, brdMgr->getKnownVendorIDs()) {
            stillLooking |= (USBMonitor::instance()->availableDevices(vendorID,-1,-1,-1).length()>0);
        }
        emit autoUpdateSignal(WAITING_DISCONNECT,QVariant());
        QTimer::singleShot(500, &loop, SLOT(quit()));
        loop.exec();
    }
    emit autoUpdateSignal(WAITING_CONNECT,0);
    connect(this, SIGNAL(rescueTimer(int)), this, SLOT(onAutoUpdateCount(int)));
    connect(this, SIGNAL(rescueFinish(bool)), &loop, SLOT(quit()));
    onRescueButtonClick();
    loop.exec();
    disconnect(this, SIGNAL(rescueTimer(int)), this, SLOT(onAutoUpdateCount(int)));
    disconnect(this, SIGNAL(rescueFinish(bool)), &loop, SLOT(quit()));
    if(getUploaderStatus() != uploader::BL_FROM_RESCUE)
    {
        emit autoUpdateSignal(FAILURE,QVariant());
        return false;
    }

    // Find firmware file in resources based on board name
    QString m_filename;
    QString board;
    if(currentBoard.board)
        board = currentBoard.board->shortName().toLower();
    else
    {
        emit autoUpdateSignal(FAILURE,QVariant());
        return false;
    }
    m_filename = QString("fw_").append(board);
    m_filename=":/build/"+m_filename+"/"+m_filename+".tlfw";
    if(!QFile::exists(m_filename))
    {
        emit autoUpdateSignal(FAILURE_FILENOTFOUND,QVariant());
        return false;
    }
    QFile m_file(m_filename);
    if (!m_file.open(QIODevice::ReadOnly)) {
        emit autoUpdateSignal(FAILURE,QVariant());
        return false;
    }
    loadedFile = m_file.readAll();
    FirmwareLoadedClear(true);
    FirmwareLoadedUpdate(loadedFile);
    setUploaderStatus(getUploaderStatus());

    onFlashButtonClick();

    connect(this, SIGNAL(uploadProgress(UploaderStatus,QVariant)), this, SIGNAL(autoUpdateSignal(UploaderStatus,QVariant)));
    connect(this, SIGNAL(uploadFinish(bool)), &loop, SLOT(quit()));
    loop.exec();
    disconnect(this, SIGNAL(uploadProgress(UploaderStatus,QVariant)), this, SIGNAL(autoUpdateSignal(UploaderStatus,QVariant)));
    disconnect(this, SIGNAL(uploadFinish(bool)), &loop, SLOT(quit()));
    onBootButtonClick();
    if(!lastUploadResult)
    {
        emit autoUpdateSignal(FAILURE, QVariant());
        return false;
    }
    emit autoUpdateSignal(SUCCESS, QVariant());
    return true;
}

/**
 * @brief Adapts uploader logic to wizzard plugin logic
 */
void UploaderGadgetWidget::onAutoUpdateCount(int i)
{
    emit autoUpdateSignal(WAITING_CONNECT, i);
}

/**
 * @brief Opens the plugin help page on the default browser
 * TODO ADD SPECIFIC NG PAGE TO THE WIKI
 */
void UploaderGadgetWidget::openHelp()
{
    QDesktopServices::openUrl( QUrl("https://github.com/TauLabs/TauLabs/wiki/OnlineHelp:-Uploader-Plugin", QUrl::StrictMode) );
}
