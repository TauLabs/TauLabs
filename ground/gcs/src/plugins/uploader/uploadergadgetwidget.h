/**
 ******************************************************************************
 *
 * @file       uploadergadgetwidget.h
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

#ifndef UPLOADERGADGETWIDGET_H
#define UPLOADERGADGETWIDGET_H

#include <QPointer>
#include "ui_uploader.h"
#include "tl_dfu.h"
#include <coreplugin/iboardtype.h>
#include "uploader_global.h"
#include "devicedescriptorstruct.h"
#include "uavobjectutilmanager.h"
#include "uavtalk/telemetrymanager.h"
#include "coreplugin/connectionmanager.h"

using namespace tl_dfu;

namespace uploader {

typedef enum { STATUSICON_OK, STATUSICON_RUNNING, STATUSICON_FAIL, STATUSICON_INFO} StatusIcon;

class UPLOADER_EXPORT UploaderGadgetWidget : public QWidget
{
    Q_OBJECT
    struct deviceInfo
    {
        QPointer<Core::IBoardType> board;
        QString bl_version;
        QString max_code_size;
        QString cpu_serial;
        QString hw_revision;
    };
    struct partitionStruc
    {
        QByteArray partitionData;
        int partitionNumber;
    };

public:
    UploaderGadgetWidget(QWidget *parent = 0);
    ~UploaderGadgetWidget();
    bool autoUpdateCapable();
public slots:
    bool autoUpdate();
signals:
    void bootloaderDetected();
    void rescueTimer(int);
    void rescueFinish(bool);
    void uploadFinish(bool);
    void uploadProgress(UploaderStatus, QVariant);
    void autoUpdateSignal(UploaderStatus, QVariant);
private slots:
    void onAutopilotConnect();
    void onAutopilotDisconnect();
    void onAutopilotReady();
    void onIAPPresentChanged(UAVDataObject*);
    void onIAPUpdated();
    void onLoadFirmwareButtonClick();
    void onHaltButtonClick();
    void onFlashButtonClick();
    void onRescueButtonClick();
    void onBootloaderDetected();
    void onBootloaderRemoved();
    void onRescueTimer(bool start = false);
    void onStatusUpdate(QString, int);
    void onUploadFinish(tl_dfu::Status);
    void onDownloadFinish(bool);
    void onPartitionSave();
    void onPartitionFlash();
    void onPartitionErase();
    void onPartitionsBundleSave();
    void onPartitionsBundleFlash();
    void onBootButtonClick();
    void onBootingTimout();
    void onAutoUpdateCount(int i);
    void openHelp();
    void onResetButtonClick();
    void onAvailableDevicesChanged(QLinkedList<Core::DevListItem>);
private:
    void FirmwareOnDeviceClear(bool clear);
    void FirmwareLoadedClear(bool clear);
    void PartitionBrowserClear();
    void DeviceInformationClear();
    void DeviceInformationUpdate(deviceInfo board);
    void FirmwareOnDeviceUpdate(deviceDescriptorStruct firmware, QString crc);
    void FirmwareLoadedUpdate(QByteArray firmwareArray);
    QString LoadFirmwareFileDialog(QString);
    uploader::UploaderStatus getUploaderStatus() const;
    void setUploaderStatus(const uploader::UploaderStatus &value);
    void CheckAutopilotReady();
    bool CheckInBootloaderState();
    void setStatusInfo(QString str, uploader::StatusIcon ic);
    void ProcessPartitionBundleFlash(bool result, bool start = false);
    void ProcessPartitionBundleSave(bool result, int count = -1);

    Ui_UploaderWidget *m_widget;
    bool telemetryConnected;
    bool iapPresent;
    bool iapUpdated;
    UAVObjectUtilManager* utilMngr;
    QByteArray loadedFile;
    DFUObject dfu;
    ExtensionSystem::PluginManager *pm;
    USBSignalFilter *usbFilterBL;
    TelemetryManager* telMngr;
    Core::ConnectionManager *conMngr;
    FirmwareIAPObj *firmwareIap;
    deviceInfo currentBoard;
    QString lastConnectedTelemetryDevice;
    uploader::UploaderStatus uploaderStatus;
    uploader::UploaderStatus previousStatus;
    QByteArray tempArray;
    bool lastUploadResult;
    QTimer bootTimeoutTimer;
};
}
#endif // UPLOADERGADGETWIDGET_H
