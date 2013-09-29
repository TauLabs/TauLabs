/**
 ******************************************************************************
 *
 * @file       devicewidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
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
#include "devicewidget.h"

deviceWidget::deviceWidget(QWidget *parent) :
    QWidget(parent)
{
    myDevice = new Ui_deviceWidget();
    myDevice->setupUi(this);

    // Initialization of the Device icon display
    myDevice->verticalGroupBox_loaded->setVisible(false);
    myDevice->groupCustom->setVisible(false);
    myDevice->youdont->setVisible(false);
    myDevice->gVDevice->setScene(new QGraphicsScene(this));
    connect(myDevice->retrievePartBundleButton, SIGNAL(clicked()), this, SLOT(downloadPartitionBundle()));
    connect(myDevice->updateButton, SIGNAL(clicked()), this, SLOT(uploadFirmware()));
    connect(myDevice->pbLoad, SIGNAL(clicked()), this, SLOT(loadFirmware()));
    connect(myDevice->youdont, SIGNAL(stateChanged(int)), this, SLOT(confirmCB(int)));
    connect(myDevice->cbShowPartBrowser, SIGNAL(stateChanged(int)),this,SLOT(showHidePartBrowser(int)));
    myDevice->cbShowPartBrowser->setChecked(false);
    myDevice->gbPartitionBrowser->setVisible(false);

    QPixmap pix = QPixmap(QString(":uploader/images/view-refresh.svg"));
    myDevice->statusIcon->setPixmap(pix);

    myDevice->lblCertified->setText("");

    QAction *partReadAct = new QAction("&Read to file", this);
    partReadAct->setShortcut(tr("Ctrl+R"));
    partReadAct->setStatusTip(tr("Reads the current partition(s) to a file"));
    connect(partReadAct, SIGNAL(triggered()), this, SLOT(readPartitions()));

    QAction *partWriteAct = new QAction("&Write from file", this);
    partWriteAct->setShortcut(tr("Ctrl+W"));
    partWriteAct->setStatusTip(tr("Writes the chosen file(s) to the current partition(s)"));
    connect(partWriteAct, SIGNAL(triggered()), this, SLOT(writePartitions()));

    QAction *partWipeAct = new QAction("Wipe", this);
    partWipeAct->setStatusTip(tr("Wipes the current partition(s)"));
    connect(partWipeAct, SIGNAL(triggered()), this, SLOT(wipePartitions()));

    QAction *partBundleAct = new QAction("Write partition &bundle file", this);
    partBundleAct->setShortcut(tr("Ctrl+b"));
    partBundleAct->setStatusTip(tr("Writes a partition bundle file to the device"));
    connect(partBundleAct, SIGNAL(triggered()), this, SLOT(writeBundlePartitions()));

    myDevice->tablePartitions->insertAction(0,partBundleAct);
    myDevice->tablePartitions->insertAction(partBundleAct,partWipeAct);
    myDevice->tablePartitions->insertAction(partWipeAct,partWriteAct);
    myDevice->tablePartitions->insertAction(partWriteAct,partReadAct);
}


void deviceWidget::showEvent(QShowEvent *event)
{
    Q_UNUSED(event)
    // Thit fitInView method should only be called now, once the
    // widget is shown, otherwise it cannot compute its values and
    // the result is usually a ahrsbargraph that is way too small
    myDevice->gVDevice->fitInView(devicePic.rect(),Qt::KeepAspectRatio);
}

void deviceWidget::resizeEvent(QResizeEvent* event)
{
    Q_UNUSED(event);
    myDevice->gVDevice->fitInView(devicePic.rect(), Qt::KeepAspectRatio);
}


void deviceWidget::setDeviceID(int devID){
    deviceID = devID;
}

void deviceWidget::setDfu(DFUObject *dfu)
{
    m_dfu = dfu;
    connect(m_dfu, SIGNAL(progressUpdated(int)), this, SLOT(setProgress(int)));
    connect(m_dfu, SIGNAL(operationProgress(QString)), this, SLOT(dfuStatus(QString)));
    connect(m_dfu, SIGNAL(uploadFinished(OP_DFU::Status)), this, SLOT(uploadFinished(OP_DFU::Status)));
    connect(m_dfu, SIGNAL(downloadFinished(bool)), this, SLOT(downloadFinished(bool)));
}

/**
  Fills the various fields for the device
  */
void deviceWidget::populate()
{

    unsigned int id = m_dfu->devices[deviceID].ID;
    myDevice->lbldevID->setText(QString("Device ID: ") + QString::number(id & 0xFFFF, 16));
    // DeviceID tells us what sort of HW we have detected:
    // display a nice icon:
    myDevice->gVDevice->scene()->clear();
    myDevice->lblDevName->setText(deviceDescriptorStruct::idToBoardName(id));
    myDevice->lblHWRev->setText(QString(tr("HW Revision: "))+QString::number(id & 0x00FF, 16));

    devicePic = deviceDescriptorStruct::idToBoardPicture(id);

    myDevice->gVDevice->scene()->addPixmap(devicePic);
    myDevice->gVDevice->setSceneRect(devicePic.rect());
    myDevice->gVDevice->fitInView(devicePic.rect(),Qt::KeepAspectRatio);

    bool r = m_dfu->devices[deviceID].Readable;
    bool w = m_dfu->devices[deviceID].Writable;

    myDevice->lblAccess->setText(QString("Flash access: ") + QString(r ? "R" : "-") + QString(w ? "W" : "-"));
    myDevice->lblMaxCode->setText(QString("Max code size: ") +QString::number(m_dfu->devices[deviceID].SizeOfCode));
    myDevice->lblCRC->setText(QString::number(m_dfu->devices[deviceID].FW_CRC));
    myDevice->lblBLVer->setText(QString("BL version: ") + QString::number(m_dfu->devices[deviceID].BL_Version));
    QString moreRecent;
    bool checked;
    if(m_dfu->devices[deviceID].BL_Version < 128)
    {
        moreRecent = (tr("You need a more recent bootloader to use this function"));
        checked = false;
        myDevice->cbShowPartBrowser->setChecked(checked);
    }
    else
    {
        moreRecent = "";
        checked = true;
    }
    myDevice->cbShowPartBrowser->setEnabled(checked);
    myDevice->cbShowPartBrowser->setToolTip(moreRecent);
    myDevice->retrievePartBundleButton->setEnabled(checked);
    myDevice->retrievePartBundleButton->setToolTip(moreRecent);
    int x = 0;
    while(true)
    {
        if(x >= m_dfu->devices[deviceID].PartitionSizes.size())
            break;
        if(m_dfu->devices[deviceID].PartitionSizes.at(x) == 0)
            break;
        myDevice->tablePartitions->insertRow(x);
        QTableWidgetItem *item = new QTableWidgetItem;
        item->setText(QString::number(x));
        myDevice->tablePartitions->setItem(x,0,item);
        item = new QTableWidgetItem;
        item->setText(QString::number(m_dfu->devices[deviceID].PartitionSizes.at(x)));
        myDevice->tablePartitions->setItem(x,1,item);
        ++x;
    }
    int size=((OP_DFU::device)m_dfu->devices[deviceID]).SizeOfDesc;
    m_dfu->enterDFU(deviceID);
    QByteArray desc = m_dfu->DownloadDescriptionAsBA(size);

    if (! populateBoardStructuredDescription(desc)) {
        //TODO
        // desc was not a structured description
        QString str = m_dfu->DownloadDescription(size);
        myDevice->lblDescription->setText(QString("Firmware custom description: ")+str.left(str.indexOf(QChar(255))));
        QPixmap pix = QPixmap(QString(":uploader/images/warning.svg"));
        myDevice->lblCertified->setPixmap(pix);
        myDevice->lblCertified->setToolTip(tr("Custom Firmware Build"));
        myDevice->lblBuildDate->setText("Warning: development firmware");
        myDevice->lblGitTag->setText("");
        myDevice->lblBrdName->setText("");
    }

    status("Ready...", STATUSICON_INFO);
}

/**
  Freezes the contents of the widget so that a user cannot
  try to modify the contents
  */
void deviceWidget::freeze()
{
    myDevice->pbLoad->setEnabled(false);
    myDevice->updateButton->setEnabled(false);
    myDevice->retrievePartBundleButton->setEnabled(false);
}

/**
  Unfreezes the contents of the widget so that a user can again
  try to modify the contents
  */
void deviceWidget::unfreeze()
{
    myDevice->pbLoad->setEnabled(true);
    myDevice->updateButton->setEnabled(true);
    myDevice->retrievePartBundleButton->setEnabled(true);
}

/**
  Populates the widget field with the description in case
  it is structured properly
  */
bool deviceWidget::populateBoardStructuredDescription(QByteArray desc)
{
    if(UAVObjectUtilManager::descriptionToStructure(desc,onBoardDescription))
    {
        myDevice->lblGitTag->setText(onBoardDescription.gitHash);
        myDevice->lblBuildDate->setText(onBoardDescription.gitDate.insert(4,"-").insert(7,"-"));
        if(onBoardDescription.gitTag.startsWith("RELEASE",Qt::CaseSensitive))
        {
            myDevice->lblDescription->setText(onBoardDescription.gitTag);
            QPixmap pix = QPixmap(QString(":uploader/images/application-certificate.svg"));
            myDevice->lblCertified->setPixmap(pix);
            myDevice->lblCertified->setToolTip(tr("Tagged officially released firmware build"));

        }
        else
        {
            myDevice->lblDescription->setText(onBoardDescription.gitTag);
            QPixmap pix = QPixmap(QString(":uploader/images/warning.svg"));
            myDevice->lblCertified->setPixmap(pix);
            myDevice->lblCertified->setToolTip(tr("Untagged or custom firmware build"));
        }

        myDevice->lblBrdName->setText(deviceDescriptorStruct::idToBoardName(onBoardDescription.boardType << 8 | onBoardDescription.boardRevision));

        return true;
    }

    return false;

}
bool deviceWidget::populateLoadedStructuredDescription(QByteArray desc)
{
    if(UAVObjectUtilManager::descriptionToStructure(desc,LoadedDescription))
    {
        myDevice->lblGitTagL->setText(LoadedDescription.gitHash);
        myDevice->lblBuildDateL->setText( LoadedDescription.gitDate.insert(4,"-").insert(7,"-"));
        if(LoadedDescription.gitTag.startsWith("RELEASE",Qt::CaseSensitive))
        {
            myDevice->lblDescritpionL->setText(LoadedDescription.gitTag);
            myDevice->description->setText(LoadedDescription.gitTag);
            QPixmap pix = QPixmap(QString(":uploader/images/application-certificate.svg"));
            myDevice->lblCertifiedL->setPixmap(pix);
            myDevice->lblCertifiedL->setToolTip(tr("Tagged officially released firmware build"));
        }
        else
        {
            myDevice->lblDescritpionL->setText(LoadedDescription.gitTag);
            myDevice->description->setText(LoadedDescription.gitTag);
            QPixmap pix = QPixmap(QString(":uploader/images/warning.svg"));
            myDevice->lblCertifiedL->setPixmap(pix);
            myDevice->lblCertifiedL->setToolTip(tr("Untagged or custom firmware build"));
        }
        myDevice->lblBrdNameL->setText(deviceDescriptorStruct::idToBoardName(LoadedDescription.boardType << 8 | LoadedDescription.boardRevision));

        return true;
    }

    return false;

}
/**
  Updates status message for messages coming from DFU
  */
void deviceWidget::dfuStatus(QString str)
{
    status(str, STATUSICON_RUNNING);
}

void deviceWidget::confirmCB(int value)
{
    if(value==Qt::Checked)
    {
        myDevice->updateButton->setEnabled(true);
    }
    else
        myDevice->updateButton->setEnabled(false);
}

/**
  Updates status message
  */
void deviceWidget::status(QString str, StatusIcon ic)
{
    QPixmap px;
    myDevice->statusLabel->setText(str);
    switch (ic) {
    case STATUSICON_RUNNING:
        px.load(QString(":/uploader/images/system-run.svg"));
        break;
    case STATUSICON_OK:
        px.load(QString(":/uploader/images/dialog-apply.svg"));
        break;
    case STATUSICON_FAIL:
        px.load(QString(":/uploader/images/process-stop.svg"));
        break;
    default:
        px.load(QString(":/uploader/images/gtk-info.svg"));
    }
    myDevice->statusIcon->setPixmap(px);
}


void deviceWidget::loadFirmware()
{
    myDevice->verticalGroupBox_loaded->setVisible(false);
    myDevice->groupCustom->setVisible(false);

    filename = setOpenFileName();

    if (filename.isEmpty()) {
        status("Empty filename", STATUSICON_FAIL);
        return;
    }

    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        status("Can't open file", STATUSICON_FAIL);
        return;
    }

    loadedFW = file.readAll();
    myDevice->youdont->setVisible(false);
    myDevice->youdont->setChecked(false);
    QByteArray desc = loadedFW.right(100);
    QPixmap px;
    if((quint32)loadedFW.length()>m_dfu->devices[deviceID].SizeOfCode)
        myDevice->lblCRCL->setText(tr("Can't calculate, file too big for device"));
    else
        myDevice->lblCRCL->setText( QString::number(DFUObject::CRCFromQBArray(loadedFW,m_dfu->devices[deviceID].SizeOfCode)));
    //myDevice->lblFirmwareSizeL->setText(QString("Firmware size: ")+QVariant(loadedFW.length()).toString()+ QString(" bytes"));
    if (populateLoadedStructuredDescription(desc))
    {
        myDevice->youdont->setChecked(true);
        myDevice->verticalGroupBox_loaded->setVisible(true);
        myDevice->groupCustom->setVisible(false);
        if(myDevice->lblCRC->text()==myDevice->lblCRCL->text())
        {
            myDevice->statusLabel->setText(tr("The board has the same firmware as loaded. No need to update"));
            px.load(QString(":/uploader/images/warning.svg"));
        }
        else if(myDevice->lblDevName->text()!=myDevice->lblBrdNameL->text())
        {
            myDevice->statusLabel->setText(tr("WARNING: the loaded firmware is for different hardware. Do not update!"));
            px.load(QString(":/uploader/images/error.svg"));
        }
        else if(QDateTime::fromString(onBoardDescription.gitDate)>QDateTime::fromString(LoadedDescription.gitDate))
        {
            myDevice->statusLabel->setText(tr("The board has newer firmware than loaded. Are you sure you want to update?"));
            px.load(QString(":/uploader/images/warning.svg"));
        }
        else if(!LoadedDescription.gitTag.startsWith("RELEASE",Qt::CaseSensitive))
        {
            myDevice->statusLabel->setText(tr("The loaded firmware is untagged or custom build. Update only if it was received from a trusted source (official website or your own build)"));
            px.load(QString(":/uploader/images/warning.svg"));
        }
        else
        {
            myDevice->statusLabel->setText(tr("This is the tagged officially released Tau Labs firmware"));
            px.load(QString(":/uploader/images/gtk-info.svg"));
        }
    }
    else
    {
        myDevice->statusLabel->setText(tr("WARNING: the loaded firmware was not packaged with the Tau Labs format. Do not update unless you know what you are doing"));
        px.load(QString(":/uploader/images/error.svg"));
        myDevice->youdont->setChecked(false);
        myDevice->youdont->setVisible(true);
        myDevice->verticalGroupBox_loaded->setVisible(false);
        myDevice->groupCustom->setVisible(true);
    }
    myDevice->statusIcon->setPixmap(px);
}

/**
  Sends a firmware to the device
  */
void deviceWidget::uploadFirmware()
{
    freeze();
    if (!m_dfu->devices[deviceID].Writable) {
        status("Device not writable!", STATUSICON_FAIL);
        unfreeze();
        return;
    }

    bool verify = false;
    /* TODO: does not work properly on current Bootloader!
    if (m_dfu->devices[deviceID].Readable)
        verify = true;
     */

    QByteArray desc = loadedFW.right(100);
    if (desc.startsWith("TlFw") || desc.startsWith("OpFw")) {
        descriptionArray = desc;
        // Now do sanity checking:
        // - Check whether board type matches firmware:
        int board = m_dfu->devices[deviceID].ID;
        int firmwareBoard = ((desc.at(12)&0xff)<<8) + (desc.at(13)&0xff);
        if((board == 0x401 && firmwareBoard == 0x402) ||
                (board == 0x901 && firmwareBoard == 0x902) || // L3GD20 revo supports Revolution firmware
                (board == 0x902 && firmwareBoard == 0x903))   // RevoMini1 supporetd by RevoMini2 firmware
        {
            // These firmwares are designed to be backwards compatible
        } else if (firmwareBoard != board) {
            status("Error: firmware does not match board", STATUSICON_FAIL);
            unfreeze();
            return;
        }
        // Check the firmware embedded in the file:
        QByteArray firmwareHash = desc.mid(40,20);
        QByteArray fileHash = QCryptographicHash::hash(loadedFW.left(loadedFW.length()-100), QCryptographicHash::Sha1);
        if (firmwareHash != fileHash) {
            status("Error: firmware file corrupt", STATUSICON_FAIL);
            unfreeze();
            return;
        }
    } else {
        // The firmware is not packaged, just upload the text in the description field
        // if it is there.
        descriptionArray.clear();
    }


    status("Starting firmware upload", STATUSICON_RUNNING);
    // We don't know which device was used previously, so we
    // are cautious and reenter DFU for this deviceID:
    emit uploadStarted();
    if(!m_dfu->enterDFU(deviceID))
    {
        status("Error:Could not enter DFU mode", STATUSICON_FAIL);
        unfreeze();
        emit uploadEnded(false);
        return;
    }
    OP_DFU::Status retstatus=m_dfu->StatusRequest();
    qDebug() << m_dfu->StatusToString(retstatus);
    m_dfu->AbortOperation(); // Necessary, otherwise I get random failures.

    bool ret = m_dfu->UploadPartition(filename,verify, deviceID,OP_DFU::FW,m_dfu->devices[deviceID].SizeOfCode);
    if(!ret ) {
        status("Could not start upload", STATUSICON_FAIL);
        unfreeze();
        emit uploadEnded(false);
        return;
    }
    status("Uploading, please wait...", STATUSICON_RUNNING);
    QEventLoop m_eventloop;
    connect(this,SIGNAL(uploadedFinishedCallback()),&m_eventloop,SLOT(quit()));
    m_eventloop.exec();
    if(last_upload_status != OP_DFU::Last_operation_Success)
    {
        emit uploadEnded(false);
        unfreeze();
        return;
    }
    if (!descriptionArray.isEmpty()) {
        // We have a structured array to save
        status(QString("Updating description"), STATUSICON_RUNNING);
        repaint(); // Make sure the text above shows right away
        retstatus = m_dfu->UploadDescription(descriptionArray);
        if( retstatus != OP_DFU::Last_operation_Success) {
            status(QString("Upload failed with code: ") + m_dfu->StatusToString(retstatus).toLatin1().data(), STATUSICON_FAIL);
            emit uploadEnded(false);
            unfreeze();
            return; emit uploadEnded(false);
            unfreeze();
            return;
        }

    } else if (!myDevice->description->text().isEmpty()) {
        // Fallback: we save the description field:
        status(QString("Updating description"), STATUSICON_RUNNING);
        repaint(); // Make sure the text above shows right away
        retstatus = m_dfu->UploadDescription(myDevice->description->text());
        if( retstatus != OP_DFU::Last_operation_Success) {
            status(QString("Upload failed with code: ") + m_dfu->StatusToString(retstatus).toLatin1().data(), STATUSICON_FAIL);
            emit uploadEnded(false);
            unfreeze();
            return;
        }
    }
    unfreeze();
    populate();
    emit uploadEnded(true);
}

/**
  Retrieves a partition bundle from the device
  */
void deviceWidget::downloadPartitionBundle()
{
    freeze();
    QEventLoop loop;
    QByteArray array;
    connect(this, SIGNAL(downloadFinishedCallback()), &loop, SLOT(quit()));
    QString bundleDirName("taulabs_partitions");
    QDir bundleDir;
    bundleDir = QDir::temp();
    if(!bundleDir.mkdir(bundleDirName))
    {
        status(tr("Could not create temporary directory"),STATUSICON_FAIL);
        unfreeze();
        return;
    }
    bundleDir.cd(bundleDirName);
    for(int i = 0;i < myDevice->tablePartitions->rowCount();++i)
    {
        int partition = myDevice->tablePartitions->item(i,0)->text().toInt();
        QString filename = bundleDir.absolutePath()+QDir::separator()+QString::number(partition)+".bin";
        status(QString("Downloading partition #%0 from device").arg(QString::number(i)), STATUSICON_RUNNING);
        array.clear(); // Empty the byte array
        int partition_size = myDevice->tablePartitions->item(i,1)->text().toInt();

        bool ret = m_dfu->DownloadPartition(&array,deviceID,partition,partition_size);
        if(!ret) {
            status("Could not start download!", STATUSICON_FAIL);
            continue;
        }
        loop.exec();
        if(last_download_success)
        {
            status(QString(tr("Partition #%0 successfully downloaded")).arg(i),STATUSICON_OK);
            m_dfu->SaveByteArrayToFile(filename, array);
        }
    }
    QString filename = QFileDialog::getSaveFileName(this,QString(tr("Select file to write the partitions bundle file to")),QDir::homePath(), "Compressed file (*.zip)");
    if(FileUtils::archive(filename,bundleDir,tr("TauLabs partitions bundle file")))
        status(tr("Partition bundle successfully saved"),STATUSICON_OK);
    else
        status(tr("Failed to save bundle"),STATUSICON_FAIL);
    FileUtils::removeDir(bundleDir.absolutePath());
    unfreeze();
}

/**
  Callback for the firmware download result
  */
void deviceWidget::downloadFinished(bool result)
{
    last_download_success = result;
    if(result)
    {
        status("Download successful", STATUSICON_OK);
        // Now save the result (use the utility function from OP_DFU)
        m_dfu->SaveByteArrayToFile(filename, downloadedFirmware);
    }
    else
            status("Download failed", STATUSICON_FAIL);
    myDevice->retrievePartBundleButton->setEnabled(true);
    emit downloadFinishedCallback();
}

/**
  Callback for the firmware upload result
  */
void deviceWidget::uploadFinished(OP_DFU::Status retstatus)
{
    last_upload_status = retstatus;

    if(retstatus != OP_DFU::Last_operation_Success) {
        status(QString("Upload failed with code: ") + m_dfu->StatusToString(retstatus).toLatin1().data(), STATUSICON_FAIL);
        emit uploadedFinishedCallback();
        return;
    }
    emit uploadedFinishedCallback();
    status("Upload successful", STATUSICON_OK);

}

/**
  Slot to update the progress bar
  */
void deviceWidget::setProgress(int percent)
{
    myDevice->progressBar->setValue(percent);
}

/**
 *Opens an open file dialog.
 */
QString deviceWidget::setOpenFileName()
{
    QFileDialog::Options options;
    QString selectedFilter;
    QString fwDirectoryStr;
    QDir fwDirectory;

    //Format filename for file chooser
#ifdef Q_OS_WIN
	fwDirectoryStr=QCoreApplication::applicationDirPath();
    fwDirectory=QDir(fwDirectoryStr);
	fwDirectory.cdUp();
	fwDirectory.cd("firmware");
	fwDirectoryStr=fwDirectory.absolutePath();
#elif defined Q_OS_LINUX
	fwDirectoryStr=QCoreApplication::applicationDirPath();
	fwDirectory=QDir(fwDirectoryStr);
    fwDirectory.cd("../../..");
    fwDirectoryStr=fwDirectory.absolutePath();
    fwDirectoryStr=fwDirectoryStr+"/fw_"+myDevice->lblBrdName->text().toLower()+"/fw_"+myDevice->lblBrdName->text().toLower()+".tlfw";
#elif defined Q_OS_MAC
    fwDirectoryStr=QCoreApplication::applicationDirPath();
    fwDirectory=QDir(fwDirectoryStr);
    fwDirectory.cd("../../../../../..");
    fwDirectoryStr=fwDirectory.absolutePath();
    fwDirectoryStr=fwDirectoryStr+"/fw_"+myDevice->lblBrdName->text().toLower()+"/fw_"+myDevice->lblBrdName->text().toLower()+".tlfw";
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
 *Set the save file name
 */
QString deviceWidget::setSaveFileName()
{
    QFileDialog::Options options;
    QString selectedFilter;
    QString fileName = QFileDialog::getSaveFileName(this,
                                                    tr("Select firmware file"),
                                                    "",
                                                    tr("Firmware Files (*.bin)"),
                                                    &selectedFilter,
                                                    options);
    return fileName;
}


void deviceWidget::readPartitions()
{
    freeze();
    QEventLoop loop;
    QByteArray array;
    connect(this, SIGNAL(downloadFinishedCallback()), &loop, SLOT(quit()));
    foreach(QTableWidgetItem *item,myDevice->tablePartitions->selectedItems())
    {
        if(item->column() == 0)
        {
            QString filename = QFileDialog::getSaveFileName(this,QString(tr("Select file to write the partition #%0 to")).arg(item->text()));
            if (!filename.isEmpty()) {
                status(QString("Downloading partition #%0 from device").arg(item->text()), STATUSICON_RUNNING);
                array.clear(); // Empty the byte array
                int partition_number = item->text().toInt();
                int partition_size = myDevice->tablePartitions->item(item->row(),1)->text().toInt();


                bool ret = m_dfu->DownloadPartition(&array,deviceID,partition_number,partition_size);
                if(!ret) {
                    status("Could not start download!", STATUSICON_FAIL);
                    continue;
                }
                status(QString("Partition #%0 download started, please wait").arg(item->text()), STATUSICON_RUNNING);
                loop.exec();
                if(last_download_success)
                    m_dfu->SaveByteArrayToFile(filename, array);
            }
        }
    }
    unfreeze();
}

void deviceWidget::wipePartitions()
{
    freeze();
    foreach(QTableWidgetItem *item,myDevice->tablePartitions->selectedItems())
    {
        if(item->column() == 0)
            m_dfu->WipePartition(item->text().toInt()); //currently no way to check success
    }
    unfreeze();
}

void deviceWidget::writeBundlePartitions()
{
    QString success;
    QString failed;
    freeze();
    QEventLoop loop;
    connect(this, SIGNAL(uploadedFinishedCallback()), &loop, SLOT(quit()));
    QString bundleDirName("taulabs_partitions");
    QDir bundleDir;
    bundleDir = QDir::temp();
    if(!bundleDir.mkdir(bundleDirName))
    {
        status(tr("Could not create temporary directory"),STATUSICON_FAIL);
        unfreeze();
        return;
    }
    bundleDir.cd(bundleDirName);
    QString filename = QFileDialog::getOpenFileName(this,QString(tr("Select file to read the partitions bundle file from")),QDir::homePath(), "Compressed file (*.zip)");
    if(!FileUtils::extractAll(filename,bundleDir.absolutePath()))
    {
        status(tr("Could not open compressed file"),STATUSICON_FAIL);
        unfreeze();
        FileUtils::removeDir(bundleDir.absolutePath());
        return;
    }
    foreach(QString file,bundleDir.entryList())
    {
        m_dfu->AbortOperation();
        bool ok;
        qDebug()<<QFileInfo(bundleDir.absolutePath()+QDir::separator()+file).baseName();
        int partition = QFileInfo(file).baseName().toInt(&ok);
        if(!ok)
        {
            status(tr("Could not parse filename"),STATUSICON_FAIL);
            continue;
        }
        QString filename = bundleDir.absolutePath()+QDir::separator()+QString::number(partition)+".bin";
        status(QString("Uploading partition #%0 to device").arg(QString::number(partition)), STATUSICON_RUNNING);
        if(partition > m_dfu->devices[deviceID].PartitionSizes.size())
        {
            if(!failed.isEmpty())
                failed.append(",");
            failed.append(QString::number(partition));
            status(QString(tr("Partition #%0 upload failed")).arg(partition),STATUSICON_FAIL);
            continue;//this should never hapen
        }
        bool ret = m_dfu->UploadPartition(filename,false,deviceID,partition,m_dfu->devices[deviceID].PartitionSizes.at(partition));
        if(!ret) {
            status("Could not start upload!", STATUSICON_FAIL);
            if(!failed.isEmpty())
                failed.append(",");
            failed.append(QString::number(partition));
            status(QString(tr("Partition #%0 upload failed")).arg(partition),STATUSICON_FAIL);
            continue;
        }
        loop.exec();
        if(last_upload_status == OP_DFU::Last_operation_Success)
        {
            if(!success.isEmpty())
                success.append(",");
            success.append(QString::number(partition));
            status(QString(tr("Partition #%0 successfully uploaded")).arg(partition),STATUSICON_OK);

        }
        else
        {
            if(!failed.isEmpty())
                failed.append(",");
            failed.append(QString::number(partition));
            status(QString(tr("Partition #%0 upload failed")).arg(partition),STATUSICON_FAIL);
        }
    }
    FileUtils::removeDir(bundleDir.absolutePath());
    unfreeze();
    QString result;
    if(!success.isEmpty())
        result.append(QString("Partition(s) %0 uploaded successfully").arg(success));
    if(!failed.isEmpty())
    {
        if(success.isEmpty())
            result.append("Partitions");
        else
            result.append(", and partition(s)");
        result.append(QString(" %0 failed to upload").arg(failed));
    }
    status(result, STATUSICON_INFO);
}

void deviceWidget::showHidePartBrowser(int value)
{
    if(value == Qt::Checked)
        myDevice->gbPartitionBrowser->setVisible(true);
    else
        myDevice->gbPartitionBrowser->setVisible(false);
}
void deviceWidget::writePartitions()
{
    freeze();
    QEventLoop loop;
    QByteArray array;
    QString success;
    QString failed;
    connect(this, SIGNAL(uploadedFinishedCallback()), &loop, SLOT(quit()));
    foreach(QTableWidgetItem *item,myDevice->tablePartitions->selectedItems())
    {
        if(item->column() == 0)
        {
            QString filename = QFileDialog::getOpenFileName(this,QString(tr("Select file to read the partition #%0 from")).arg(item->text()));
            if (!filename.isEmpty()) {
                status(QString("Uploading partition #%0 to device").arg(item->text()), STATUSICON_RUNNING);
                array.clear(); // Empty the byte array
                int partition_number = item->text().toInt();
                if(partition_number > m_dfu->devices[deviceID].PartitionSizes.size())
                {
                    if(!failed.isEmpty())
                        failed.append(",");
                    failed.append(QString::number(partition_number));
                    status(QString(tr("Partition #%0 upload failed")).arg(partition_number),STATUSICON_FAIL);
                    continue;//this should never hapen
                }
                bool ret = m_dfu->UploadPartition(filename,false,deviceID,partition_number,m_dfu->devices[deviceID].PartitionSizes.at(partition_number));
                if(!ret) {
                    status("Could not start upload!", STATUSICON_FAIL);
                    if(!failed.isEmpty())
                        failed.append(",");
                    failed.append(QString::number(partition_number));
                    status(QString(tr("Partition #%0 upload failed")).arg(partition_number),STATUSICON_FAIL);
                    continue;
                }
                status(QString(tr("Upload of partition #%0 started, please wait")).arg(partition_number), STATUSICON_RUNNING);
                loop.exec();
                if(last_upload_status == OP_DFU::Last_operation_Success)
                {
                    if(!success.isEmpty())
                        success.append(",");
                    success.append(QString::number(partition_number));
                    status(QString(tr("Partition #%0 successfully uploaded")).arg(partition_number),STATUSICON_OK);
                }
                else
                {
                    if(!failed.isEmpty())
                        failed.append(",");
                    failed.append(QString::number(partition_number));
                    status(QString(tr("Partition #%0 upload failed")).arg(partition_number),STATUSICON_FAIL);

                }
            }
        }
    }
    QString result;
    if(!success.isEmpty())
        result.append(QString("Partition(s) %0 uploaded successfully").arg(success));
    if(!failed.isEmpty())
    {
        if(success.isEmpty())
            result.append("Partitions");
        else
            result.append(", and partition(s)");
        result.append(QString(" %0 failed to upload").arg(failed));
    }
    unfreeze();
    status(result, STATUSICON_INFO);
}
