/**
 ******************************************************************************
 * @file       taulinkgadgetwidget.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup TauLinkGadgetPlugin Tau Link Gadget Plugin
 * @{
 * @brief A gadget to monitor and configure the RFM22b link
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
#include "taulinkgadgetwidget.h"
#include "rfm22bstatus.h"
#include "ui_taulink.h"

TauLinkGadgetWidget::TauLinkGadgetWidget(QWidget *parent) : ConfigTaskWidget(parent)
{
    ui = new Ui_TauLink();
    ui->setupUi(this);

    // Connect to the LinkStatus object updates
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    rfm22bStatusObj = dynamic_cast<UAVDataObject*>(objManager->getObject("RFM22BStatus"));

    if (rfm22bStatusObj != NULL ) {
        connect(rfm22bStatusObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updateStatus(UAVObject*)));
        rfm22bStatusObj->requestUpdate();

        autoLoadWidgets();

        addUAVObjectToWidgetRelation("RFM22BStatus", "RxGood", ui->Good);
        addUAVObjectToWidgetRelation("RFM22BStatus", "RxCorrected", ui->Corrected);
        addUAVObjectToWidgetRelation("RFM22BStatus", "RxErrors", ui->Errors);
        addUAVObjectToWidgetRelation("RFM22BStatus", "RxMissed", ui->Missed);
        addUAVObjectToWidgetRelation("RFM22BStatus", "RxFailure", ui->RxFailure);
        addUAVObjectToWidgetRelation("RFM22BStatus", "UAVTalkErrors", ui->UAVTalkErrors);
        addUAVObjectToWidgetRelation("RFM22BStatus", "TxDropped", ui->Dropped);
        addUAVObjectToWidgetRelation("RFM22BStatus", "TxResent", ui->Resent);
        addUAVObjectToWidgetRelation("RFM22BStatus", "TxFailure", ui->TxFailure);
        addUAVObjectToWidgetRelation("RFM22BStatus", "Resets", ui->Resets);
        addUAVObjectToWidgetRelation("RFM22BStatus", "Timeouts", ui->Timeouts);
        addUAVObjectToWidgetRelation("RFM22BStatus", "RSSI", ui->RSSI);
        addUAVObjectToWidgetRelation("RFM22BStatus", "LinkQuality", ui->LinkQuality);
        addUAVObjectToWidgetRelation("RFM22BStatus", "RXSeq", ui->RXSeq);
        addUAVObjectToWidgetRelation("RFM22BStatus", "TXSeq", ui->TXSeq);
        addUAVObjectToWidgetRelation("RFM22BStatus", "RXRate", ui->RXRate);
        addUAVObjectToWidgetRelation("RFM22BStatus", "TXRate", ui->TXRate);
        addUAVObjectToWidgetRelation("RFM22BStatus", "LinkState", ui->LinkState);
    }

}

TauLinkGadgetWidget::~TauLinkGadgetWidget()
{
   // Do nothing
}

/*!
  \brief Called by updates to @RFM22BStatus
  */
void TauLinkGadgetWidget::updateStatus(UAVObject *object)
{
    // Update the detected devices.
    UAVObjectField* pairIdField = object->getField("PairIDs");
    if (pairIdField) {
        ui->PairID1->setText(QString::number(pairIdField->getValue(0).toUInt(), 16).toUpper());
        ui->PairID1->setEnabled(false);

        ui->PairID2->setText(QString::number(pairIdField->getValue(1).toUInt(), 16).toUpper());
        ui->PairID2->setEnabled(false);

        ui->PairID3->setText(QString::number(pairIdField->getValue(2).toUInt(), 16).toUpper());
        ui->PairID3->setEnabled(false);

        ui->PairID4->setText(QString::number(pairIdField->getValue(3).toUInt(), 16).toUpper());
        ui->PairID4->setEnabled(false);
    }

    UAVObjectField* pairRssiField = object->getField("PairSignalStrengths");
    if (pairRssiField) {
        ui->PairSignalStrengthBar1->setValue(pairRssiField->getValue(0).toInt());
        ui->PairSignalStrengthBar2->setValue(pairRssiField->getValue(1).toInt());
        ui->PairSignalStrengthBar3->setValue(pairRssiField->getValue(2).toInt());
        ui->PairSignalStrengthBar4->setValue(pairRssiField->getValue(3).toInt());
        ui->PairSignalStrengthLabel1->setText(QString("%1dB").arg(pairRssiField->getValue(0).toInt()));
        ui->PairSignalStrengthLabel2->setText(QString("%1dB").arg(pairRssiField->getValue(1).toInt()));
        ui->PairSignalStrengthLabel3->setText(QString("%1dB").arg(pairRssiField->getValue(2).toInt()));
        ui->PairSignalStrengthLabel4->setText(QString("%1dB").arg(pairRssiField->getValue(3).toInt()));
    }

    // Update the Description field
    UAVObjectField* descField = object->getField("Description");
    if (descField) {
        /*
         * This looks like a binary with a description at the end
         *  4 bytes: header: "TlFw"
         *  4 bytes: git commit hash (short version of SHA1)
         *  4 bytes: Unix timestamp of last git commit
         *  2 bytes: target platform. Should follow same rule as BOARD_TYPE and BOARD_REVISION in board define files.
         *  26 bytes: commit tag if it is there, otherwise "Unreleased". Zero-padded
         *   ---- 40 bytes limit ---
         *  20 bytes: SHA1 sum of the firmware.
         *  40 bytes: free for now.
         */
        char buf[RFM22BStatus::DESCRIPTION_NUMELEM];
        for (unsigned int i = 0; i < 26; ++i)
            buf[i] = descField->getValue(i + 14).toChar().toLatin1();
        buf[26] = '\0';
        QString descstr(buf);
        quint32 gitDate = descField->getValue(11).toChar().toLatin1() & 0xFF;
        for (int i = 1; i < 4; i++) {
            gitDate = gitDate << 8;
            gitDate += descField->getValue(11-i).toChar().toLatin1() & 0xFF;
        }
        QString date = QDateTime::fromTime_t(gitDate).toUTC().toString("yyyy-MM-dd HH:mm");
        ui->FirmwareVersion->setText(descstr + " " + date);
    }

    // Update the serial number field
    UAVObjectField* serialField = object->getField("CPUSerial");
    if (serialField) {
        char buf[RFM22BStatus::CPUSERIAL_NUMELEM * 2 + 1];
        for (unsigned int i = 0; i < RFM22BStatus::CPUSERIAL_NUMELEM; ++i)
        {
            unsigned char val = serialField->getValue(i).toUInt() >> 4;
            buf[i * 2] = ((val < 10) ? '0' : '7') + val;
            val = serialField->getValue(i).toUInt() & 0xf;
            buf[i * 2 + 1] = ((val < 10) ? '0' : '7') + val;
        }
        buf[RFM22BStatus::CPUSERIAL_NUMELEM * 2] = '\0';
        ui->SerialNumber->setText(buf);
    }

    // Update the DeviceID field
    UAVObjectField* idField = object->getField("DeviceID");
    if (idField) {
        ui->DeviceID->setText(QString::number(idField->getValue().toUInt(), 16).toUpper());
    }

    // Update the link state
    UAVObjectField* linkField = object->getField("LinkState");
    if (linkField) {
        ui->LinkState->setText(linkField->getValue().toString());
    }
}
/**
  * @}
  * @}
  */
