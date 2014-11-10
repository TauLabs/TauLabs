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
        addUAVObjectToWidgetRelation("RFM22BStatus", "RxFailure", ui->RxFailure);
        addUAVObjectToWidgetRelation("RFM22BStatus", "Resets", ui->Resets);
        addUAVObjectToWidgetRelation("RFM22BStatus", "Timeouts", ui->Timeouts);
        addUAVObjectToWidgetRelation("RFM22BStatus", "RSSI", ui->RSSI);
        addUAVObjectToWidgetRelation("RFM22BStatus", "LinkQuality", ui->LinkQuality);
        addUAVObjectToWidgetRelation("RFM22BStatus", "RXRate", ui->RXRate);
        addUAVObjectToWidgetRelation("RFM22BStatus", "TXRate", ui->TXRate);
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
