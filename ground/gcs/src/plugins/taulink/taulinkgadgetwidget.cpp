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
#include "tllinkstatus.h"
#include "ui_taulink.h"

TauLinkGadgetWidget::TauLinkGadgetWidget(QWidget *parent) : ConfigTaskWidget(parent)
{
    ui = new Ui_TauLink();
    ui->setupUi(this);

    autoLoadWidgets();

    addUAVObjectToWidgetRelation("TLLinkStatus", "RxGood", ui->Good);
    addUAVObjectToWidgetRelation("TLLinkStatus", "RxCorrected", ui->Corrected);
    addUAVObjectToWidgetRelation("TLLinkStatus", "RxErrors", ui->Errors);
    addUAVObjectToWidgetRelation("TLLinkStatus", "RxMissed", ui->Missed);
    addUAVObjectToWidgetRelation("TLLinkStatus", "RxFailure", ui->RxFailure);
    addUAVObjectToWidgetRelation("TLLinkStatus", "UAVTalkErrors", ui->UAVTalkErrors);
    addUAVObjectToWidgetRelation("TLLinkStatus", "TxDropped", ui->Dropped);
    addUAVObjectToWidgetRelation("TLLinkStatus", "TxResent", ui->Resent);
    addUAVObjectToWidgetRelation("TLLinkStatus", "TxFailure", ui->TxFailure);
    addUAVObjectToWidgetRelation("TLLinkStatus", "Resets", ui->Resets);
    addUAVObjectToWidgetRelation("TLLinkStatus", "Timeouts", ui->Timeouts);
    addUAVObjectToWidgetRelation("TLLinkStatus", "RSSI", ui->RSSI);
    addUAVObjectToWidgetRelation("TLLinkStatus", "LinkQuality", ui->LinkQuality);
    addUAVObjectToWidgetRelation("TLLinkStatus", "RXSeq", ui->RXSeq);
    addUAVObjectToWidgetRelation("TLLinkStatus", "TXSeq", ui->TXSeq);
    addUAVObjectToWidgetRelation("TLLinkStatus", "RXRate", ui->RXRate);
    addUAVObjectToWidgetRelation("TLLinkStatus", "TXRate", ui->TXRate);
}

TauLinkGadgetWidget::~TauLinkGadgetWidget()
{
   // Do nothing
}

/**
  * @}
  * @}
  */
