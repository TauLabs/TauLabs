/**
 ******************************************************************************
 *
 * @file       configtelemetrywidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief The Configuration Gadget used to update settings in the firmware
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
#include "config_cc_hw_widget.h"
#include "hwcoptercontrol.h"

#include <QDebug>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <QDesktopServices>
#include <QUrl>
#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>


ConfigCCHWWidget::ConfigCCHWWidget(QWidget *parent) : ConfigTaskWidget(parent)
{
    m_CC_HW_Widget = new Ui_CC_HW_Widget();
    m_CC_HW_Widget->setupUi(this);

    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if(!settings->useExpertMode())
        m_CC_HW_Widget->saveTelemetryToRAM->setVisible(false);


    UAVObjectUtilManager* utilMngr = pm->getObject<UAVObjectUtilManager>();
    int id = utilMngr->getBoardModel();

    switch (id) {
    case 0x0101:
        m_CC_HW_Widget->label_2->setPixmap(QPixmap(":/uploader/images/deviceID-0101.svg"));
        break;
    case 0x0301:
        m_CC_HW_Widget->label_2->setPixmap(QPixmap(":/uploader/images/deviceID-0301.svg"));
        break;
    case 0x0401:
        m_CC_HW_Widget->label_2->setPixmap(QPixmap(":/configgadget/images/coptercontrol.svg"));
        break;
    case 0x0402:
        m_CC_HW_Widget->label_2->setPixmap(QPixmap(":/configgadget/images/coptercontrol.svg"));
        break;
    case 0x0201:
        m_CC_HW_Widget->label_2->setPixmap(QPixmap(":/uploader/images/deviceID-0201.svg"));
        break;
    default:
        m_CC_HW_Widget->label_2->setPixmap(QPixmap(":/configgadget/images/coptercontrol.svg"));
        break;
    }
    addApplySaveButtons(m_CC_HW_Widget->saveTelemetryToRAM,m_CC_HW_Widget->saveTelemetryToSD);
    addUAVObjectToWidgetRelation("HwCopterControl","FlexiPort",m_CC_HW_Widget->cbFlexi);
    addUAVObjectToWidgetRelation("HwCopterControl","MainPort",m_CC_HW_Widget->cbTele);
    addUAVObjectToWidgetRelation("HwCopterControl","RcvrPort",m_CC_HW_Widget->cbRcvr);
    addUAVObjectToWidgetRelation("HwCopterControl","USB_HIDPort",m_CC_HW_Widget->cbUsbHid);
    addUAVObjectToWidgetRelation("HwCopterControl","USB_VCPPort",m_CC_HW_Widget->cbUsbVcp);
    addUAVObjectToWidgetRelation("ModuleSettings","TelemetrySpeed",m_CC_HW_Widget->telemetrySpeed);
    addUAVObjectToWidgetRelation("ModuleSettings","GPSSpeed",m_CC_HW_Widget->gpsSpeed);
    addUAVObjectToWidgetRelation("ModuleSettings","ComUsbBridgeSpeed",m_CC_HW_Widget->comUsbBridgeSpeed);

    // Load UAVObjects to widget relations from UI file
    // using objrelation dynamic property
    autoLoadWidgets();

    connect(m_CC_HW_Widget->cchwHelp,SIGNAL(clicked()),this,SLOT(openHelp()));
    enableControls(false);
    populateWidgets();
    refreshWidgetsValues();
    forceConnectedState();
}

ConfigCCHWWidget::~ConfigCCHWWidget()
{
    // Do nothing
}

void ConfigCCHWWidget::refreshValues()
{
}

void ConfigCCHWWidget::widgetsContentsChanged()
{
    ConfigTaskWidget::widgetsContentsChanged();
    if (((m_CC_HW_Widget->cbTele->currentIndex() == HwCopterControl::MAINPORT_DEBUGCONSOLE) &&
	 (m_CC_HW_Widget->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_DEBUGCONSOLE)) ||
	((m_CC_HW_Widget->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_DEBUGCONSOLE) &&
	 (m_CC_HW_Widget->cbUsbVcp->currentIndex() == HwCopterControl::USB_VCPPORT_DEBUGCONSOLE)) ||
	((m_CC_HW_Widget->cbUsbVcp->currentIndex() == HwCopterControl::USB_VCPPORT_DEBUGCONSOLE) &&
	 (m_CC_HW_Widget->cbTele->currentIndex() == HwCopterControl::MAINPORT_DEBUGCONSOLE)))
    {
        enableControls(false);
        m_CC_HW_Widget->problems->setText(tr("Warning: you have configured more than one DebugConsole, this currently is not supported"));
    }
    else if (((m_CC_HW_Widget->cbTele->currentIndex() == HwCopterControl::MAINPORT_TELEMETRY) && (m_CC_HW_Widget->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_TELEMETRY)) ||
        ((m_CC_HW_Widget->cbTele->currentIndex() == HwCopterControl::MAINPORT_GPS) && (m_CC_HW_Widget->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_GPS)) ||
        ((m_CC_HW_Widget->cbTele->currentIndex() == HwCopterControl::MAINPORT_DEBUGCONSOLE) && (m_CC_HW_Widget->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_DEBUGCONSOLE)) ||
        ((m_CC_HW_Widget->cbTele->currentIndex() == HwCopterControl::MAINPORT_COMBRIDGE) && (m_CC_HW_Widget->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_COMBRIDGE)))
    {
        enableControls(false);
        m_CC_HW_Widget->problems->setText(tr("Warning: you have configured both MainPort and FlexiPort for the same function, this currently is not supported"));
    }
    else if ((m_CC_HW_Widget->cbUsbHid->currentIndex() == HwCopterControl::USB_HIDPORT_USBTELEMETRY) && (m_CC_HW_Widget->cbUsbVcp->currentIndex() == HwCopterControl::USB_VCPPORT_USBTELEMETRY))
    {
        enableControls(false);
        m_CC_HW_Widget->problems->setText(tr("Warning: you have configured both USB HID Port and USB VCP Port for the same function, this currently is not supported"));
    }
    else if ((m_CC_HW_Widget->cbUsbHid->currentIndex() != HwCopterControl::USB_HIDPORT_USBTELEMETRY) && (m_CC_HW_Widget->cbUsbVcp->currentIndex() != HwCopterControl::USB_VCPPORT_USBTELEMETRY))
    {
        enableControls(false);
        m_CC_HW_Widget->problems->setText(tr("Warning: you have disabled USB Telemetry on both USB HID Port and USB VCP Port, this currently is not supported"));
    }
    else
    {
        m_CC_HW_Widget->problems->setText("");
        enableControls(true);
    }
}

void ConfigCCHWWidget::openHelp()
{
    QDesktopServices::openUrl( QUrl("http://wiki.taulabs.org/OnlineHelp:-Hardware-Settings", QUrl::StrictMode) );
}

/**
  * @}
  * @}
  */
