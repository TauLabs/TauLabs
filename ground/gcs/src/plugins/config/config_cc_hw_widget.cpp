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
#include <QtGui/QWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QPushButton>
#include <QDesktopServices>
#include <QUrl>
#include <extensionsystem/pluginmanager.h>
#include <coreplugin/generalsettings.h>


ConfigCCHWWidget::ConfigCCHWWidget(QWidget *parent) : ConfigTaskWidget(parent)
{
    m_telemetry = new Ui_CC_HW_Widget();
    m_telemetry->setupUi(this);

    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if(!settings->useExpertMode())
        m_telemetry->saveTelemetryToRAM->setVisible(false);


    UAVObjectUtilManager* utilMngr = pm->getObject<UAVObjectUtilManager>();
    int id = utilMngr->getBoardModel();

    switch (id) {
    case 0x0101:
        m_telemetry->label_2->setPixmap(QPixmap(":/uploader/images/deviceID-0101.svg"));
        break;
    case 0x0301:
        m_telemetry->label_2->setPixmap(QPixmap(":/uploader/images/deviceID-0301.svg"));
        break;
    case 0x0401:
        m_telemetry->label_2->setPixmap(QPixmap(":/configgadget/images/coptercontrol.svg"));
        break;
    case 0x0402:
        m_telemetry->label_2->setPixmap(QPixmap(":/configgadget/images/coptercontrol.svg"));
        break;
    case 0x0201:
        m_telemetry->label_2->setPixmap(QPixmap(":/uploader/images/deviceID-0201.svg"));
        break;
    default:
        m_telemetry->label_2->setPixmap(QPixmap(":/configgadget/images/coptercontrol.svg"));
        break;
    }
    addApplySaveButtons(m_telemetry->saveTelemetryToRAM,m_telemetry->saveTelemetryToSD);
    addUAVObjectToWidgetRelation("HwCopterControl","FlexiPort",m_telemetry->cbFlexi);
    addUAVObjectToWidgetRelation("HwCopterControl","MainPort",m_telemetry->cbTele);
    addUAVObjectToWidgetRelation("HwCopterControl","RcvrPort",m_telemetry->cbRcvr);
    addUAVObjectToWidgetRelation("HwCopterControl","USB_HIDPort",m_telemetry->cbUsbHid);
    addUAVObjectToWidgetRelation("HwCopterControl","USB_VCPPort",m_telemetry->cbUsbVcp);
    addUAVObjectToWidgetRelation("ModuleSettings","TelemetrySpeed",m_telemetry->telemetrySpeed);
    addUAVObjectToWidgetRelation("ModuleSettings","GPSSpeed",m_telemetry->gpsSpeed);
    addUAVObjectToWidgetRelation("ModuleSettings","ComUsbBridgeSpeed",m_telemetry->comUsbBridgeSpeed);

    // Load UAVObjects to widget relations from UI file
    // using objrelation dynamic property
    autoLoadWidgets();

    connect(m_telemetry->cchwHelp,SIGNAL(clicked()),this,SLOT(openHelp()));
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
    if (((m_telemetry->cbTele->currentIndex() == HwCopterControl::MAINPORT_DEBUGCONSOLE) &&
	 (m_telemetry->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_DEBUGCONSOLE)) ||
	((m_telemetry->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_DEBUGCONSOLE) &&
	 (m_telemetry->cbUsbVcp->currentIndex() == HwCopterControl::USB_VCPPORT_DEBUGCONSOLE)) ||
	((m_telemetry->cbUsbVcp->currentIndex() == HwCopterControl::USB_VCPPORT_DEBUGCONSOLE) &&
	 (m_telemetry->cbTele->currentIndex() == HwCopterControl::MAINPORT_DEBUGCONSOLE)))
    {
        enableControls(false);
        m_telemetry->problems->setText(tr("Warning: you have configured more than one DebugConsole, this currently is not supported"));
    }
    else if (((m_telemetry->cbTele->currentIndex() == HwCopterControl::MAINPORT_TELEMETRY) && (m_telemetry->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_TELEMETRY)) ||
        ((m_telemetry->cbTele->currentIndex() == HwCopterControl::MAINPORT_GPS) && (m_telemetry->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_GPS)) ||
        ((m_telemetry->cbTele->currentIndex() == HwCopterControl::MAINPORT_DEBUGCONSOLE) && (m_telemetry->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_DEBUGCONSOLE)) ||
        ((m_telemetry->cbTele->currentIndex() == HwCopterControl::MAINPORT_COMBRIDGE) && (m_telemetry->cbFlexi->currentIndex() == HwCopterControl::FLEXIPORT_COMBRIDGE)))
    {
        enableControls(false);
        m_telemetry->problems->setText(tr("Warning: you have configured both MainPort and FlexiPort for the same function, this currently is not supported"));
    }
    else if ((m_telemetry->cbUsbHid->currentIndex() == HwCopterControl::USB_HIDPORT_USBTELEMETRY) && (m_telemetry->cbUsbVcp->currentIndex() == HwCopterControl::USB_VCPPORT_USBTELEMETRY))
    {
        enableControls(false);
        m_telemetry->problems->setText(tr("Warning: you have configured both USB HID Port and USB VCP Port for the same function, this currently is not supported"));
    }
    else if ((m_telemetry->cbUsbHid->currentIndex() != HwCopterControl::USB_HIDPORT_USBTELEMETRY) && (m_telemetry->cbUsbVcp->currentIndex() != HwCopterControl::USB_VCPPORT_USBTELEMETRY))
    {
        enableControls(false);
        m_telemetry->problems->setText(tr("Warning: you have disabled USB Telemetry on both USB HID Port and USB VCP Port, this currently is not supported"));
    }
    else
    {
        m_telemetry->problems->setText("");
        enableControls(true);
    }
}

void ConfigCCHWWidget::openHelp()
{
    QDesktopServices::openUrl( QUrl("http://wiki.openpilot.org/x/D4AUAQ", QUrl::StrictMode) );
}


/**
 * Gets a variant from a widget
 * @param widget pointer to the widget from where to get the value
 * @param scale scale to be used on the assignement
 * @return returns the value of the widget times the scale
 */
QVariant ConfigCCHWWidget::getVariantFromWidget(QWidget * widget, double scale)
{
    if(QGroupBox * groupBox=qobject_cast<QGroupBox *>(widget))
    {
        QString ret;
        if(groupBox->property("TrueString").isValid() && groupBox->property("FalseString").isValid()){
            if(groupBox->isChecked())
                ret = groupBox->property("TrueString").toString();
            else
                ret = groupBox->property("FalseString").toString();
        }
        else{
            if(groupBox->isChecked())
                ret = "TRUE";
            else
                ret = "FALSE";
        }

        return ret;
    }
    else if(QCheckBox * checkBox=qobject_cast<QCheckBox *>(widget))
    {
        QString ret;
        if(checkBox->property("TrueString").isValid() && checkBox->property("FalseString").isValid()){
            if(checkBox->isChecked())
                ret = checkBox->property("TrueString").toString();
            else
                ret = checkBox->property("FalseString").toString();
        }
        else{
            if(checkBox->isChecked())
                ret = "TRUE";
            else
                ret = "FALSE";
        }

        return ret;
    }
    else
    {
        return ConfigTaskWidget::getVariantFromWidget(widget, scale);
    }
}


/**
 * Sets a widget from a variant
 * @param widget pointer for the widget to set
 * @param scale scale to be used on the assignement
 * @param value value to be used on the assignement
 * @return returns true if the assignement was successfull
 */
bool ConfigCCHWWidget::setWidgetFromVariant(QWidget *widget, QVariant value, double scale)
{

    if(QGroupBox * groupBox=qobject_cast<QGroupBox *>(widget))
        {
            bool bvalue;
            if(groupBox->property("TrueString").isValid() && groupBox->property("FalseString").isValid()){
                bvalue = value.toString()==groupBox->property("TrueString").toString();
            }
            else{
                bvalue = value.toString()=="TRUE";
            }
            groupBox->setChecked(bvalue);
            return true;
        }
        else if(QCheckBox * checkBox=qobject_cast<QCheckBox *>(widget))
        {
            bool bvalue;
            if(checkBox->property("TrueString").isValid() && checkBox->property("FalseString").isValid()){
                bvalue = value.toString()==checkBox->property("TrueString").toString();
            }
            else{
                bvalue = value.toString()=="TRUE";
            }
            checkBox->setChecked(bvalue);
            return true;
        }
    else
    {
        return ConfigTaskWidget::setWidgetFromVariant(widget, value, scale);
    }
}

/**
  * @}
  * @}
  */
