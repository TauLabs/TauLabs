/**
 ******************************************************************************
 *
 * @file       controllerpage.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup SetupWizard Setup Wizard
 * @{
 * @brief
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

#include "controllerpage.h"
#include "ui_controllerpage.h"
#include "setupwizard.h"

#include <extensionsystem/pluginmanager.h>
#include <uavobjectutil/uavobjectutilmanager.h>

ControllerPage::ControllerPage(SetupWizard *wizard, QWidget *parent) :
    AbstractWizardPage(wizard, parent),
    ui(new Ui::ControllerPage)
{
    ui->setupUi(this);

    m_connectionManager = getWizard()->getConnectionManager();
    Q_ASSERT(m_connectionManager);
    connect(m_connectionManager, SIGNAL(availableDevicesChanged(QLinkedList<Core::DevListItem>)), this, SLOT(devicesChanged(QLinkedList<Core::DevListItem>)));

    ExtensionSystem::PluginManager *pluginManager = ExtensionSystem::PluginManager::instance();
    Q_ASSERT(pluginManager);
    m_telemtryManager = pluginManager->getObject<TelemetryManager>();
    Q_ASSERT(m_telemtryManager);
    connect(m_telemtryManager, SIGNAL(connected()), this, SLOT(connectionStatusChanged()));
    connect(m_telemtryManager, SIGNAL(disconnected()), this, SLOT(connectionStatusChanged()));

    connect(ui->connectButton, SIGNAL(clicked()), this, SLOT(connectDisconnect()));

    setupDeviceList();
}

ControllerPage::~ControllerPage()
{
    delete ui;
}

void ControllerPage::initializePage()
{
    if (anyControllerConnected()) {
        Core::IBoardType* type = getControllerType();
        setControllerType(type);
    } else {
        setControllerType(NULL);
    }
    emit completeChanged();
}

bool ControllerPage::isComplete() const
{
    return (getControllerType() != NULL) &&
           m_connectionManager->getCurrentDevice().getConName().startsWith("USB:", Qt::CaseInsensitive);
}

bool ControllerPage::validatePage()
{
    getWizard()->setControllerType(getControllerType());
    return true;
}

bool ControllerPage::anyControllerConnected()
{
    return m_telemtryManager->isConnected();
}

/**
 * @brief ControllerPage::getControllerType get the interface for
 * the connected board
 * @return the IBoardType
 */
Core::IBoardType *ControllerPage::getControllerType() const
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectUtilManager *utilMngr     = pm->getObject<UAVObjectUtilManager>();
    return utilMngr->getBoardType();
}

void ControllerPage::setupDeviceList()
{
    devicesChanged(m_connectionManager->getAvailableDevices());
    connectionStatusChanged();
}

void ControllerPage::setControllerType(Core::IBoardType *board)
{
    if (board == NULL)
        ui->boardTypeLabel->setText("Unknown");
    else
        ui->boardTypeLabel->setText(board->shortName());
}

void ControllerPage::devicesChanged(QLinkedList<Core::DevListItem> devices)
{
    // Get the selected item before the update if any
    QString currSelectedDeviceName = ui->deviceCombo->currentIndex() != -1 ?
                                     ui->deviceCombo->itemData(ui->deviceCombo->currentIndex(), Qt::ToolTipRole).toString() : "";

    // Clear the box
    ui->deviceCombo->clear();

    int indexOfSelectedItem = -1;
    int i = 0;

    // Loop and fill the combo with items from connectionmanager
    foreach(Core::DevListItem deviceItem, devices) {
        ui->deviceCombo->addItem(deviceItem.getConName());
        QString deviceName = (const QString)deviceItem.getConName();
        ui->deviceCombo->setItemData(ui->deviceCombo->count() - 1, deviceName, Qt::ToolTipRole);
        if (!deviceName.startsWith("USB:", Qt::CaseInsensitive)) {
            ui->deviceCombo->setItemData(ui->deviceCombo->count() - 1, QVariant(0), Qt::UserRole - 1);
        }
        if (currSelectedDeviceName != "" && currSelectedDeviceName == deviceName) {
            indexOfSelectedItem = i;
        }
        i++;
    }

    // Re select the item that was selected before if any
    if (indexOfSelectedItem != -1) {
        ui->deviceCombo->setCurrentIndex(indexOfSelectedItem);
    }
}

void ControllerPage::connectionStatusChanged()
{
    if (m_connectionManager->isConnected()) {
        ui->deviceCombo->setEnabled(false);
        ui->connectButton->setText(tr("Disconnect"));
        QString connectedDeviceName = m_connectionManager->getCurrentDevice().getConName();
        for (int i = 0; i < ui->deviceCombo->count(); ++i) {
            if (connectedDeviceName == ui->deviceCombo->itemData(i, Qt::ToolTipRole).toString()) {
                ui->deviceCombo->setCurrentIndex(i);
                break;
            }
        }

        setControllerType(getControllerType());
        qDebug() << "Connection status changed: Connected, controller type: " << getControllerType();
    } else {
        ui->deviceCombo->setEnabled(true);
        ui->connectButton->setText(tr("Connect"));
        setControllerType(NULL);
        qDebug() << "Connection status changed: Disconnected";
    }
    emit completeChanged();
}

void ControllerPage::connectDisconnect()
{
    if (m_connectionManager->isConnected()) {
        m_connectionManager->disconnectDevice();
    } else {
        m_connectionManager->connectDevice(m_connectionManager->findDevice(ui->deviceCombo->itemData(ui->deviceCombo->currentIndex(), Qt::ToolTipRole).toString()));
    }
    emit completeChanged();
}
