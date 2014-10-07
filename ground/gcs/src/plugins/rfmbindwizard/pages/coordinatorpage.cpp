/**
 ******************************************************************************
 *
 * @file       coordinatorpage.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup RfmBindWizard Setup Wizard
 * @{
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

#include "coordinatorpage.h"
#include "ui_coordinatorpage.h"
#include "rfmbindwizard.h"

#include <extensionsystem/pluginmanager.h>
#include <uavobjectutil/uavobjectutilmanager.h>

#include <QTimer>
#include <coreplugin/iboardtype.h>

CoordinatorPage::CoordinatorPage(RfmBindWizard *wizard, QWidget *parent) :
    AbstractWizardPage(wizard, parent), ui(new Ui::CoordinatorPage),
    m_coordinatorConfigured(false), m_boardType(NULL)
{
    ui->setupUi(this);
    ui->setCoordinator->setEnabled(false);

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

    connect(ui->setCoordinator, SIGNAL(clicked()), this, SLOT(configureCoordinator()));

    setupDeviceList();
}

CoordinatorPage::~CoordinatorPage()
{
    delete ui;
}

void CoordinatorPage::initializePage()
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objMngr = pm->getObject<UAVObjectManager>();
    QList <Core::IBoardType *> boards = pm->getObjects<Core::IBoardType>();

    // Store all the board hardware settings for probing
    boardPluginMap.clear();
    foreach (Core::IBoardType *board, boards) {
        if (board->queryCapabilities(Core::IBoardType::BOARD_CAPABILITIES_RADIO) ) {
            UAVObject *obj = objMngr->getObject(board->getHwUAVO());
            if (obj != NULL) {
                boardPluginMap.insert(obj, board);
                connect(obj, SIGNAL(transactionCompleted(UAVObject*,bool)),
                        this, SLOT(transactionReceived(UAVObject*,bool)),
                        Qt::UniqueConnection);
            }
        }
    }

    connect(&probeTimer, SIGNAL(timeout()), this, SLOT(probeRadio()));
    setControllerType(NULL);

    emit completeChanged();
}

bool CoordinatorPage::isComplete() const
{
    return m_coordinatorConfigured;
}

bool CoordinatorPage::validatePage()
{
    disconnect(this);
    return true;
}


/**
 * @brief ControllerPage::getControllerType get the interface for
 * the connected board
 * @return the IBoardType
 */
Core::IBoardType *CoordinatorPage::getControllerType() const
{
    return m_boardType;
}

//! Indicate the type of board detected and enable binding button
void CoordinatorPage::setControllerType(Core::IBoardType *board)
{
    m_boardType = board;
    if (board == NULL) {
        ui->boardTypeLabel->setText("Unknown");
        ui->setCoordinator->setEnabled(false);
    } else {
        ui->boardTypeLabel->setText(board->shortName());

        // Do not allow performing this for multiple boards
        if (!m_coordinatorConfigured)
            ui->setCoordinator->setEnabled(true);
    }
}

void CoordinatorPage::setupDeviceList()
{
    devicesChanged(m_connectionManager->getAvailableDevices());
    connectionStatusChanged();
}


void CoordinatorPage::devicesChanged(QLinkedList<Core::DevListItem> devices)
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

//! Called when a board is connected or disconnected
void CoordinatorPage::connectionStatusChanged()
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
        probeTimer.start();
        setControllerType(NULL);
    } else {
        ui->deviceCombo->setEnabled(true);
        ui->connectButton->setText(tr("Connect"));
        qDebug() << "Connection status changed: Disconnected";
        probeTimer.stop();
        setControllerType(NULL);
    }
    emit completeChanged();
}

//! Called when the connect/disconnect button is clicked
void CoordinatorPage::connectDisconnect()
{
    if (m_connectionManager->isConnected()) {
        m_connectionManager->disconnectDevice();
        probeTimer.stop();
        setControllerType(NULL);
    } else {
        m_connectionManager->connectDevice(m_connectionManager->findDevice(ui->deviceCombo->itemData(ui->deviceCombo->currentIndex(), Qt::ToolTipRole).toString()));
        probeTimer.start();
        setControllerType(NULL);
    }
    emit completeChanged();
}

bool CoordinatorPage::configureCoordinator()
{
    Core::IBoardType *board = getControllerType();
    if (board == NULL)
        return false;

    // Get the ID for this board and make it a coordinator
    quint32 rfmId = board->getRfmID();
    board->setCoordID(0);

    // Store the coordinator ID
    getWizard()->setCoordID(rfmId);

    m_coordinatorConfigured = true;

    qDebug() << "Coordinator ID: " << rfmId;

    return true;
}

void CoordinatorPage::probeRadio()
{
    qDebug() << "Testing if radio is attached";

    // Probe for each board by checking for the corresponding
    // hardware settings. This is inelegant but because the
    // modem does not establish a connection with the GCS, it
    // is necessary.
    QList <UAVObject*> objects = boardPluginMap.keys();
    foreach (UAVObject *obj,  objects) {
        qDebug() << "Probing " << obj->getName();
        obj->requestUpdate();
    }

}

void CoordinatorPage::transactionReceived(UAVObject *obj, bool success)
{
    qDebug() << obj->getName() << " " << success;
    if (success) {
        ui->boardTypeLabel->setText(obj->getName());
        setControllerType(boardPluginMap[obj]);
        probeTimer.stop();
    }
}
