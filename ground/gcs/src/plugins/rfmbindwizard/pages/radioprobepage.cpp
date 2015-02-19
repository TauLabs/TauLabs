/**
 ******************************************************************************
 *
 * @file       radioprobepage.cpp
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

#include "radioprobepage.h"

#include <QTimer>

#include <extensionsystem/pluginmanager.h>
#include <uavobjectutil/uavobjectutilmanager.h>

RadioProbePage::RadioProbePage(RfmBindWizard *wizard, QWidget *parent) :
    AbstractWizardPage(wizard, parent), m_boardType(NULL), m_allowProbing(true)
{
    m_connectionManager = getWizard()->getConnectionManager();
    Q_ASSERT(m_connectionManager);
    connect(m_connectionManager, SIGNAL(deviceConnected(QIODevice*)), this, SLOT(connectionStatusChanged()));
    connect(m_connectionManager, SIGNAL(deviceDisconnected()), this, SLOT(connectionStatusChanged()));

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

    setBoardType(NULL);

    probeTimer.setInterval(500);
    connect(&probeTimer, SIGNAL(timeout()), this, SLOT(probeRadio()));

    if (m_connectionManager->isConnected())
        probeTimer.start();
}

/**
 * @brief RadioProbePage::getControllerType get the interface for
 * the connected board
 * @return the IBoardType
 */
Core::IBoardType *RadioProbePage::getBoardType() const
{
    return m_boardType;
}

//! Indicate the type of board detected and enable binding button
void RadioProbePage::setBoardType(Core::IBoardType *board)
{
    m_boardType = board;
    if (board == NULL) {
        emit probeChanged(false);
    } else {
        emit probeChanged(true);
    }
}

//! Called when a board is connected or disconnected
void RadioProbePage::connectionStatusChanged()
{
    if (m_connectionManager->isConnected() && m_allowProbing) {
        probeTimer.start();
        setBoardType(NULL);
        emit probeChanged(false);
    } else {
        probeTimer.stop();
        setBoardType(NULL);
        emit probeChanged(false);
    }
}

//! Disallow further probing
void RadioProbePage::stopProbing()
{
    m_allowProbing = false;
}

//! Probe for any boards configure objects that support radio
void RadioProbePage::probeRadio()
{
    // Probe for each board by checking for the corresponding
    // hardware settings. This is inelegant but because the
    // modem does not establish a connection with the GCS, it
    // is necessary.
    QList <UAVObject*> objects = boardPluginMap.keys();
    foreach (UAVObject *obj,  objects) {
        obj->requestUpdate();
    }

}

//! Called whenever a radio board object is received
void RadioProbePage::transactionReceived(UAVObject *obj, bool success)
{
    if (success) {
        setBoardType(boardPluginMap[obj]);
        probeTimer.stop();
        emit probeChanged(true);
    }
}
