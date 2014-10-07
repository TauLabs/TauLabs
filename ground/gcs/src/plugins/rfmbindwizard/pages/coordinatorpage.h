/**
 ******************************************************************************
 *
 * @file       coordinatorpage.h
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

#ifndef COORDINATORPAGE_H
#define COORDINATORPAGE_H

#include <QMap>
#include <coreplugin/iboardtype.h>
#include <coreplugin/icore.h>
#include <coreplugin/connectionmanager.h>
#include "rfmbindwizard.h"
#include "uavtalk/telemetrymanager.h"
#include "abstractwizardpage.h"

namespace Ui {
class CoordinatorPage;
}

class CoordinatorPage : public AbstractWizardPage {
    Q_OBJECT

public:
    explicit CoordinatorPage(RfmBindWizard *wizard, QWidget *parent = 0);
    ~CoordinatorPage();
    void initializePage();
    bool isComplete() const;
    bool validatePage();

private:
    Ui::CoordinatorPage *ui;
    Core::IBoardType* getControllerType() const;
    void setupDeviceList();
    void setControllerType(Core::IBoardType *);
    Core::ConnectionManager *m_connectionManager;
    TelemetryManager *m_telemtryManager;

    bool m_coordinatorConfigured;
    Core::IBoardType *m_boardType;

    QMap <UAVObject*, Core::IBoardType*> boardPluginMap;
    QTimer probeTimer;

private slots:
    void devicesChanged(QLinkedList<Core::DevListItem> devices);
    void connectionStatusChanged();
    void connectDisconnect();

    //! Configure this board as the coordinator
    bool configureCoordinator();

    //! Probe if a radio is plugged in
    void probeRadio();

    //! Receive if a hardware object is updated
    void transactionReceived(UAVObject*obj, bool success);
};

#endif // COORDINATORPAGE_H
