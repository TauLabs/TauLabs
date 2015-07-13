/**
 ******************************************************************************
 *
 * @file       radioprobepage.h
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

#ifndef RADIOPROBEPAGE_H
#define RADIOPROBEPAGE_H

#include <QMap>
#include <coreplugin/iboardtype.h>
#include <coreplugin/icore.h>
#include <coreplugin/connectionmanager.h>
#include <uavobject.h>

#include "abstractwizardpage.h"
#include "rfmbindwizard.h"

class RadioProbePage : public AbstractWizardPage
{
    Q_OBJECT
public:
    explicit RadioProbePage(RfmBindWizard *wizard, QWidget *parent = 0);

private:
    //! Set the board device for probed radio board
    void setBoardType(Core::IBoardType *);

    Core::IBoardType *m_boardType;
    Core::ConnectionManager *m_connectionManager;
    QMap <UAVObject*, Core::IBoardType*> boardPluginMap;
    QTimer probeTimer;
    bool m_allowProbing;

protected:
    //! Get the board device for probed radio board
    Core::IBoardType* getBoardType() const;

    //! Stop probing once configured
    void stopProbing();

private slots:
    //! Start or stop probing when boards are added
    void connectionStatusChanged();

    //! Probe if a radio is plugged in
    void probeRadio();

    //! Receive if a hardware object is updated
    void transactionReceived(UAVObject*obj, bool success);

signals:
    //! Indicate probing of a board has changed
    void probeChanged(bool);

    //! Indicate connection changed
    void connectionChanged(QString);
};

#endif // RADIOPROBEPAGE_H
