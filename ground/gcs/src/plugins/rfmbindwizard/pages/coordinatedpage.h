/**
 ******************************************************************************
 *
 * @file       coordinatedpage.h
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

#ifndef COORDINATEDPAGE_H
#define COORDINATEDPAGE_H

#include <QMap>
#include <coreplugin/iboardtype.h>
#include <coreplugin/icore.h>
#include <coreplugin/connectionmanager.h>
#include "rfmbindwizard.h"
#include "radioprobepage.h"

#include <uavobject.h>

namespace Ui {
class CoordinatedPage;
}

class CoordinatedPage : public RadioProbePage {
    Q_OBJECT

public:
    explicit CoordinatedPage(RfmBindWizard *wizard, QWidget *parent = 0);
    ~CoordinatedPage();
    void initializePage();
    bool isComplete() const;
    bool validatePage();

private:
    Ui::CoordinatedPage *ui;
    bool m_coordinatorConfigured;

private slots:
    //! Update UI to indicate either a board was probed or not
    void updateProbe(bool);

    //! Bind this board to the coordinator
    bool bindCoordinator();
};

#endif // COORDINATEDPAGE_H
