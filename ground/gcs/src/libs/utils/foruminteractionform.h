/**
 ******************************************************************************
 * @file       foruminteractionform.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup [Utils
 * @{
 * @addtogroup ForumInteractionForm
 * @{
 * @brief Utility to present a form to the user where he can input is forum
 * credentials and aircraft details
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

#ifndef FORUMINTERACTIONFORM_H
#define FORUMINTERACTIONFORM_H

#include <QDialog>
#include "utils_global.h"

namespace Ui {
class ForumInteractionForm;
}

namespace Utils {

class QTCREATOR_UTILS_EXPORT ForumInteractionForm : public QDialog
{
    Q_OBJECT

public:
    explicit ForumInteractionForm(QWidget *parent = 0);
    ~ForumInteractionForm();
    void setPassword(QString value);
    void setUserName(QString value);
    QString getUserName();
    QString getPassword();
    void setObservations(QString value);
    void setAircraftDescription(QString value);
    QString getObservations();
    QString getAircraftDescription();
    bool getSaveCredentials();

private:
    Ui::ForumInteractionForm *ui;
};

}
#endif // FORUMINTERACTIONFORM_H
