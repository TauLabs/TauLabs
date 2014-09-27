/**
 ******************************************************************************
 * @file       forumcredentialsform.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup [Group]
 * @{
 * @addtogroup ForumCredentialsForm
 * @{
 * @brief [Brief]
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

#ifndef FORUMCREDENTIALSFORM_H
#define FORUMCREDENTIALSFORM_H

#include <QDialog>
#include "utils_global.h"

namespace Ui {
class ForumCredentialsForm;
}

namespace Utils {

class QTCREATOR_UTILS_EXPORT ForumCredentialsForm : public QDialog
{
    Q_OBJECT

public:
    explicit ForumCredentialsForm(QWidget *parent = 0);
    ~ForumCredentialsForm();
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
    Ui::ForumCredentialsForm *ui;
};

}
#endif // FORUMCREDENTIALSFORM_H
