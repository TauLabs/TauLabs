/**
 ******************************************************************************
 *
 * @file       globalmessaging.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief The Core GCS plugin
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

#ifndef GLOBALMESSAGING_H
#define GLOBALMESSAGING_H
#include "core_global.h"
#include <QObject>
#include <QString>

namespace Core {

typedef enum {ERROR,WARNING,INFO} MessageType;
class CORE_EXPORT GlobalMessage : public QObject
{
    Q_OBJECT
public:
    explicit GlobalMessage(QString brief,QString description,MessageType type,QObject *parent = 0);
    explicit GlobalMessage(MessageType=ERROR,QObject *parent = 0);
    void setActive(bool value);
    bool isActive(){return m_active;}
    void setText(QString brief,QString description){m_brief=brief; m_description=description; emit changed(this);}
    void setBrief(QString brief){m_brief=brief; emit changed(this);}
    void setDescription(QString description){m_description=description; emit changed(this);}
    QString getBrief(){return m_brief;}
    QString getDescription(){return m_description;}
    MessageType getType(){return m_type;}
signals:
    void changed(GlobalMessage *);
private:
    QString m_brief;
    QString m_description;
    MessageType m_type;
    bool m_active;
};

class CORE_EXPORT GlobalMessaging : public QObject
{
    Q_OBJECT
public:
    explicit GlobalMessaging(QObject *parent = 0);
    GlobalMessage * addErrorMessage(QString brief,QString description);
    GlobalMessage * addWarningMessage(QString brief,QString description);
    GlobalMessage * addInfoMessage(QString brief,QString description);
    void addMessage(GlobalMessage*);
    QList<GlobalMessage*> getActiveErrors();
    QList<GlobalMessage*> getActiveWarnings();
    QList<GlobalMessage*> getActiveInfos();
    QList<GlobalMessage*> getErrors();
    QList<GlobalMessage*> getWarnings();
    QList<GlobalMessage*> getInfos();

signals:
    void newMessage(GlobalMessage*);
    void newError(GlobalMessage*);
    void newWarning(GlobalMessage*);
    void newInfo(GlobalMessage*);
    void changedMessage(GlobalMessage*);
    void changedError(GlobalMessage*);
    void changedWarning(GlobalMessage*);
    void changedInfo(GlobalMessage*);
    void deletedMessage();
    void deletedError();
    void deletedWarning();
    void deletedInfo();

public slots:
    void messageDeleted();
private:
    QList<GlobalMessage*> errorList;
    QList<GlobalMessage*> warningList;
    QList<GlobalMessage*> infoList;
};
}
#endif // GLOBALMESSAGING_H
