/**
 ******************************************************************************
 *
 * @file       globalmessaging.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
 *
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

#include "globalmessaging.h"

namespace Core {

GlobalMessage::GlobalMessage(QString brief, QString description, MessageType type, QObject *parent):QObject(parent),
    m_brief(brief),
    m_description(description),
    m_type(type),m_active(true)
{
}

GlobalMessage::GlobalMessage(MessageType type, QObject *parent):QObject(parent),m_type(type),m_active(true)
{

}

void GlobalMessage::setActive(bool value)
{
    m_active=value;
    emit changed(this);
}

GlobalMessage *GlobalMessaging::addErrorMessage(QString brief, QString description)
{
    GlobalMessage * message=new GlobalMessage(brief,description,ERROR,this);
    connect(message,SIGNAL(destroyed()),this,SLOT(messageDeleted()));
    connect(message,SIGNAL(changed(GlobalMessage*)),this,SIGNAL(changedMessage(GlobalMessage*)));
    connect(message,SIGNAL(changed(GlobalMessage*)),this,SIGNAL(changedError(GlobalMessage*)));
    errorList.append(message);
    emit newMessage(message);
    emit newError(message);
    return message;
}

GlobalMessage *GlobalMessaging::addWarningMessage(QString brief, QString description)
{
    GlobalMessage * message=new GlobalMessage(brief,description,WARNING,this);
    connect(message,SIGNAL(destroyed()),this,SLOT(messageDeleted()));
    connect(message,SIGNAL(changed(GlobalMessage*)),this,SIGNAL(changedMessage(GlobalMessage*)));
    connect(message,SIGNAL(changed(GlobalMessage*)),this,SIGNAL(changedWarning(GlobalMessage*)));
    warningList.append(message);
    emit newMessage(message);
    emit newWarning(message);
    return message;
}

GlobalMessage *GlobalMessaging::addInfoMessage(QString brief, QString description)
{
    GlobalMessage * message=new GlobalMessage(brief,description,INFO,this);
    connect(message,SIGNAL(destroyed()),this,SLOT(messageDeleted()));
    connect(message,SIGNAL(changed(GlobalMessage*)),this,SIGNAL(changedMessage(GlobalMessage*)));
    connect(message,SIGNAL(changed(GlobalMessage*)),this,SIGNAL(changedInfo(GlobalMessage*)));
    infoList.append(message);
    emit newMessage(message);
    emit newInfo(message);
    return message;
}

void GlobalMessaging::addMessage(GlobalMessage * message)
{
    connect(message,SIGNAL(destroyed()),this,SLOT(messageDeleted()));
    connect(message,SIGNAL(changed(GlobalMessage*)),this,SIGNAL(changedMessage(GlobalMessage*)));
    connect(message,SIGNAL(changed(GlobalMessage*)),this,SIGNAL(changedInfo(GlobalMessage*)));
    switch(message->getType())
    {
    case ERROR:
        errorList.append(message);
        emit newError(message);
        break;
    case WARNING:
        warningList.append(message);
        emit newWarning(message);
        break;
    case INFO:
        infoList.append(message);
        emit newInfo(message);
        break;
    default:
        break;
    }
    emit newMessage(message);
}

QList<GlobalMessage *> GlobalMessaging::getActiveErrors()
{
    QList<GlobalMessage *> temp;
    foreach(GlobalMessage * message,errorList)
    {
        if(message->isActive())
            temp.append(message);
    }
    return temp;
}

QList<GlobalMessage *> GlobalMessaging::getActiveWarnings()
{
    QList<GlobalMessage *> temp;
    foreach(GlobalMessage * message,warningList)
    {
        if(message->isActive())
            temp.append(message);
    }
    return temp;
}

QList<GlobalMessage *> GlobalMessaging::getActiveInfos()
{
    QList<GlobalMessage *> temp;
    foreach(GlobalMessage * message,infoList)
    {
        if(message->isActive())
            temp.append(message);
    }
    return temp;
}

QList<GlobalMessage *> GlobalMessaging::getErrors()
{
    return errorList;
}
QList<GlobalMessage *> GlobalMessaging::getWarnings()
{
    return warningList;
}
QList<GlobalMessage *> GlobalMessaging::getInfos()
{
    return infoList;
}

void GlobalMessaging::messageDeleted()
{
    GlobalMessage * message=dynamic_cast<GlobalMessage *>(sender());
    if(!message)
        return;
    switch(message->getType())
    {
    case ERROR:
        errorList.removeAll(message);
        emit deletedError();
        break;
    case WARNING:
        warningList.removeAll(message);
        emit deletedWarning();
        break;
    case INFO:
        infoList.removeAll(message);
        emit deletedInfo();
        break;
    default:
        break;
    }
    emit deletedMessage();
}

GlobalMessaging::GlobalMessaging(QObject *parent):QObject(parent)
{
}

}
