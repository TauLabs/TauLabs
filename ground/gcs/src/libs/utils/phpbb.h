/**
 ******************************************************************************
 * @file       phpbb.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup Utils
 * @{
 * @addtogroup PHPBB
 * @{
 * @brief Utility class to comunicate with a phpbb3 forum
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

#ifndef PHPBB_H
#define PHPBB_H

#include <QtNetwork/QNetworkAccessManager>
#include <QtNetwork/QNetworkRequest>
#include <QtNetwork/QNetworkReply>
#include <QtNetwork/QNetworkCookie>
#include <QtNetwork/QNetworkCookieJar>
#include <QMap>
#include <QTimer>
#include <QEventLoop>
#include "utils_global.h"
#include <QFile>

namespace Utils {

class QTCREATOR_UTILS_EXPORT PHPBB: public QNetworkAccessManager
{
    Q_OBJECT
public:
    struct forumPost
    {
        QString author;
        QString link;
        QString title;
        QString text;
    };
    struct fileAttachment
    {
        QString fileName;
        QString fileComment;
        QString fileTypeSpec;
        QByteArray fileData;
    };
    struct replyFields
    {
        QString fieldName;
        QString fieldValue;
        fileAttachment attachment;
    };

    PHPBB(QString host, QObject * parent = 0);
    QNetworkReply *postData(QString url);
    bool postReply(int forumID, int threadID, QString subject, QString message, QList<fileAttachment> attachments = QList<fileAttachment>());
    QList<forumPost> getAllPosts(int forumID, int threadID);
    bool login(QString username, QString password);
    void addField(QString name, QString value, fileAttachment attachment = fileAttachment());
    ~PHPBB();
private:
    QList<replyFields> fields;
    QNetworkRequest request;
    QString host;
};

}
#endif // PHPBB_H
