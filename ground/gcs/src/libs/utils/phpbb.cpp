/**
 ******************************************************************************
 * @file       phpbb.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup [Group]
 * @{
 * @addtogroup PHPBB
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

#include "phpbb.h"

#define TIMEOUT 10000

namespace Utils {

PHPBB::PHPBB(QString host, QObject *parent):QNetworkAccessManager(parent), host(host)
{
    new QNetworkCookieJar(this);
    request.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("application/x-www-form-urlencoded"));
    request.setRawHeader("User-Agent", "Firefox/3.0.10");
}


void PHPBB::addField(QString name, QString value) {
    fields.insert(name, value);
}

PHPBB::~PHPBB()
{
}

QNetworkReply *PHPBB::postData(QString url) {
    QString host;
    host = url.right(url.length()-url.indexOf("://")-3);
    host = host.left(host.indexOf("/"));
    QString crlf = "\r\n";
    qsrand(QDateTime::currentDateTime().toTime_t());
    QString b = QVariant(qrand()).toString()+QVariant(qrand()).toString()+QVariant(qrand()).toString();
    QString boundary="---------------------------"+b;
    QString endBoundary = crlf + "--" + boundary + "--"+crlf;
    QString contentType = "multipart/form-data; boundary=" + boundary;
    boundary="--" + boundary + crlf;
    QByteArray bond = boundary.toLatin1();
    QByteArray send;
    bool first = true;
    foreach (QString name, fields.keys()) {
        send.append(bond);
        if (first) {
            boundary = crlf + boundary;
            bond = boundary.toLatin1();
            first = false;
        }
        send.append(QString("Content-Disposition: form-data; name=\"" + name + "\"" + crlf).toLatin1());
        send.append(QString("Content-Transfer-Encoding: 8bit" + crlf).toLatin1());
        send.append(crlf.toLatin1());
        send.append(fields.value(name).toUtf8());
    }
    send.append(endBoundary.toLatin1());

    fields.clear();

    request.setRawHeader("Host", host.toLatin1());
    request.setHeader(QNetworkRequest::ContentTypeHeader, contentType.toLatin1());
    request.setHeader(QNetworkRequest::ContentLengthHeader, QVariant(send.size()).toString());
    request.setUrl(QUrl(url));
    qDebug()<<url;
    qDebug()<<send;
    return this->post(request, send);
}

bool PHPBB::postReply(int forumID, int threadID, QString subject, QString message)
{
    addField("post", "Submit");
    QEventLoop loop;
    QTimer::singleShot(TIMEOUT, &loop, SLOT(quit()));
    QNetworkReply *reply = postData(host + QString("/posting.php?mode=reply&f=%0&t=%1").arg(forumID).arg(threadID));
    connect(reply, SIGNAL(finished()), &loop, SLOT(quit()));
    loop.exec();
    QString secrets = reply->readAll();
    if( !(secrets.contains("creation_time") && secrets.contains("form_token")) )
        return false;
    int creation_index = secrets.indexOf("creation_time");
    creation_index = secrets.indexOf("value=", creation_index);
    int creation_index1 = secrets.indexOf('"',creation_index);
    int creation_index2 = secrets.indexOf('"', creation_index1 + 1);
    QString creationTime = secrets.mid(creation_index1 + 1, creation_index2 - creation_index1 -1 );
    creation_index = secrets.indexOf("form_token");
    creation_index = secrets.indexOf("value=", creation_index);
    creation_index1 = secrets.indexOf('"',creation_index);
    creation_index2 = secrets.indexOf('"', creation_index1 + 1);
    QString formToken = secrets.mid(creation_index1 + 1, creation_index2 - creation_index1 -1 );

    QTimer::singleShot(2000, &loop, SLOT(quit()));
    loop.exec();

    addField("form_token", formToken);
    addField("creation_time", creationTime);
    addField("subject", subject);
    addField("message", message);
    addField("post", "Submit"); // adds fields in name/value pairs    QEventLoop loop;

    QTimer::singleShot(TIMEOUT, &loop, SLOT(quit()));
    reply = postData(host + QString("/posting.php?mode=reply&f=%0&t=%1").arg(forumID).arg(threadID));
    connect(reply, SIGNAL(finished()), &loop, SLOT(quit()));
    loop.exec();
    if(reply->error() != QNetworkReply::NoError)
        return false;
    QString replyStr(reply->readAll());
    qDebug()<<replyStr;
    if(replyStr.contains("This message has been posted successfully"))
        return true;
    else
        return false;
}

bool PHPBB::login(QString username, QString password)
{
    addField("username", username);
    addField("password", password);
    addField("login", "");
    QEventLoop loop;
    QTimer::singleShot(TIMEOUT, &loop, SLOT(quit()));
    QNetworkReply *reply = postData(host + "/ucp.php?mode=login");
    //connect(this ,SIGNAL(finished(QNetworkReply *)),this,SLOT(readData(QNetworkReply *)),Qt::UniqueConnection);
    //return true;
    connect(reply, SIGNAL(finished()), &loop, SLOT(quit()));
    loop.exec();
    if(reply->error() != QNetworkReply::NoError)
        return false;
    QString replyStr(reply->readAll());
    qDebug()<<replyStr;
    if(replyStr.contains("You have been successfully logged in"))
        return true;
    else
        return false;
}

}
