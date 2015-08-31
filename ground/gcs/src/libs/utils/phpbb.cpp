/**
 ******************************************************************************
 * @file       phpbb.cpp
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

#include "phpbb.h"

#define TIMEOUT 60000

namespace Utils {

PHPBB::PHPBB(QString host, QObject *parent):QNetworkAccessManager(parent), host(host)
{
    new QNetworkCookieJar(this);
    request.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("application/x-www-form-urlencoded"));
    request.setRawHeader("User-Agent", "Firefox/3.0.10");
}


void PHPBB::addField(QString name, QString value, fileAttachment attachment) {
    replyFields reply;
    reply.fieldName = name;
    reply.fieldValue = value;
    reply.attachment = attachment;
    fields.append(reply);
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
    foreach (replyFields field, fields) {
        send.append(bond);
        if (first) {
            boundary = crlf + boundary;
            bond = boundary.toLatin1();
            first = false;
        }
        if(field.attachment.fileName.isEmpty()) {
            send.append(QString("Content-Disposition: form-data; name=\"" + field.fieldName + "\"" + crlf).toLatin1());
            send.append(QString("Content-Transfer-Encoding: 8bit" + crlf).toLatin1());
            send.append(crlf.toLatin1());
            send.append(field.fieldValue.toUtf8());
        }
        else {
            send.append(QString("Content-Disposition: form-data; name=\"fileupload\"; filename=\"" + field.attachment.fileName + "\"" + crlf).toLatin1());
            send.append(QString("Content-Type: " + field.attachment.fileTypeSpec + crlf).toLatin1());
            send.append(crlf.toLatin1());
            send.append(field.attachment.fileData);
            send.append(bond);
            send.append(QString("Content-Disposition: form-data; name=\"filecomment\"" + crlf).toLatin1());
            send.append(crlf.toLatin1());
            send.append(field.attachment.fileComment.toLatin1());
        }
    }
    send.append(endBoundary.toLatin1());

    fields.clear();
    request.setRawHeader("Host", host.toLatin1());
    request.setHeader(QNetworkRequest::ContentTypeHeader, contentType.toLatin1());
    request.setHeader(QNetworkRequest::ContentLengthHeader, QVariant(send.size()).toString());
    request.setUrl(QUrl(url));
    return this->post(request, send);
}

bool PHPBB::postReply(int forumID, int threadID, QString subject, QString message, QList<fileAttachment> attachments)
{
    addField("post", "Submit");
    QEventLoop loop;
    QTimer::singleShot(TIMEOUT, &loop, SLOT(quit()));
    QNetworkReply *reply = postData(host + QString("/posting.php?mode=reply&f=%0&t=%1").arg(forumID).arg(threadID));
    connect(reply, SIGNAL(finished()), &loop, SLOT(quit()));
    loop.exec();
    QString secrets = reply->readAll();
    if ( !(secrets.contains("creation_time") && secrets.contains("form_token")) )
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
    foreach (fileAttachment attach, attachments) {
        addField("", "", attach);
    }
    addField("post", "Submit");

    QTimer::singleShot(TIMEOUT, &loop, SLOT(quit()));
    reply = postData(host + QString("/posting.php?mode=reply&f=%0&t=%1").arg(forumID).arg(threadID));
    connect(reply, SIGNAL(finished()), &loop, SLOT(quit()));
    loop.exec();
    if (reply->error() != QNetworkReply::NoError)
        return false;
    QString replyStr(reply->readAll());
    if (replyStr.contains("This message has been posted successfully"))
        return true;
    else
        return false;
}

QList<PHPBB::forumPost> PHPBB::getAllPosts(int forumID, int threadID)
{
    addField("post", "Submit");
    QEventLoop loop;
    QList<PHPBB::forumPost> list;
    int current_page = 0;
    int total_of_pages = 0;
    while(true){
        QTimer::singleShot(1000, &loop, SLOT(quit()));
        loop.exec();
        int cindex = 0;
        QString pageStr;
        if (current_page != 0) {
            pageStr = "&start=" + QString::number(current_page * 10);
        }
        QString url = host + QString("/viewtopic.php?f=%0&t=%1%2").arg(forumID).arg(threadID).arg(pageStr);
        QTimer::singleShot(TIMEOUT, &loop, SLOT(quit()));
        QNetworkReply *reply = this->postData(url);
        connect(reply, SIGNAL(finished()), &loop, SLOT(quit()));
        loop.exec();
        QString str = reply->readAll();
        total_of_pages = str.count("<span class=\"page-sep\">, </span>");
        while(true) {
            PHPBB::forumPost post;
            int start_index = str.indexOf("<div class=\"postbody\">", cindex);
            if (start_index == -1) {
                break;
            }
            start_index = str.indexOf("<a href", start_index);
            start_index = str.indexOf("\">", start_index);
            int end_index = str.indexOf("</a>", start_index);
            cindex = end_index;
            post.title = str.mid(start_index + 2, end_index - start_index - 2);
            start_index = str.indexOf("class=\"author\"", end_index);
            start_index = str.indexOf("href=\"", start_index);
            end_index = str.indexOf("\">", start_index);
            post.link = host + str.mid(start_index + 6 + 1, end_index -start_index - 7);
            start_index = str.indexOf("<strong>", start_index);
            start_index = str.indexOf("\">", start_index);
            end_index = str.indexOf("</a>", start_index);
            post.author = str.mid(start_index + 2, end_index -start_index - 2);
            start_index = str.indexOf("<div class=\"content\">", start_index);
            end_index = str.indexOf("</div>", start_index);
            post.text = str.mid(start_index + QString("<div class=\"content\">").length(), end_index - start_index - QString("<div class=\"content\">").length());
            list.append(post);
        }
        ++current_page;
        if(current_page >= total_of_pages)
            break;
    }
    return list;
}

bool PHPBB::login(QString username, QString password)
{
    addField("username", username);
    addField("password", password);
    addField("login", "");
    QEventLoop loop;
    QTimer::singleShot(TIMEOUT, &loop, SLOT(quit()));
    QNetworkReply *reply = postData(host + "/ucp.php?mode=login");
    connect(reply, SIGNAL(finished()), &loop, SLOT(quit()));
    loop.exec();
    if (reply->error() != QNetworkReply::NoError)
        return false;
    QString replyStr(reply->readAll());
    if (replyStr.contains("You have been successfully logged in"))
        return true;
    else
        return false;
}
}
