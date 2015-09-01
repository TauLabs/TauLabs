/**
 ******************************************************************************
 * @file
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @addtogroup gitHubAPI
 * @{
 * @addtogroup
 * @{
 * @brief Library to handle gitHub releases using it's API
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

#include "githubreleaseapi.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#define NETWORK_TIMEOUT 250000

gitHubReleaseAPI::gitHubReleaseAPI(QObject *parent) : QObject(parent)
{
    connect(&m_WebCtrl, SIGNAL(finished(QNetworkReply*)), &eventLoop,
            SLOT(quit()));
    connect(&timeOutTimer, SIGNAL(timeout()), &eventLoop,
            SLOT(quit()));
    timeOutTimer.setInterval(NETWORK_TIMEOUT);
    timeOutTimer.setSingleShot(true);
    connect(this, SIGNAL(logInfo(QString)), SLOT(onLogInfo(QString)));
    connect(this, SIGNAL(logError(QString)), SLOT(onLogError(QString)));
}

void gitHubReleaseAPI::setRepo(QString owner, QString repo)
{
    m_url = QString("https://api.github.com/repos/%0/%1/").arg(owner).arg(repo);
}

void gitHubReleaseAPI::setRepo(QString url)
{
    m_url = url;
    if(!url.endsWith("/"))
        m_url = m_url + "/";
}

QString gitHubReleaseAPI::getRepo()
{
    return m_url;
}

QHash<int, gitHubReleaseAPI::release> gitHubReleaseAPI::getReleases()
{
    QHash<int, gitHubReleaseAPI::release> ret;
    QString workUrl = QString("%0%1").arg(m_url).arg("releases");
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.get(request);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while getting releases");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return ret;
    }
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &error);
    if(error.error != QJsonParseError::NoError) {
        emit logError("Parsing failed:" + error.errorString());
        setLastError(PARSING_ERROR);
        return ret;
    }
    emit logInfo("Parsing succeeded");
    QJsonArray array = doc.array();
    foreach (QJsonValue value, array) {
        release rel = processRelease(value.toObject());
        ret.insert(rel.id, rel);
    }
    reply->deleteLater();
    setLastError(NO_ERROR);
    if(ret.values().count() == 0)
        setLastError(UNDEFINED_ERROR);
    emit logInfo("Releases fetching finished");
    return ret;
}

gitHubReleaseAPI::release gitHubReleaseAPI::getSingleRelease(int id)
{
    gitHubReleaseAPI::release ret;
    QString workUrl = QString("%0%1/%2").arg(m_url).arg("releases").arg(id);
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.get(request);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while getting release");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return ret;
    }
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &error);
    if(error.error != QJsonParseError::NoError) {
        emit logError("Parsing failed:" + error.errorString());
        setLastError(PARSING_ERROR);
        return ret;
    }
    setLastError(NO_ERROR);
    emit logInfo("Release fetching finished");
    ret = processRelease(doc.object());
    return ret;
}

QByteArray gitHubReleaseAPI::downloadAsset(int id)
{
    QByteArray ret;
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg("assets").arg(id);
    QNetworkRequest request;
    request.setUrl(workUrl);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.get(request);
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while getting release");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return ret;
    }
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &error);
    if(error.error != QJsonParseError::NoError) {
        emit logError("Parsing failed:" + error.errorString());
        setLastError(PARSING_ERROR);
        return ret;
    }
    GitHubAsset asset = processAsset(doc.object());
    request.setUrl(asset.browser_download_url);
    timeOutTimer.start();
    reply = m_WebCtrl.get(request);
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while getting release asset");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        return ret;
    }
    int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    qDebug() << statusCode;
    if(statusCode == 301 || statusCode == 302) {
        emit logInfo("Received HTML redirect header");
        QUrl redirectUrl = reply->attribute(QNetworkRequest::RedirectionTargetAttribute).toUrl();
        request.setUrl(redirectUrl);
        timeOutTimer.start();
        reply = m_WebCtrl.get(request);
        currentNetworkReply = reply;
        connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
        eventLoop.exec();
        if(reply->error() !=QNetworkReply::NoError) {
            emit logError("Network error:" + reply->errorString());
            return ret;
        }
    }
    ret = reply->readAll();
    setLastError(NO_ERROR);
    emit logInfo("Asset fetching finished");
    return ret;
}

gitHubReleaseAPI::GitHubAsset gitHubReleaseAPI::getAsset(int id)
{
    gitHubReleaseAPI::GitHubAsset ret;
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg("assets").arg(id);
    QNetworkRequest request;
    request.setUrl(workUrl);
    request.setRawHeader("Content-Type", "application/octet-stream");
    handleCredentials(request);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.get(request);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while getting asset");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return ret;
    }
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &error);
    if(error.error != QJsonParseError::NoError) {
        emit logError("Parsing failed:" + error.errorString());
        setLastError(PARSING_ERROR);
        return ret;
    }
    ret = processAsset(doc.object());
    setLastError(NO_ERROR);
    emit logInfo("Asset fetching finished");
    return ret;
}

QList<gitHubReleaseAPI::GitHubAsset> gitHubReleaseAPI::getReleaseAssets(int id)
{
    QList <gitHubReleaseAPI::GitHubAsset> ret;
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg(id).arg("assets");
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.get(request);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while getting release assets");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return ret;
    }
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &error);
    if(error.error != QJsonParseError::NoError) {
        emit logError("Parsing failed:" + error.errorString());
        setLastError(PARSING_ERROR);
        return ret;
    }
    foreach (QJsonValue obj, doc.array()) {
        ret.append(processAsset(obj.toObject()));
    }
    setLastError(NO_ERROR);
    emit logInfo("Release assets fetching finished");
    return ret;
}

bool gitHubReleaseAPI::uploadReleaseAsset(QString filename, QString label, int releaseID)
{
    if(releaseID < 0)
        return false;
    QString uploadURL = getSingleRelease(releaseID).upload_url;
    QFile file(filename);
    if(!file.open(QIODevice::ReadOnly)) {
        emit logError("Could not open file");
        return false;
    }
    QByteArray data = file.readAll();
    QFileInfo info(filename);
    QString fname = info.fileName();
    uploadURL.replace("{?name}", QString("?name=%0&label=%1").arg(fname).arg(label));

    QNetworkRequest request;
    request.setUrl(uploadURL);
    request.setRawHeader("Content-Type",QMimeDatabase().mimeTypeForFile(filename).name().toLocal8Bit());
    handleCredentials(request);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.post(request, data);
    connect(reply, SIGNAL(uploadProgress(qint64,qint64)), SIGNAL(uploadProgress(qint64,qint64)));
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while uploading release asset");
        setLastError(TIMEOUT_ERROR);
        return false;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return false;
    }
    setLastError(NO_ERROR);
    emit logInfo("Asset uploading finished");
    return true;
}

gitHubReleaseAPI::release gitHubReleaseAPI::getReleaseByTagName(QString name)
{
    gitHubReleaseAPI::release ret;
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg("tags").arg(name);
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.get(request);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while getting release");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return ret;
    }
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &error);
    if(error.error != QJsonParseError::NoError) {
        emit logError("Parsing failed:" + error.errorString());
        setLastError(PARSING_ERROR);
        return ret;
    }
    setLastError(NO_ERROR);
    emit logInfo("Release fetching finished");
    ret = processRelease(doc.object());
    return ret;
}

gitHubReleaseAPI::release gitHubReleaseAPI::createRelease(gitHubReleaseAPI::newGitHubRelease release)
{
    gitHubReleaseAPI::release ret;
    QString workUrl = QString("%0%1").arg(m_url).arg("releases");
    QNetworkRequest request;
    request.setRawHeader("Content-Type", "application/x-www-form-urlencoded");
    request.setUrl(workUrl);
    handleCredentials(request);
    QJsonObject obj;
    obj.insert("tag_name", release.tag_name);
    obj.insert("target_commitish", release.target_commitish);
    obj.insert("name", release.name);
    obj.insert("body", release.body);
    obj.insert("draft", release.draft);
    obj.insert("prerelease", release.prerelease);
    QJsonDocument data(obj);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.post(request, data.toJson());
    connect(reply, SIGNAL(uploadProgress(qint64,qint64)), SIGNAL(uploadProgress(qint64,qint64)));
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while creating release");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return ret;
    }
    QJsonParseError error;
    QByteArray array = reply->readAll();
    QJsonDocument doc = QJsonDocument::fromJson(array, &error);
    if(error.error != QJsonParseError::NoError) {
        emit logError("Parsing failed:" + error.errorString());
        setLastError(PARSING_ERROR);
        return ret;
    }
    setLastError(NO_ERROR);
    emit logInfo("Release creation finished");
    ret = processRelease(doc.object());
    return ret;
}

gitHubReleaseAPI::release gitHubReleaseAPI::editRelease(int id, gitHubReleaseAPI::newGitHubRelease release)
{
    gitHubReleaseAPI::release ret;
    if(id < 0)
        return ret;
    QString workUrl = QString("%0%1/%2").arg(m_url).arg("releases").arg(id);
    QNetworkRequest request;
    request.setRawHeader("Content-Type", "application/x-www-form-urlencoded");
    request.setUrl(workUrl);
    handleCredentials(request);
    QJsonObject obj;
    obj.insert("tag_name", release.tag_name);
    obj.insert("target_commitish", release.target_commitish);
    obj.insert("name", release.name);
    obj.insert("body", release.body);
    obj.insert("draft", release.draft);
    obj.insert("prerelease", release.prerelease);

    QJsonDocument data(obj);
    QByteArray array = data.toJson();
    QBuffer buffer(&array);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.sendCustomRequest(request, "PATCH", &buffer);
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while editing release");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return ret;
    }
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &error);
    if(error.error != QJsonParseError::NoError) {
        emit logError("Parsing failed:" + error.errorString());
        setLastError(PARSING_ERROR);
        return ret;
    }
    setLastError(NO_ERROR);
    emit logInfo("Release editing finished");
    ret = processRelease(doc.object());
    return ret;
}

gitHubReleaseAPI::GitHubAsset gitHubReleaseAPI::editAsset(int id, QString filename, QString newLabel)
{
    gitHubReleaseAPI::GitHubAsset ret;
    if(id < 0)
        return ret;
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg("assets").arg(id);
    QNetworkRequest request;
    request.setRawHeader("Content-Type", "application/x-www-form-urlencoded");
    request.setUrl(workUrl);
    handleCredentials(request);
    QJsonObject obj;
    obj.insert("name", filename);
    obj.insert("label", newLabel);

    QJsonDocument data(obj);
    QByteArray array = data.toJson();
    QBuffer buffer(&array);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.sendCustomRequest(request, "PATCH", &buffer);
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while editing asset");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return ret;
    }
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &error);
    if(error.error != QJsonParseError::NoError) {
        emit logError("Parsing failed:" + error.errorString());
        setLastError(PARSING_ERROR);
        return ret;
    }
    ret = processAsset(doc.object());
    setLastError(NO_ERROR);
    emit logInfo("Asset editing finished");
    return ret;
}

bool gitHubReleaseAPI::deleteAsset(int id)
{
    if(id < 0)
        return false;
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg("assets").arg(id);
    QNetworkRequest request;
    request.setRawHeader("Content-Type", "application/x-www-form-urlencoded");
    request.setUrl(workUrl);
    handleCredentials(request);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.deleteResource(request);
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while deleting asset");
        setLastError(TIMEOUT_ERROR);
        return false;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return false;
    }
    setLastError(NO_ERROR);
    emit logInfo("Asset deletion finished");
    return true;
}

bool gitHubReleaseAPI::deleteRelease(int id)
{
    if(id < 0)
        return false;
    QString workUrl = QString("%0%1/%2").arg(m_url).arg("releases").arg(id);
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.deleteResource(request);
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while deleting release");
        setLastError(TIMEOUT_ERROR);
        return false;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return false;
    }
    setLastError(NO_ERROR);
    emit logInfo("Release deletion finished");
    return true;
}

gitHubReleaseAPI::release gitHubReleaseAPI::getLatestRelease()
{
    gitHubReleaseAPI::release ret;
    QString workUrl = QString("%0%1/%2").arg(m_url).arg("releases").arg("latest");
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    timeOutTimer.start();
    QNetworkReply *reply = m_WebCtrl.get(request);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
    eventLoop.exec();
    if(!timeOutTimer.isActive()) {
        emit logError("Timeout while getting release");
        setLastError(TIMEOUT_ERROR);
        return ret;
    }
    timeOutTimer.stop();
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError("Network error:" + reply->errorString());
        setLastError(NETWORK_ERROR);
        return ret;
    }
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &error);
    if(error.error != QJsonParseError::NoError) {
        emit logError("Parsing failed:" + error.errorString());
        setLastError(PARSING_ERROR);
        return ret;
    }
    ret = processRelease(doc.object());
    setLastError(NO_ERROR);
    emit logInfo("Release fetching finished");
    return ret;
}

gitHubReleaseAPI::errors gitHubReleaseAPI::getLastError() const
{
    return lastError;
}

void gitHubReleaseAPI::setLastError(const gitHubReleaseAPI::errors &value)
{
    lastError = value;
}

void gitHubReleaseAPI::onLogInfo(QString str)
{
    qDebug() << "INFO:" << str;
}

void gitHubReleaseAPI::onLogError(QString str)
{
    qDebug() << "ERROR:" << str;
}

void gitHubReleaseAPI::abortOperation()
{
    if(currentNetworkReply.isNull())
        return;
    currentNetworkReply.data()->abort();
}

gitHubReleaseAPI::release gitHubReleaseAPI::processRelease(QJsonObject obj)
{
    release rel;
    rel.url = QUrl(obj.value("url").toString());
    rel.assets_url = QUrl(obj.value("assets_url").toString());
    rel.upload_url = obj.value("upload_url").toString();
    rel.html_url = QUrl(obj.value("html_url").toString());
    rel.body = obj.value("body").toString();
    rel.id = obj.value("id").toInt();
    rel.tag_name = obj.value("tag_name").toString();
    rel.target_commitish = obj.value("target_commitish").toString();
    rel.name = obj.value("name").toString();
    rel.draft = obj.value("draft").toBool();
    rel.author = processUser(obj.value("author").toObject());
    rel.prerelease = obj.value("prerelease").toBool();
    rel.created_at = QDateTime::fromString(obj.value("created_at").toString(), "yyyy-MM-ddThh:mm:ssZ");
    rel.published_at = QDateTime::fromString(obj.value("published_at").toString(), "yyyy-MM-ddThh:mm:ssZ");
    foreach (QJsonValue asset, obj.value("assets").toArray()) {
        GitHubAsset a = processAsset(asset.toObject());
        rel.assets.insert(a.id, a);
    }
    rel.tarball_url = obj.value("tarball_url").toString();
    rel.zipball_url = obj.value("zipball_url").toString();
    return rel;
}

gitHubReleaseAPI::GitHubAsset gitHubReleaseAPI::processAsset(QJsonObject obj)
{
    GitHubAsset asset;
    asset.url = obj.value("url").toString();
    asset.id = obj.value("id").toInt();
    asset.name = obj.value("name").toString();
    asset.label = obj.value("label").toString();
    asset.uploader = processUser(obj.value("uploader").toObject());
    asset.content_type = obj.value("content_type").toString();
    asset.state = obj.value("state").toString();
    asset.size = obj.value("size").toInt();
    asset.download_count = obj.value("download_count").toInt();
    asset.created_at = QDateTime::fromString(obj.value("created_at").toString(), "yyyy-MM-ddThh:mm:ssZ");
    asset.updated_at = QDateTime::fromString(obj.value("updated_at").toString(), "yyyy-MM-ddThh:mm:ssZ");
    asset.browser_download_url = obj.value("browser_download_url").toString();
    return asset;
}

gitHubReleaseAPI::GitHubUser gitHubReleaseAPI::processUser(QJsonObject obj)
{
    GitHubUser user;
    user.login = obj.value("login").toString();
    user.id = obj.value("id").toInt();
    user.avatar_url = obj.value("avatar_url").toString();
    user.gravatar_id = obj.value("gravatar_id").toString();
    user.url = obj.value("url").toString();
    user.html_url = obj.value("html_url").toString();
    user.followers_url = obj.value("followers_url").toString();
    user.following_url = obj.value("following_url").toString();
    user.gists_url = obj.value("gists_url").toString();
    user.starred_url = obj.value("starred_url").toString();
    user.subscriptions_url = obj.value("subscriptions_url").toString();
    user.organizations_url = obj.value("organizations_url").toString();
    user.repos_url = obj.value("repos_url").toString();
    user.events_url = obj.value("events_url").toString();
    user.received_events_url = obj.value("received_events_url").toString();
    user.type = obj.value("type").toString();
    user.site_admin = obj.value("site_admin").toBool();
    return user;
}

void gitHubReleaseAPI::handleCredentials(QNetworkRequest &request)
{
    if(!m_username.isEmpty() && !m_password.isEmpty()) {
        QString concatenated = m_username + ":" + m_password;
        QByteArray data = concatenated.toLocal8Bit().toBase64();
        QString headerData = "Basic " + data;
        request.setRawHeader("Authorization", headerData.toLocal8Bit());
    }
}

void gitHubReleaseAPI::setCredentials(QString username, QString password)
{
    m_username = username;
    m_password = password;
    qDebug() << username << password;
}
