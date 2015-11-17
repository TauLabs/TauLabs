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
#include <QApplication>

#define NETWORK_TIMEOUT 250000
#define CURRENT_STEP "current_step"
#define NEXT_STEP "next_step"
#define CUSTOM_CONTEXT "custom_context"
#define DATA_CONTAINER "data_container"

/**
 * @brief Class constucter
 * @param parent pointer to parent object
 */
gitHubReleaseAPI::gitHubReleaseAPI(QObject *parent) : QObject(parent)
{
    connect(this, SIGNAL(logInfo(QString)), SLOT(onLogInfo(QString)));
    connect(this, SIGNAL(logError(QString)), SLOT(onLogError(QString)));
    connect(&m_WebCtrl, SIGNAL(finished(QNetworkReply*)), this, SLOT(onFinishedNetworkAction(QNetworkReply*)));
}

/**
 * @brief Sets the gitHub repository to be used by the class instance
 * @param owner repository owner
 * @param repo  the name of the repository to use
 */
void gitHubReleaseAPI::setRepo(QString owner, QString repo)
{
    m_url = QString("https://api.github.com/repos/%0/%1/").arg(owner).arg(repo);
}

/**
 * @brief Sets the gitHub repository to be used by the class instance
 * @param url url of the gitHub repository
 */
void gitHubReleaseAPI::setRepo(QString url)
{
    m_url = url;
    if(!url.endsWith("/"))
        m_url = m_url + "/";
}

/**
 * @brief Gets the gitHub repository used by the class instance
 * @return the url of the current repository
 */
QString gitHubReleaseAPI::getRepo()
{
    return m_url;
}

/**
 * @brief Gets the list of releases of the present repository
 * This function returns immediately
 * @param context context of the request
 */
void gitHubReleaseAPI::getReleases(QVariant context)
{
    QString workUrl = QString("%0%1").arg(m_url).arg("releases");
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    QNetworkReply *reply = m_WebCtrl.get(request);
    reply->setProperty(CUSTOM_CONTEXT, context);
    reply->setProperty(CURRENT_STEP, FETCHING_ALL_RELEASES);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
}

/**
 * @brief Gets a single release structure from the repository
 * This function returns immediately
 * @param releaseID the ID of the release to retrieve
 * @param context context of the request
 * Emits releaseDownloaded(...) when done
 */
void gitHubReleaseAPI::getSingleRelease(int releaseID, QVariant context)
{
    newAssetFile assetFile;
    assetFile.releaseID = -1;
    getSingleRelease(releaseID, context, assetFile);
}

void gitHubReleaseAPI::getSingleRelease(int id, QVariant context, newAssetFile assetFile)
{
    QString workUrl = QString("%0%1/%2").arg(m_url).arg("releases").arg(id);
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    QNetworkReply *reply = m_WebCtrl.get(request);
    reply->setProperty(CUSTOM_CONTEXT, context);
    reply->setProperty(CURRENT_STEP, FETCHING_RELEASE);
    if(assetFile.releaseID != -1) {
        QVariant data;
        data.setValue(assetFile);
        reply->setProperty(DATA_CONTAINER, data);
        reply->setProperty(CURRENT_STEP, UPLOADING_ASSET_FILE);
    }
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
}

/**
 * @brief Downloads a file asset from a release
 * This function returns immediately
 * @param assetID the asset ID of the file to download
 * @param context context of the request
 * Emits fileDownloaded(...) when done
 */
void gitHubReleaseAPI::downloadAssetFile(int assetID, QVariant context)
{
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg("assets").arg(assetID);
    QNetworkRequest request;
    request.setUrl(workUrl);
    QNetworkReply *reply = m_WebCtrl.get(request);
    reply->setProperty(CURRENT_STEP, FETCHING_ASSET_FILE_URL);
    reply->setProperty(CUSTOM_CONTEXT, context);
}

/**
 * @brief Called by the QNetworkManager finished() signal after a download or upload action
 * @param reply handle to the particular network operation reply
 */
void gitHubReleaseAPI::onFinishedNetworkAction(QNetworkReply *reply)
{
    if (reply->property(CURRENT_STEP) != FETCHING_ASSET_FILE)
        reply->deleteLater();
    setLastError(NO_ERROR);
    bool networkError = false;
    if(reply->property(CURRENT_STEP).isNull())
        return;
    if(reply->error() !=QNetworkReply::NoError) {
        emit logError(tr("Network error:") + reply->errorString());
        networkError = true;
        setLastError(NETWORK_ERROR);
    }
    QVariant context = reply->property(CUSTOM_CONTEXT);
    if(reply->property(CURRENT_STEP) == FETCHING_ASSET_FILE_URL) {
        if(networkError) {
            emit fileDownloaded(NULL, NETWORK_ERROR, context);
            reply->deleteLater();
            return;
        }
        QList<gitHubReleaseAPI::GitHubAsset> assets;
        errors error;
        assets = processAssetFromNetworkReply(reply, error);
        if(error != NO_ERROR) {
            setLastError(error);
            emit fileDownloaded(NULL, error, context);
            reply->deleteLater();
            return;
        }
        QNetworkRequest request;
        request.setUrl(assets.first().browser_download_url);
        reply = m_WebCtrl.get(request);
        currentNetworkReply = reply;
        reply->setProperty(CUSTOM_CONTEXT, context);
        connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
        reply->setProperty(CURRENT_STEP, FETCHING_ASSET_FILE);
    }
    else if(reply->property(CURRENT_STEP) == FETCHING_ASSET_INFO) {
        if(networkError) {
            GitHubAsset asset;
            asset.id = -1;
            emit finishedAssetInfoDownload(asset, NETWORK_ERROR, context);
            return;
        }
        QList<gitHubReleaseAPI::GitHubAsset> assets;
        errors error;
        assets = processAssetFromNetworkReply(reply, error);
        if(error != NO_ERROR) {
            GitHubAsset asset;
            asset.id = -1;
            setLastError(error);
            emit finishedAssetInfoDownload(asset, error, context);
            return;
        }
        emit finishedAssetInfoDownload(assets.first(), NO_ERROR, context);
        return;
    }
    else if(reply->property(CURRENT_STEP) == FETCHING_ALL_ASSETS_INFO) {
        QList<gitHubReleaseAPI::GitHubAsset> assets;
        if(networkError) {
            emit finishedReleaseAssetsInfoDownload(assets, NETWORK_ERROR, context);
            return;
        }
        errors error;
        assets = processAssetFromNetworkReply(reply, error);
        if(error != NO_ERROR) {
            setLastError(error);
            emit finishedReleaseAssetsInfoDownload(assets, error, context);
            return;
        }
        emit finishedReleaseAssetsInfoDownload(assets, NO_ERROR, context);
        return;
    }
    else if (reply->property(CURRENT_STEP).toInt() == FETCHING_ASSET_FILE) {
        if(networkError) {
            emit fileDownloaded(NULL, NETWORK_ERROR, context);
            return;
        }
        int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if(statusCode == 301 || statusCode == 302) {
            emit logInfo(tr("Received HTML redirect header"));
            QUrl redirectUrl = reply->attribute(QNetworkRequest::RedirectionTargetAttribute).toUrl();
            QNetworkRequest request;
            request.setUrl(redirectUrl);
            reply->deleteLater();
            reply = m_WebCtrl.get(request);
            reply->setProperty(CUSTOM_CONTEXT, context);
            currentNetworkReply = reply;
            connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
            reply->setProperty(CURRENT_STEP, FETCHING_ASSET_FILE);
        }
        else {
            emit fileDownloaded(reply, NO_ERROR, context);
        }
    }
    else if (reply->property(CURRENT_STEP) == FETCHING_RELEASE) {
        if(networkError) {
            gitHubReleaseAPI::release release;
            release.id = -1;
            emit releaseDownloaded(release, NETWORK_ERROR, context);
            return;
        }
        gitHubReleaseAPI::errors error;
        QHash<int, gitHubReleaseAPI::release> releases;
        releases = processReleasesFromNetworkReply(reply, error);
        if(error != NO_ERROR) {
            gitHubReleaseAPI::release release;
            release.id = -1;
            setLastError(error);
            emit releaseDownloaded(release, error, context);
            return;
        }
        if(releases.values().length() != 1) {
            gitHubReleaseAPI::release release;
            release.id = -1;
            setLastError(UNDEFINED_ERROR);
            emit releaseDownloaded(release, UNDEFINED_ERROR, context);
            return;
        }
        emit releaseDownloaded(releases.values().first(), NO_ERROR, context);
        return;
    }
    else if(reply->property(CURRENT_STEP) == FETCHING_ALL_RELEASES) {
        QHash<int, gitHubReleaseAPI::release> releases;
        if(networkError) {
            emit allReleasesDownloaded(releases, NETWORK_ERROR, context);
        }
        gitHubReleaseAPI::errors error;
        releases = processReleasesFromNetworkReply(reply, error);
        if(error != NO_ERROR) {
            setLastError(error);
            emit allReleasesDownloaded(releases, error, context);
            return;
        }
        emit allReleasesDownloaded(releases, NO_ERROR, context);
        return;
    }
    else if(reply->property(CURRENT_STEP) == UPLOADING_ASSET_FILE) {
        if(networkError) {
            emit logError(tr("Asset file upload failed"));
            emit finishedAssetFileUpload(NETWORK_ERROR, context);
        }
        if(!reply->property(DATA_CONTAINER).isValid()) {
            emit logInfo(tr("Asset uploading finished"));
            emit finishedAssetFileUpload(NO_ERROR, context);
            return;
        }
        gitHubReleaseAPI::errors error;
        QHash<int, gitHubReleaseAPI::release> releases = processReleasesFromNetworkReply(reply, error);
        if(error != NO_ERROR) {
            setLastError(error);
            emit finishedAssetFileUpload(error, context);
            return;
        }
        if(releases.values().length() != 1) {
            setLastError(UNDEFINED_ERROR);
            emit finishedAssetFileUpload(UNDEFINED_ERROR, context);
            return;
        }
        gitHubReleaseAPI::release release = releases.values().first();
        newAssetFile nasset = reply->property(DATA_CONTAINER).value<newAssetFile>();
        QFile file(nasset.filename);
        if(!file.open(QIODevice::ReadOnly)) {
            emit logError(tr("Could not open file"));
            emit finishedAssetFileUpload(FILE_HANDLING_ERROR, context);
            return;
        }
        QByteArray data = file.readAll();
        QFileInfo info(nasset.filename);
        QString fname = info.fileName();
        QString uploadURL = release.upload_url;
        uploadURL.replace("{?name}", QString("?name=%0&label=%1").arg(fname).arg(nasset.label));
        QNetworkRequest request;
        request.setUrl(uploadURL);
        request.setRawHeader("Content-Type",QMimeDatabase().mimeTypeForFile(nasset.filename).name().toLocal8Bit());
        handleCredentials(request);
        QNetworkReply *reply = m_WebCtrl.post(request, data);
        currentNetworkReply = reply;
        reply->setProperty(CUSTOM_CONTEXT, context);
        connect(reply, SIGNAL(uploadProgress(qint64,qint64)), SIGNAL(uploadProgress(qint64,qint64)));
        return;
    }
    else if(reply->property(CURRENT_STEP) == CREATING_RELEASE) {
        release rel;
        if(networkError) {
            emit logError(tr("Release creation failed"));
            emit finishedCreatingRelease(rel, NETWORK_ERROR, context);
        }
        gitHubReleaseAPI::errors error;
        QHash<int, gitHubReleaseAPI::release> releases = processReleasesFromNetworkReply(reply, error);
        if((error != NO_ERROR) || (releases.values().length() != 1)) {
            logError(tr("Release creation failed"));
            emit finishedCreatingRelease(rel, UNDEFINED_ERROR, context);
            setLastError(UNDEFINED_ERROR);
            return;
        }
        emit logInfo(tr("Release creation finished"));
        emit finishedCreatingRelease(releases.values().first(), NO_ERROR, context);
    }
    else if(reply->property(CURRENT_STEP) == DELETING_RELEASE) {
        if(networkError) {
            emit logError(tr("Release deletion failed"));
            emit finishedReleaseDeletion(NETWORK_ERROR, context);
            return;
        }
        emit logInfo(tr("Release deletion completed"));
        emit finishedReleaseDeletion(NO_ERROR, context);
        return;
    }
    else if(reply->property(CURRENT_STEP) == DELETING_ASSET) {
        if(networkError) {
            emit logError(tr("Asset deletion failed"));
            emit finishedAssetDeletion(NETWORK_ERROR, context);
            return;
        }
        emit logInfo(tr("Asset deletion completed"));
        emit finishedAssetDeletion(NO_ERROR, context);
        return;
    }
    else if(reply->property(CURRENT_STEP) == EDITING_RELEASE) {
        if(networkError) {
            emit logError(tr("Release editing failed"));
            release ret;
            ret.id = -1;
            emit finishedReleaseEditing(ret, NETWORK_ERROR, context);
            return;
        }
        gitHubReleaseAPI::errors error;
        QHash<int, gitHubReleaseAPI::release> releases = processReleasesFromNetworkReply(reply, error);
        if((error != NO_ERROR) || (releases.values().length() != 1)) {
            emit logError(tr("Release editing failed"));
            release ret;
            ret.id = -1;
            emit finishedReleaseEditing(ret, PARSING_ERROR, context);
            setLastError(PARSING_ERROR);
            return;
        }
        emit logError(tr("Release editing completed"));
        emit finishedReleaseEditing(releases.values().first(), NO_ERROR, context);
        return;
    }
    else if(reply->property(CURRENT_STEP) == EDITING_ASSET) {
        gitHubReleaseAPI::GitHubAsset failedAsset;
        failedAsset.id = -1;
        if(networkError) {
            emit logError(tr("Asset editing failed"));
            emit finishedAssetEditing(failedAsset, NETWORK_ERROR, context);
            return;
        }
        gitHubReleaseAPI::errors error;
        QList<GitHubAsset> assets = processAssetFromNetworkReply(reply, error);
        if((assets.length() != 1) || (error != NO_ERROR)) {
            setLastError(PARSING_ERROR);
            emit logError(tr("Asset editing failed"));
            emit finishedAssetEditing(failedAsset, PARSING_ERROR, context);
            return;
        }
        emit finishedAssetEditing(assets.first(), NO_ERROR, context);
    }
}

/**
 * @brief Gets the information structure of a particular asset
 * This function returns immediately
 * @param assetID ID of the asset to retrieve
 * @param context context of the request
 * Emits finishedAssetInfoDownload(...) when done
 */
void gitHubReleaseAPI::getAsset(int assetID, QVariant context)
{
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg("assets").arg(assetID);
    QNetworkRequest request;
    request.setUrl(workUrl);
    request.setRawHeader("Content-Type", "application/octet-stream");
    handleCredentials(request);
    QNetworkReply *reply = m_WebCtrl.get(request);
    reply->setProperty(CURRENT_STEP, FETCHING_ASSET_INFO);
    reply->setProperty(CUSTOM_CONTEXT, context);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
}

/**
 * @brief Gets the information structures of all assets of a particular release
 * This function returns immediately
 * @param releaseID the ID of the release of which the retrieve the assets information
 * @param context the context of the request
 * Emits finishedReleaseAssetsInfoDownload(...) when done
 */
void gitHubReleaseAPI::getReleaseAssets(int releaseID, QVariant context)
{
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg(releaseID).arg("assets");
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    QNetworkReply *reply = m_WebCtrl.get(request);
    reply->setProperty(CURRENT_STEP, FETCHING_ALL_ASSETS_INFO);
    reply->setProperty(CUSTOM_CONTEXT, context);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
}

/**
 * @brief Uploads a file asset to a particular release
 * This function returns immediately
 * @param filename filename of the file to upload
 * @param label label to be assigned to the asset file
 * @param releaseID ID of the release for which to upload the asset
 * @param context context or the request
 * Emits finishedAssetFileUpload(...) when done
 */
void gitHubReleaseAPI::uploadReleaseAssetFile(QString filename, QString label, int releaseID, QVariant context)
{
    if(releaseID < 0) {
        emit finishedAssetFileUpload(UNDEFINED_ERROR, context);
        return;
    }
    newAssetFile nasset;
    nasset.filename = filename;
    nasset.label = label;
    nasset.releaseID = releaseID;
    getSingleRelease(releaseID, context, nasset);
}

/**
 * @brief Gets a release information structure with the given tag
 * This function returns immediately
 * @param name name of the release tag to retrieve
 * @param context context of the request
 * Emits releaseDownloaded(...) when done
 */
void gitHubReleaseAPI::getReleaseByTagName(QString name, QVariant context)
{
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg("tags").arg(name);
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    QNetworkReply *reply = m_WebCtrl.get(request);
    reply->setProperty(CUSTOM_CONTEXT, context);
    reply->setProperty(CURRENT_STEP, FETCHING_RELEASE);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
}

/**
 * @brief Creates a new release
 * This function returns immediately
 * @param release structure containing the data of the new release
 * @param context context of the request
 * Emits finishedCreatingRelease(...) when done
 */
void gitHubReleaseAPI::createRelease(gitHubReleaseAPI::newGitHubRelease release, QVariant context)
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
    QNetworkReply *reply = m_WebCtrl.post(request, data.toJson());
    reply->setProperty(CUSTOM_CONTEXT, context);
    reply->setProperty(CURRENT_STEP, CREATING_RELEASE);
    currentNetworkReply = reply;
    connect(reply, SIGNAL(uploadProgress(qint64,qint64)), SIGNAL(uploadProgress(qint64,qint64)));
}

/**
 * @brief Edits a release
 * This function returns immediately
 * @param releaseID ID of the release to edit
 * @param editedRelease structure containing the edited data
 * @param context context of the request
 * Emits finishedReleaseEditing(...) when done
 */
void gitHubReleaseAPI::editRelease(int releaseID, gitHubReleaseAPI::newGitHubRelease editedRelease, QVariant context)
{
    gitHubReleaseAPI::release ret;
    if(releaseID < 0) {
        emit logError(tr("Failed to edit release"));
        lastError = UNDEFINED_ERROR;
        emit finishedReleaseEditing(ret, UNDEFINED_ERROR, context);
        return;
    }
    QString workUrl = QString("%0%1/%2").arg(m_url).arg("releases").arg(releaseID);
    QNetworkRequest request;
    request.setRawHeader("Content-Type", "application/x-www-form-urlencoded");
    request.setUrl(workUrl);
    handleCredentials(request);
    QJsonObject obj;
    obj.insert("tag_name", editedRelease.tag_name);
    obj.insert("target_commitish", editedRelease.target_commitish);
    obj.insert("name", editedRelease.name);
    obj.insert("body", editedRelease.body);
    obj.insert("draft", editedRelease.draft);
    obj.insert("prerelease", editedRelease.prerelease);
    QJsonDocument data(obj);
    QByteArray array = data.toJson();
    QBuffer buffer(&array);
    QNetworkReply *reply = m_WebCtrl.sendCustomRequest(request, "PATCH", &buffer);
    reply->setProperty(CURRENT_STEP, EDITING_RELEASE);
    reply->setProperty(CUSTOM_CONTEXT, context);
    return;
}

/**
 * @brief Edits a particular release asset
 * This function returns immediately
 * @param assetID ID of the asset to edit
 * @param filename new filename of the asset
 * @param newLabel new label of the asset
 * @param context context of the request
 * Emits finishedAssetEditing(...) when done
 */
void gitHubReleaseAPI::editAsset(int assetID, QString filename, QString newLabel, QVariant context)
{
    gitHubReleaseAPI::GitHubAsset ret;
    if(assetID < 0) {
        emit logError("Asset editing failed");
        emit finishedAssetEditing(ret, UNDEFINED_ERROR, context);
        return;
    }
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg("assets").arg(assetID);
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
    QNetworkReply *reply = m_WebCtrl.sendCustomRequest(request, "PATCH", &buffer);
    reply->setProperty(CUSTOM_CONTEXT, context);
    reply->setProperty(CURRENT_STEP, EDITING_ASSET);
    return;
}

/**
 * @brief Deletes a release asset
 * This function returns immediately
 * @param assetID ID of the asset to delete
 * @param context context of the request
 * Emits finishedAssetDeletion(...) when done
 */
void gitHubReleaseAPI::deleteAsset(int assetID, QVariant context)
{
    if(assetID < 0) {
        emit finishedAssetDeletion(UNDEFINED_ERROR, context);
        return;
    }
    QString workUrl = QString("%0%1/%2/%3").arg(m_url).arg("releases").arg("assets").arg(assetID);
    QNetworkRequest request;
    request.setRawHeader("Content-Type", "application/x-www-form-urlencoded");
    request.setUrl(workUrl);
    handleCredentials(request);
    QNetworkReply *reply = m_WebCtrl.deleteResource(request);
    reply->setProperty(CURRENT_STEP, DELETING_ASSET);
    reply->setProperty(CUSTOM_CONTEXT, context);
    currentNetworkReply = reply;
}

/**
 * @brief Deletes a release
 * This function returns immediately
 * @param releaseID ID of the release to delete
 * @param context context of the request
 * Emits finishedReleaseDeletion(...) when done
 */
void gitHubReleaseAPI::deleteRelease(int releaseID, QVariant context)
{
    if(releaseID < 0) {
        emit finishedReleaseDeletion(UNDEFINED_ERROR, context);
        return;
    }
    QString workUrl = QString("%0%1/%2").arg(m_url).arg("releases").arg(releaseID);
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    QNetworkReply *reply = m_WebCtrl.deleteResource(request);
    currentNetworkReply = reply;
    reply->setProperty(CURRENT_STEP, DELETING_RELEASE);
    reply->setProperty(CUSTOM_CONTEXT, context);
    return;
}

/**
 * @brief Gets the latest release of the repository
 * This function returns immediately
 * @param context context of the request
 * Emits releaseDownloaded(...) when done
 */
void gitHubReleaseAPI::getLatestRelease(QVariant context)
{
    qDebug() << "getLatestRelease";
    QString workUrl = QString("%0%1/%2").arg(m_url).arg("releases").arg("latest");
    QNetworkRequest request;
    request.setUrl(workUrl);
    handleCredentials(request);
    QNetworkReply *reply = m_WebCtrl.get(request);
    reply->setProperty(CUSTOM_CONTEXT, context);
    reply->setProperty(CURRENT_STEP, FETCHING_RELEASE);
    connect(reply, SIGNAL(downloadProgress(qint64,qint64)), SIGNAL(downloadProgress(qint64,qint64)));
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

/**
 * @brief Call to abort current network operation
 */
void gitHubReleaseAPI::abortOperation()
{
    if(currentNetworkReply.isNull())
        return;
    currentNetworkReply.data()->abort();
}

QHash<int, gitHubReleaseAPI::release> gitHubReleaseAPI::processReleasesFromNetworkReply(QNetworkReply *reply, gitHubReleaseAPI::errors &error)
{
    QHash<int, gitHubReleaseAPI::release> releaseList;
    QJsonParseError perror;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &perror);
    if(perror.error != QJsonParseError::NoError) {
        emit logError(tr("Parsing failed:") + perror.errorString());
        setLastError(PARSING_ERROR);
        error = PARSING_ERROR;
        return releaseList;
    }
    if(doc.isArray()) {
        QJsonArray array = doc.array();
        foreach (QJsonValue value, array) {
            release rel = processRelease(value.toObject());
            releaseList.insert(rel.id, rel);
        }
    }
    else if(doc.isObject()) {
        release rel = processRelease(doc.object());
        releaseList.insert(rel.id, rel);
    }
    error = NO_ERROR;
    return releaseList;
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

QList<gitHubReleaseAPI::GitHubAsset> gitHubReleaseAPI::processAssetFromNetworkReply(QNetworkReply *reply, errors &error) {
    QList<gitHubReleaseAPI::GitHubAsset> assets;
    QJsonParseError perror;
    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll(), &perror);
    if(perror.error != QJsonParseError::NoError) {
        error = PARSING_ERROR;
        return assets;
    }
    if(doc.isArray()) {
        foreach (QJsonValue obj, doc.array()) {
            assets.append(processAsset(obj.toObject()));
        }
    }
    else if(doc.isObject())
    {
        assets.append(processAsset(doc.object()));
    }
    error = NO_ERROR;
    return assets;
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

/**
 * @brief Sets the user credentials to use
 * Needs to be set to be able to retrieve draft releases
 * @param username
 * @param password
 */
void gitHubReleaseAPI::setCredentials(QString username, QString password)
{
    m_username = username;
    m_password = password;
}
