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

#ifndef GITHUBRELEASEAPI_H
#define GITHUBRELEASEAPI_H

#include <QObject>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QXmlStreamReader>
#include <QDomDocument>
#include <QTimer>
#include <QEventLoop>
#include <QBuffer>
#include <QFile>
#include <QMimeDatabase>
#include <QFileInfo>
#include <QTableWidgetItem>
#include <QPointer>

#if defined (GITHUBRELEASEAPI_LIBRARY)
#define GITHUBRELEASEAPI_EXPORT Q_DECL_EXPORT
#else
#define GITHUBRELEASEAPI_EXPORT Q_DECL_IMPORT
#endif

class GITHUBRELEASEAPI_EXPORT gitHubReleaseAPI : public QObject
{
    Q_OBJECT
public:
    gitHubReleaseAPI(QObject *parent);
    void setRepo(QString owner, QString repo);
    void setRepo(QString url);
    enum errors { NETWORK_ERROR, TIMEOUT_ERROR, PARSING_ERROR, UNDEFINED_ERROR, NO_ERROR };
    struct GitHubUser {
        QString login;
        int id;
        QUrl avatar_url;
        QString gravatar_id;
        QUrl url;
        QUrl html_url;
        QUrl followers_url;
        QString following_url;
        QString gists_url;
        QString starred_url;
        QUrl subscriptions_url;
        QUrl organizations_url;
        QUrl repos_url;
        QString events_url;
        QUrl received_events_url;
        QString type;
        bool site_admin;
    };
    struct GitHubAsset {
        QUrl url;
        int id;
        QString name;
        QString label;
        GitHubUser uploader;
        QString content_type;
        QString state;
        int size;
        int download_count;
        QDateTime created_at;
        QDateTime updated_at;
        QUrl browser_download_url;
    };

    struct release {
        QUrl url;
        QUrl html_url;
        QUrl assets_url;
        QString upload_url;
        QUrl tarball_url;
        QUrl zipball_url;
        int id;
        QString tag_name;
        QString target_commitish;
        QString name;
        QString body;
        bool draft;
        bool prerelease;
        QDateTime created_at;
        QDateTime published_at;
        GitHubUser author;
        QHash<int, GitHubAsset> assets;
    };

    struct newGitHubRelease {
        QString tag_name;
        QString target_commitish;
        QString name;
        QString body;
        bool draft;
        bool prerelease;
    };

    release processRelease(QJsonObject obj);
    gitHubReleaseAPI::GitHubUser processUser(QJsonObject obj);
    gitHubReleaseAPI::GitHubAsset processAsset(QJsonObject obj);
    QHash<int, gitHubReleaseAPI::release> getReleases();

    gitHubReleaseAPI::release getSingleRelease(int id);
    gitHubReleaseAPI::release getLatestRelease();
    gitHubReleaseAPI::release getReleaseByTagName(QString name);

    gitHubReleaseAPI::release createRelease(gitHubReleaseAPI::newGitHubRelease release);
    void setCredentials(QString username, QString password);
    gitHubReleaseAPI::release editRelease(int id, gitHubReleaseAPI::newGitHubRelease release);
    bool deleteRelease(int id);
    void handleCredentials(QNetworkRequest &request);
    QList<gitHubReleaseAPI::GitHubAsset> getReleaseAssets(int id);
    bool uploadReleaseAsset(QString filename, QString label, int releaseID);
    QByteArray downloadAsset(int id);
    gitHubReleaseAPI::GitHubAsset getAsset(int id);
    bool deleteAsset(int id);
    gitHubReleaseAPI::GitHubAsset editAsset(int id, QString filename, QString newLabel);
    errors getLastError() const;
    void setLastError(const errors &value);

    QString getRepo();
private:
    QString m_url;
    QString m_username;
    QString m_password;
    void downloadFile(QUrl &url);
    QNetworkAccessManager m_WebCtrl;
    QEventLoop eventLoop;
    QTimer timeOutTimer;
    errors lastError;
    QPointer<QNetworkReply> currentNetworkReply;
private slots:
    void onLogInfo(QString);
    void onLogError(QString);

public slots:
    void abortOperation();

signals:
    void fileDownloaded();
    void logInfo(QString);
    void logError(QString);
    void downloadProgress(qint64 total, qint64 downloaded);
    void uploadProgress(qint64 total, qint64 uploaded);
};

#endif // GITHUBRELEASEAPI_H
