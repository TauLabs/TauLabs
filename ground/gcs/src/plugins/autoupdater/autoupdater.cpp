
/**
 ******************************************************************************
 * @file       autoupdater.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2015
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup AutoUpdater plugin
 * @{
 *
 * @brief Auto updates the GCS from GitHub releases
 * This class starts by periodically calling gitHubs API getLatestRelease
 *  or getReleases (for prereleases)
 * The onLatestReleaseFetchingCompleted or onAllReleasesFetchingCompleted slot
 * is then called
 * After the latest release found on the previous step is found the
 * packageversioninfo.json file is downloaded
 * Currently this file contains the following tags inside the main package_info tag:
 * -uavo_hash contains the UAVO set hash of the release to be used to alert
 *  the user if the set is different
 * -uavo_hash_text same as above in text format
 * If a valid update is found the update dialog will popup
 * If the user starts the update process the Helper application will be
 * downloaded and decompressed to the temp folder
 * The new application package is then downloaded and decompressed
 * to the temp folder
 * The helper application will be called and the main application will exit
 *****************************************************************************/
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Generalpackage_info Public License as published by
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

#include "autoupdater.h"
#include <QtGlobal>
#include <QDir>
#include <QMessageBox>
#include <QApplication>
#include <QtMath>
#include <QSysInfo>
#include "quazipfile.h"
#include "JlCompress.h"
#include <QThread>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrentRun>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QJsonArray>
#include "gcsversioninfo.h"
#include "coreplugin/coreconstants.h"

#ifdef Q_OS_WIN
#define EXEC "taulabsgcs.exe"
#define COPY_APP_EXEC "copyApp.exe"
#define PLATFORM "winx86"
#elif defined(Q_OS_LINUX)
#define EXEC "taulabsgcs"
#define COPY_APP_EXEC "copyApp"
#define PLATFORM "linux"
#else
#define EXEC "Tau Labs GCS"
#define COPY_APP_EXEC "copyApp"
#define PLATFORM "osx"
#endif

/**
 * @brief Class constructer
 * @param parent pointer to parent object
 * @param ObjMngr pointer to UAV object manager
 * @param refreshIntervalSec frequency in seconds to look for updates
 * @param usePrereleases true if the plugin should look for prereleases (requires user authentication and permissions)
 * @param gitHubUrl URL of the gitHub repository in which to look for updates
 * @param gitHubUsername gitHub username - only needed to use prereleases
 * @param gitHubPassword gitHub password - only needed to use prereleases
 */
AutoUpdater::AutoUpdater(QWidget *parent, UAVObjectManager * ObjMngr, int refreshIntervalSec, bool usePrereleases, QString gitHubUrl,  QString gitHubUsername, QString gitHubPassword) : usePrereleases(usePrereleases), mainAppApi(this), helperAppApi(this), dialog(NULL), parent(parent)
{
    Q_UNUSED(ObjMngr)
    mainAppApi.setRepo(gitHubUrl);
    if(!gitHubUsername.isEmpty() && !gitHubPassword.isEmpty())
        mainAppApi.setCredentials(gitHubUsername, gitHubPassword);
    helperAppApi.setRepo("PTDreamer", "copyApp");

    refreshTimer.setInterval(refreshIntervalSec * 1000);
    if(refreshIntervalSec != 0)
        refreshTimer.start();

    process = new QProcess(parent);

    preferredPlatformStr = PLATFORM;

    if(QSysInfo::WordSize == 64) {
        compatiblePlatformStr = QString("%0_%1").arg(preferredPlatformStr).arg(32);
    }
    preferredPlatformStr.append(QString("_%0").arg(QSysInfo::WordSize));

    connect(this, SIGNAL(updateFound(gitHubReleaseAPI::release, packageVersionInfo)), this, SLOT(onUpdateFound(gitHubReleaseAPI::release, packageVersionInfo)));
    connect(this, SIGNAL(progressMessage(QString)), this, SLOT(onProgressText(QString)));
    connect(this, SIGNAL(decompressProgress(int)), this, SLOT(onProgress(int)));
    connect(this, SIGNAL(currentOperationMessage(QString)), this, SLOT(onNewOperation(QString)));

    connect(&refreshTimer, SIGNAL(timeout()), this, SLOT(onRefreshTimerTimeout()));
    connect(&mainAppApi, SIGNAL(downloadProgress(qint64,qint64)), this, SLOT(downloadProgress(qint64,qint64)));
    connect(&helperAppApi, SIGNAL(downloadProgress(qint64,qint64)), this, SLOT(downloadProgress(qint64,qint64)));
    connect(&helperAppApi, SIGNAL(allReleasesDownloaded(QHash<int,gitHubReleaseAPI::release>,gitHubReleaseAPI::errors,QVariant)), this, SLOT(onAllReleasesFetchingCompleted(QHash<int,gitHubReleaseAPI::release>,gitHubReleaseAPI::errors,QVariant)));
    connect(&helperAppApi, SIGNAL(releaseDownloaded(gitHubReleaseAPI::release,gitHubReleaseAPI::errors,QVariant)), this, SLOT(onLatestReleaseFetchingCompleted(gitHubReleaseAPI::release,gitHubReleaseAPI::errors,QVariant)));
    connect(&helperAppApi, SIGNAL(fileDownloaded(QNetworkReply*,gitHubReleaseAPI::errors,QVariant)), this, SLOT(onFileDownloaded(QNetworkReply*,gitHubReleaseAPI::errors,QVariant)));
    connect(&mainAppApi, SIGNAL(allReleasesDownloaded(QHash<int,gitHubReleaseAPI::release>,gitHubReleaseAPI::errors,QVariant)), this, SLOT(onAllReleasesFetchingCompleted(QHash<int,gitHubReleaseAPI::release>,gitHubReleaseAPI::errors,QVariant)));
    connect(&mainAppApi, SIGNAL(releaseDownloaded(gitHubReleaseAPI::release,gitHubReleaseAPI::errors,QVariant)), this, SLOT(onLatestReleaseFetchingCompleted(gitHubReleaseAPI::release,gitHubReleaseAPI::errors,QVariant)));
    connect(&mainAppApi, SIGNAL(fileDownloaded(QNetworkReply*,gitHubReleaseAPI::errors,QVariant)), this, SLOT(onFileDownloaded(QNetworkReply*,gitHubReleaseAPI::errors,QVariant)));
}

/**
 * @brief Called by the plugin configuration to update configuration values
 * @param refreshInterval
 * @param usePrereleases
 * @param gitHubUrl
 * @param gitHubUsername
 * @param gitHubPassword
 */
void AutoUpdater::refreshSettings(int refreshInterval, bool usePrereleases, QString gitHubUrl, QString gitHubUsername, QString gitHubPassword)
{
    refreshTimer.setInterval(refreshInterval *1000);
    this->usePrereleases = usePrereleases;
    mainAppApi.setRepo(gitHubUrl);
    if(!gitHubUsername.isEmpty() && !gitHubPassword.isEmpty())
        mainAppApi.setCredentials(gitHubUsername, gitHubPassword);
    if(refreshInterval == 0)
        refreshTimer.stop();
    else
        refreshTimer.start();
}

/**
 * @brief Called by the refresh timer. Looks for new releases
 */
void AutoUpdater::onRefreshTimerTimeout()
{
    if(!dialog.isNull()) {
        return;
    }
    if(!usePrereleases)
        mainAppApi.getLatestRelease(FETCHING_LATEST_APP_RELEASE);
    else
        mainAppApi.getReleases(FETCHING_ALL_APP_RELEASES);
}

/**
 * @brief Called by the gitHubAPI when a file download is completed
 * @param reply pointer to the QNetworkReply returned by the API
 * @param error error of the operation which resulted in this call
 * @param context context as passed in the request which lead to this call
 */
void AutoUpdater::onFileDownloaded(QNetworkReply *reply, gitHubReleaseAPI::errors error, QVariant context)
{
    //Safe as it will only get deleted from the main eventloop
    reply->deleteLater();
    download_asset_struct asset = context.value<AutoUpdater::download_asset_struct>();
    process_steps_enum step = asset.step;
    QVariant data = asset.data1;
    if(step == FETCHING_VERSION_INFO_FILE) {
        packageVersionInfo ret;
        QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
        if(doc.isNull()) {
            return;
        }
        QJsonObject obj = doc.object();
        QJsonValue val = obj.value("package_info");
        if(val.isUndefined()) {
            return;
        }
        if(!val.toObject().value("uavo_hash").isUndefined())
            ret.uavo_hash = val.toObject().value("uavo_hash").toString();
        if(!val.toObject().value("uavo_hash_text").isUndefined())
            ret.uavo_hash_txt = val.toObject().value("uavo_hash_text").toString();
        if(!val.toObject().value("update_assets").isUndefined()) {
            foreach (QJsonValue value, val.toObject().value("update_assets").toArray()) {
                ret.package_assets.append(value.toString());
            }
        }
        if(!val.toObject().value("date").isUndefined())
            ret.date = QDateTime::fromString(val.toObject().value("date").toString(),"yyyyMMdd hh:mm");
        QDateTime currentGCSDate;
        QString versionData = QLatin1String(Core::Constants::GCS_REVISION_STR);
        currentGCSDate = QDateTime::fromString(versionData.split(" ").last(),"yyyyMMdd");
        if(ret.date > currentGCSDate)
            ret.isNewer = true;
        else
        {
            ret.isNewer = false;
            return;
        }
        QString uavoHash = QString::fromLatin1(Core::Constants::UAVOSHA1_STR);
        ret.isUAVODifferent = (uavoHash != ret.uavo_hash);
        bool existsForThisPlattform = checkIfReleaseContainsProperAsset(data.value<gitHubReleaseAPI::release>());
        if(existsForThisPlattform && ret.isNewer) {
            gitHubReleaseAPI::release release = data.value<gitHubReleaseAPI::release>();
            emit updateFound(release, ret);
        }
    }
    if(step == FETCHING_HELPER_PACKAGE) {
        if(error != gitHubReleaseAPI::NO_ERROR) {
            QMessageBox::critical(parent, tr("ERROR"), tr("Could not retrieve application package needed for the update process"));
            if(!dialog.isNull())
                dialog.data()->close();
            return;
        }
        gitHubReleaseAPI::GitHubAsset helperAsset = data.value<gitHubReleaseAPI::GitHubAsset>();
        QFile copyAppFile;
        copyAppFile.setFileName(QDir::tempPath() + QDir::separator() + helperAsset.name);
        if(!copyAppFile.open(QIODevice::WriteOnly)) {
            QMessageBox::critical(parent, tr("ERROR"), QString(tr("Could not open file for writing %0")).arg(copyAppFile.fileName()));
            if(!dialog.isNull())
                dialog.data()->close();
            return;
        }
        copyAppFile.write(reply->readAll());
        copyAppFile.close();
        emit currentOperationMessage(tr("Decompressing helper application"));
        AutoUpdater::decompressResult copyAppDecompressResult = fileDecompress(copyAppFile.fileName(), QDir::tempPath(), COPY_APP_EXEC);
        if(!copyAppDecompressResult.success) {
            QMessageBox::critical(parent, tr("ERROR"), tr("Something went wrong during helper application file decompression!"));
            if(!dialog.isNull())
                dialog.data()->close();
            return;
        }
        gitHubReleaseAPI::release appRelease = dialog.data()->getContext().value<gitHubReleaseAPI::release>();
        gitHubReleaseAPI::GitHubAsset appAsset = appRelease.assets.value(lookForPlattformAsset(appRelease.assets));
        emit currentOperationMessage(tr("Downloading latest application package"));
        download_asset_struct contextData;
        contextData.step = FETCHING_APP_PACKAGE;
        contextData.data1.setValue(appAsset);
        contextData.data2.setValue(copyAppDecompressResult);
        QVariant context;
        context.setValue(contextData);
        mainAppApi.downloadAssetFile(appAsset.id, context);
    }
    else if(step == FETCHING_APP_PACKAGE) {
        if(error != gitHubReleaseAPI::NO_ERROR) {
            QMessageBox::critical(parent, tr("ERROR"), QString(tr("Could not download application package file")));
            if(!dialog.isNull())
                dialog.data()->close();
            return;
        }
        gitHubReleaseAPI::GitHubAsset appAsset = asset.data1.value<gitHubReleaseAPI::GitHubAsset>();
        QFile file;
        file.setFileName(QDir::tempPath() + QDir::separator() + appAsset.name);
        if(!file.open(QIODevice::WriteOnly)) {
            QMessageBox::critical(parent, tr("ERROR"), QString(tr("Could not open file for writing %0")).arg(file.fileName()));
            if(!dialog.isNull())
                dialog.data()->close();
            return;
        }
        file.write(reply->readAll());
        emit currentOperationMessage(tr("Decompressing new application compressed file"));
        AutoUpdater::decompressResult appDecompressResult = fileDecompress(file.fileName(), QDir::tempPath(), QFileInfo(qApp->applicationFilePath()).fileName());
        if(!appDecompressResult.success) {
            QMessageBox::critical(parent, tr("ERROR"), tr("Something went wrong during file decompression!"));
            if(!dialog.isNull())
                dialog.data()->close();
            return;
        }
        QDir appDir = QApplication::applicationDirPath();
        appDir.cdUp();
        appDecompressResult.execPath.cdUp();
        QString appExec = QApplication::applicationFilePath();
#ifdef Q_OS_OSX
        appDir.cdUp();
        appDecompressResult.execPath.cdUp();
        appExec = QString("%0%1%2").arg("../").arg(EXEC).arg(".app");
#endif
        decompressResult copyAppDecompressResult = asset.data2.value<AutoUpdater::decompressResult>();
        qDebug() << copyAppDecompressResult.execPath << copyAppDecompressResult.execPath.absolutePath();
        QMessageBox::information(parent, tr("Update Ready"), tr("The file fetching process is completed, press ok to continue the update"));
        qDebug() << "starting copyapp" << copyAppDecompressResult.execPath.absolutePath() + QDir::separator() + COPY_APP_EXEC <<appDecompressResult.execPath.absolutePath() <<  appDir.absolutePath() << QApplication::applicationFilePath();
        process->startDetached(copyAppDecompressResult.execPath.absolutePath() + QDir::separator() + COPY_APP_EXEC, QStringList() << appDecompressResult.execPath.absolutePath() <<  appDir.absolutePath() << appExec);
        QApplication::quit();
    }
}

void AutoUpdater::onLatestReleaseFetchingCompleted(gitHubReleaseAPI::release release, gitHubReleaseAPI::errors error, QVariant context)
{
    if(error != gitHubReleaseAPI::NO_ERROR)
        return;
    process_steps_enum step = (process_steps_enum)context.toInt();
    if(step == FETCHING_LATEST_APP_RELEASE) {
        processLatestRelease(release);
    }
    else if(step == FETCHING_LATEST_HELPER_RELEASE) {
        int helperID = lookForPlattformAsset(release.assets);
        if(helperID == -1) {
            QMessageBox::critical(parent, tr("ERROR"), tr("Could not retrieve application package needed for the update process"));
            if(!dialog.isNull())
                dialog.data()->close();
            return;
        }
        download_asset_struct asset;
        asset.step = FETCHING_HELPER_PACKAGE;
        asset.data1.setValue(release.assets.value(helperID));
        QVariant context;
        context.setValue(asset);
        helperAppApi.downloadAssetFile(helperID, context);
        emit currentOperationMessage(tr("Downloading latest helper application file"));
    }
}

void AutoUpdater::onAllReleasesFetchingCompleted(QHash<int, gitHubReleaseAPI::release> releaseList, gitHubReleaseAPI::errors error, QVariant context)
{
    if(error != gitHubReleaseAPI::NO_ERROR)
        return;
    process_steps_enum step = (process_steps_enum)context.toInt();
    if(step == FETCHING_ALL_APP_RELEASES) {
        gitHubReleaseAPI::release mostRecentRelease = releaseList.values().first();
        foreach (gitHubReleaseAPI::release release, releaseList.values()) {
            if(release.published_at > mostRecentRelease.published_at)
                mostRecentRelease = release;
        }
        processLatestRelease(mostRecentRelease);
    }
    else {
        Q_ASSERT(0);
    }
}

void AutoUpdater::onUpdateFound(gitHubReleaseAPI::release release, packageVersionInfo info)
{
    if(!dialog.isNull()) {
        return;
    }
    QVariant context;
    context.setValue(release);
    dialog = new updaterFormDialog(release.body, info.isUAVODifferent, context, parent);
    dialog.data()->show();
    dialog.data()->raise();
    dialog.data()->activateWindow();
    connect(dialog.data(), SIGNAL(startUpdate()), this, SLOT(onDialogStartUpdate()));
    connect(dialog.data(), SIGNAL(cancelDownload()), &mainAppApi, SLOT(abortOperation()));
    connect(dialog.data(), SIGNAL(dialogAboutToClose(bool)), this, SLOT(onCancel(bool)));
    connect(dialog.data(), SIGNAL(cancelDownload()), &helperAppApi, SLOT(abortOperation()));
}

void AutoUpdater::onDialogStartUpdate()
{
    emit currentOperationMessage(tr("Looking for latest helper application"));
    helperAppApi.getLatestRelease(FETCHING_LATEST_HELPER_RELEASE);
}

void AutoUpdater::onCancel(bool dontShowAgain)
{
    if(dontShowAgain)
        refreshTimer.stop();
    else
        refreshTimer.start();
}

void AutoUpdater::downloadProgress(qint64 progress, qint64 total)
{
    if(total <= 0)
        return;
    if(!dialog.isNull()) {
        dialog.data()->setProgress(QString(tr("%0 of %1 bytes downloaded")).arg(progress).arg(total));
        int p = (progress * 100) / total;
        dialog.data()->setProgress(p);
    }
}

void AutoUpdater::onNewOperation(QString newOp)
{
    if(!dialog.isNull())
        dialog.data()->setOperation(newOp);
}

void AutoUpdater::onProgressText(QString newTxt)
{
    if(!dialog.isNull())
        dialog.data()->setProgress(newTxt);
}

void AutoUpdater::onProgress(int value)
{
    if(!dialog.isNull())
        dialog.data()->setProgress(value);
}

#if defined(Q_OS_LINUX)
AutoUpdater::decompressResult AutoUpdater::fileDecompress(QString fileName, QString destinationPath, QString execFile)
{
    AutoUpdater::decompressResult ret;
    QFile file(fileName);
    if(!file.exists()) {
        ret.success = false;
        return ret;
    }
    QString cmd = QString("/bin/sh -c \"xz -l %0 | grep -oh -m2 \"[0-9]*\\.[0-9].MiB\" | tail -1\"").arg(fileName);
    QEventLoop loop;
    QProcess process;
    connect(&process, SIGNAL(finished(int)), &loop, SLOT(quit()));
    process.start(cmd);
    loop.exec();
    if(process.exitStatus() != QProcess::NormalExit) {
        ret.success = false;
        return ret;
    }
    QString totalSizeStr = process.readAll();
    QString unit = totalSizeStr.split(" ").value(1);
    QString value = totalSizeStr.split(" ").value(0);
    qint64 multiplier;
    if(unit.contains("KiB"))
        multiplier = qPow(2,10);
    else if(unit.contains("MiB"))
        multiplier = qPow(2,20);
    else if(unit.contains("GiB"))
        multiplier = qPow(2,30);
    else
        multiplier = 1;
    bool ok;
    qint64 size = value.replace(",", ".").toDouble(&ok) * multiplier;
    qint64 partial = 0;
    if(!ok)
        size = 0;
    cmd = QString("tar -xvvf %0 -C %1").arg(fileName).arg(destinationPath);
    process.start("/bin/sh", QStringList() << "-c" << cmd);
    connect(&process, SIGNAL(readyRead()), &loop, SLOT(quit()));
    QString receiveBuffer;
    while(process.state() != QProcess::NotRunning) {
        loop.exec();
        do {
            receiveBuffer = QString(process.readLine());
            qDebug() << "decompress" << receiveBuffer;
            if(!receiveBuffer.isEmpty()) {
                emit progressMessage(QString("Extracting %0").arg(QString(receiveBuffer.split(QRegExp("\\s+")).value(5)).split(QDir::separator()).last()));
                if(QString(receiveBuffer.split(QRegExp("\\s+")).value(5)).split(QDir::separator()).last() == execFile) {

                    ret.execPath = QDir(QDir::tempPath() + QDir::separator() + receiveBuffer.split(QRegExp("\\s+")).value(5));
                }
                if(size != 0) {
                    QString s = receiveBuffer.split(QRegExp("\\s+")).value(2);
                    if(true) {
                        partial += s.toInt();
                        emit decompressProgress(partial * 100 / size);
                    }
                }
            }
        } while (process.canReadLine());

    }
    if(process.exitStatus() != QProcess::NormalExit) {
        ret.success = false;
        return ret;
    }
    emit decompressProgress(100);
    ret.success = true;
    ret.execPath.cdUp();
    return ret;
}
#endif
#if defined(Q_OS_WIN) || defined(Q_OS_OSX)
AutoUpdater::decompressResult AutoUpdater::fileDecompress(QString zipfile, QString destinationPath, QString execFile)
{
    AutoUpdater::decompressResult res;
    QEventLoop loop;
    QFutureWatcher<AutoUpdater::decompressResult> watcher;
    connect(&watcher, SIGNAL(finished()), &loop, SLOT(quit()));
    if(!QFile(zipfile).exists())
        return res;
    QFuture <AutoUpdater::decompressResult> future = QtConcurrent::run(this, &AutoUpdater::zipFileDecompress, zipfile, destinationPath, execFile);
    watcher.setFuture(future);
    loop.exec();
    return future.result();
}

AutoUpdater::decompressResult AutoUpdater::zipFileDecompress(QString zipfile, QString destinationPath, QString exeFile)
{
    AutoUpdater::decompressResult ret;
    qint64 totalDecompressedSize = 0;
    QDir destination(destinationPath);
    QuaZip archive(zipfile);
    if(!archive.open(QuaZip::mdUnzip)) {
        ret.success = false;
        return ret;
    }
    for( bool f = archive.goToFirstFile(); f; f = archive.goToNextFile() )
    {
        QuaZipFileInfo64 *info = new QuaZipFileInfo64;
        if(archive.getCurrentFileInfo(info)) {
            totalDecompressedSize += info->uncompressedSize;
        }
    }
    qint64 currentDecompressedSize = 0;
    for( bool f = archive.goToFirstFile(); f; f = archive.goToNextFile() )
    {
        QuaZipFileInfo64 info;
        archive.getCurrentFileInfo(&info);
        currentDecompressedSize += info.uncompressedSize;
        emit progressMessage(QString("Extracting %0").arg(QFileInfo(info.name).fileName()));
        QString name = archive.getCurrentFileName();
        QString absFilePath = destination.absoluteFilePath(name);
        if(!exeFile.isEmpty() && (exeFile == QFileInfo(name).fileName())) {
            ret.execPath = QDir(QFileInfo(absFilePath).canonicalPath());
            ret.success = true;
        }
        emit decompressProgress(currentDecompressedSize * 100 / totalDecompressedSize);
    }
    emit decompressProgress(100);
    return ret;
}
#endif

void AutoUpdater::processLatestRelease(gitHubReleaseAPI::release release)
{
    int versionInfoID = -1;
    foreach (int key, release.assets.keys()) {
        if(release.assets.value(key).name == "packageversioninfo.json") {
            versionInfoID = key;
        }
    }
    if(versionInfoID == -1) {
        return;
    }
    download_asset_struct asset;
    asset.data1.setValue(release);
    asset.step = FETCHING_VERSION_INFO_FILE;
    QVariant context;
    context.setValue(asset);
    mainAppApi.downloadAssetFile(versionInfoID, context);
}

bool AutoUpdater::checkIfReleaseContainsProperAsset(gitHubReleaseAPI::release release)
{
    foreach (gitHubReleaseAPI::GitHubAsset asset, release.assets.values()) {
        if(asset.label.contains("gcs") && (asset.label.contains(preferredPlatformStr) || asset.label.contains(compatiblePlatformStr)))
            return true;
    }
    return false;
}

int AutoUpdater::lookForPlattformAsset(QHash<int, gitHubReleaseAPI::GitHubAsset> assets)
{
    int copyAppID = -1;
    if(assets.values().size() == 0) {
        return -1;
    }
    foreach (int assetID, assets.keys()) {
        if(assets.value(assetID).label.contains(compatiblePlatformStr, Qt::CaseInsensitive)) {
            copyAppID = assetID;
        }
        if(assets.value(assetID).label.contains(preferredPlatformStr, Qt::CaseInsensitive)) {
            copyAppID = assetID;
            break;
        }
    }
    return copyAppID;
}
