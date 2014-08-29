/**
 ******************************************************************************
 *
 * @file       fileutils.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Uploader Uploader Plugin
 * @{
 * @brief File functions helper class (zip, unzip, delete dirs...)
 *****************************************************************************/
 /* This program is free software; you can redistribute it and/or modify
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

#include "fileutils.h"
#include <QDebug>

FileUtils::FileUtils()
{
}

/*!
   Delete a directory along with all of its contents.

   \param dirName Path of directory to remove.
   \return true on success; false on error.
*/
bool FileUtils::removeDir(const QString &dirName)
{
    bool result = true;
    QDir dir(dirName);

    if (dir.exists(dirName)) {
        Q_FOREACH(QFileInfo info, dir.entryInfoList(QDir::NoDotAndDotDot | QDir::System | QDir::Hidden  | QDir::AllDirs | QDir::Files, QDir::DirsFirst)) {
            if (info.isDir()) {
                result = removeDir(info.absoluteFilePath());
            }
            else {
                result = QFile::remove(info.absoluteFilePath());
            }

            if (!result) {
                return result;
            }
        }
        result = dir.rmdir(dirName);
    }

    return result;
}

bool FileUtils::archive(const QString & filePath, const QDir & dir, const QString & directory,const QString & comment) {

    QuaZip zip(filePath);
    zip.setFileNameCodec("IBM866");

    if (!zip.open(QuaZip::mdCreate)) {
        return false;
    }

    if (!dir.exists()) {
        return false;
    }

    QFile inFile;
    QStringList sl;
    foreach(QString filename,dir.entryList())
        sl.append(dir.absolutePath()+QDir::separator()+filename);
    QFileInfoList files;
    foreach (QString fn, sl) files << QFileInfo(fn);

    QuaZipFile outFile(&zip);

    char c;
    foreach(QFileInfo fileInfo, files) {

        if (!fileInfo.isFile())
            continue;

        QString fileName = fileInfo.filePath().remove(0, dir.absolutePath().length() + 1);

        inFile.setFileName(fileInfo.filePath());

        if (!inFile.open(QIODevice::ReadOnly)) {
            return false;
        }

        if (!outFile.open(QIODevice::WriteOnly, QuaZipNewInfo(directory + QDir::separator() + fileName, fileName))) {
            return false;
        }

        while (inFile.getChar(&c) && outFile.putChar(c));

        if (outFile.getZipError() != UNZ_OK) {
            return false;
        }

        outFile.close();

        if (outFile.getZipError() != UNZ_OK) {
            return false;
        }

        inFile.close();
    }

    if (!comment.isEmpty())
        zip.setComment(comment);

    zip.close();

    if (zip.getZipError() != 0) {
        return false;
    }

    return true;
}
bool FileUtils::extractAll(QString zipfile,QDir destination)
{
    QuaZip archive(zipfile);
    if(!archive.open(QuaZip::mdUnzip))
        return false;
    for( bool f = archive.goToFirstFile(); f; f = archive.goToNextFile() )
    {
        // set source file in archive
        QString filePath = archive.getCurrentFileName();
        QuaZipFile zFile( archive.getZipName(), filePath );
        // open the source file
        if(!zFile.open( QIODevice::ReadOnly ))
            return false;
        // create a bytes array and write the file data into it
        QByteArray ba = zFile.readAll();
        // close the source file
        zFile.close();
        // set destination file
        QFileInfo file(filePath);
        QFile dstFile( destination.absolutePath()+QDir::separator()+file.fileName() );
        // open the destination file
        if(!dstFile.open( QIODevice::WriteOnly))
            return false;
        // write the data from the bytes array into the destination file
        dstFile.write(ba);
        //close the destination file
        dstFile.close();
    }
    return true;
}
