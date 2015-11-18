/**
 ******************************************************************************
 *
 * @file       fileutils.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Uploader Uploader Plugin
 * @{
 * @brief File functions helper class (zip, unzip, delete dirs...)
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

#ifndef FILEUTILS_H
#define FILEUTILS_H
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFileInfoList>
#include "quazipfile.h"

class FileUtils
{
public:
    FileUtils();
    static bool removeDir(const QString &dirName);
    static bool archive(const QString &filePath, const QDir &dir, const QString &directory, const QString &comment);
    static bool extractAll(QString zipfile, QDir destination);
};

#endif // FILEUTILS_H
