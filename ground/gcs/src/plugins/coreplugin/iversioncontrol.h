/**
 ******************************************************************************
 *
 * @file       iversioncontrol.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
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

#ifndef IVERSIONCONTROL_H
#define IVERSIONCONTROL_H

#include "core_global.h"

#include <QtCore/QObject>
#include <QtCore/QString>

namespace Core {

class CORE_EXPORT IVersionControl : public QObject
{
    Q_OBJECT
public:
    enum Operation { AddOperation, DeleteOperation, OpenOperation };

    IVersionControl(QObject *parent = 0) : QObject(parent) {}
    virtual ~IVersionControl() {}

    virtual QString name() const = 0;

    virtual bool isEnabled() const = 0;

    /*!
     * Enable the VCS, that is, make its menu actions visible.
     */
    virtual void setEnabled(bool enabled) = 0;

    /*!
     * Returns whether files in this directory should be managed with this
     * version control.
     */
    virtual bool managesDirectory(const QString &filename) const = 0;

    /*!
     * This function should return the topmost directory, for which this
     * IVersionControl should be used. The VCSManager assumes that all files in
     * the returned directory are managed by the same IVersionControl.
     *
     * Note that this is used as an optimization, so that the VCSManager
     * doesn't need to call managesDirectory(..) for each directory.
     *
     * This function is called after finding out that the directory is managed
     * by a specific version control.
     */
    virtual QString findTopLevelForDirectory(const QString &directory) const = 0;

    /*!
     * Called to query whether a VCS supports the respective operations.
     */
    virtual bool supportsOperation(Operation operation) const = 0;

    /*!
     * Called prior to save, if the file is read only. Should be implemented if
     * the scc requires a operation before editing the file, e.g. 'p4 edit'
     *
     * \note The EditorManager calls this for the editors.
     */
    virtual bool vcsOpen(const QString &fileName) = 0;

    /*!
     * Called after a file has been added to a project If the version control
     * needs to know which files it needs to track you should reimplement this
     * function, e.g. 'p4 add', 'cvs add', 'svn add'.
     *
     * \note This function should be called from IProject subclasses after
     *       files are added to the project.
     */
    virtual bool vcsAdd(const QString &filename) = 0;

    /*!
     * Called after a file has been removed from the project (if the user
     * wants), e.g. 'p4 delete', 'svn delete'.
     *
     * You probably want to call VcsManager::showDeleteDialog, which asks the
     * user to confirm the deletion.
     */
    virtual bool vcsDelete(const QString &filename) = 0;

signals:
    void repositoryChanged(const QString &repository);
    void filesChanged(const QStringList &files);

    // TODO: ADD A WAY TO DETECT WHETHER A FILE IS MANAGED, e.g
    // virtual bool sccManaged(const QString &filename) = 0;
};

} // namespace Core

#endif // IVERSIONCONTROL_H
