/**
 ******************************************************************************
 * @file       kmlexportplugin.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup KmlExportPlugin
 * @{
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

#include "kmlexportplugin.h"
#include <QDebug>
#include <QtPlugin>
#include <QThread>
#include <QStringList>
#include <QDir>
#include <QFileDialog>
#include <QList>
#include <QErrorMessage>
#include <QWriteLocker>

#include <extensionsystem/pluginmanager.h>
#include <QKeySequence>
#include "uavobjectmanager.h"

#include "kmlexport.h"

KmlExportPlugin::KmlExportPlugin()
{
}

/**
 * Add KmlExport option to the tools menu
 */
bool KmlExportPlugin::initialize(const QStringList& args, QString *errMsg)
{
    Q_UNUSED(args);
    Q_UNUSED(errMsg);

    // Add Menu entry
    Core::ActionManager* am = Core::ICore::instance()->actionManager();
    Core::ActionContainer* ac = am->actionContainer(Core::Constants::M_TOOLS);

    // Command to convert log file to KML
    exportToKmlCmd = am->registerAction(new QAction(this), "KmlExport.ExportToKML",
                             QList<int>() << Core::Constants::C_GLOBAL_ID);
    exportToKmlCmd->action()->setText("Export logfile to KML");

    ac->menu()->addSeparator();
    ac->appendGroup("KML Export");
    ac->addAction(exportToKmlCmd, "KML Export");

    connect(exportToKmlCmd->action(), SIGNAL(triggered(bool)), this, SLOT(exportToKML()));

    return true;
}

/**
 * Allow user to select a UAVTalk log file and output KML file and passes
 * that to the @ref KmlExport class
 */
void KmlExportPlugin::exportToKML()
{
    // Get input file
    QString inputFileName = QFileDialog::getOpenFileName(NULL, tr("Open file"), QString(""), tr("Tau Labs Log (*.tll)"));
    if (inputFileName.isEmpty())
        return;

    // Get output file. Suggest to user that output have same base name and location as input file.
    QString filters = tr("Keyhole Markup Language (compressed) (*.kmz);; Keyhole Markup Language (uncompressed) (*.kml)");
    QString outputFileName = QFileDialog::getSaveFileName(NULL, tr("Export log"),
                                inputFileName.split(".",QString::SkipEmptyParts).at(0),
                                filters);

    if (outputFileName.isEmpty())
        return;

    // Create kmlExport instance, and trigger export
    KmlExport kmlExport(inputFileName, outputFileName);
    kmlExport.exportToKML();
}

void KmlExportPlugin::extensionsInitialized()
{
}

void KmlExportPlugin::shutdown()
{
    // Do nothing
}
Q_EXPORT_PLUGIN(KmlExportPlugin)

/**
 * @}
 * @}
 */
