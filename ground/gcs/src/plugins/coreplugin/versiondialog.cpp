/**
 ******************************************************************************
 *
 * @file       versiondialog.cpp
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

#include "versiondialog.h"

#include "coreconstants.h"
#include "icore.h"

#include <utils/qtcassert.h>

#include <QtCore/QDate>
#include <QtCore/QFile>
#include <QtCore/QSysInfo>

#include <QDialogButtonBox>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QTextBrowser>
#include <QApplication>

using namespace Core;
using namespace Core::Internal;
using namespace Core::Constants;

VersionDialog::VersionDialog(QWidget *parent)
    : QDialog(parent)
{
    // We need to set the window icon explicitly here since for some reason the
    // application icon isn't used when the size of the dialog is fixed (at least not on X11/GNOME)
    setWindowIcon(QIcon(":/core/images/taulabs_logo_32.png"));

    setWindowTitle(tr("About Tau Labs GCS"));
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
    QGridLayout *layout = new QGridLayout(this);
    layout->setSizeConstraint(QLayout::SetFixedSize);
    QString versionName;
    QString versionHash;

    QString versionData = QLatin1String(GCS_REVISION_PRETTY_STR);
    versionName = versionData.split("%@%").at(0);
    versionHash = versionData.split("%@%").at(1);

    QString ideRev;
#ifdef GCS_REVISION
     //: This gets conditionally inserted as argument %8 into the description string.
     QString revision = QString::fromLatin1(GCS_REVISION_STR).remove(0, 1+(QString::fromLatin1(GCS_REVISION_STR).indexOf(":")));
     ideRev = tr("From revision %1<br/>").arg(revision);
#endif
     QString uavoHashStr;
 #ifdef UAVO_HASH
      //: This gets conditionally inserted as argument %11 into the description string.
     QByteArray uavoHashArray;
     QString uavoHash = QString::fromLatin1(Core::Constants::UAVOSHA1_STR);
     uavoHash.chop(2);
     uavoHash.remove(0,2);
     uavoHash=uavoHash.trimmed();
     bool ok;
     foreach(QString str,uavoHash.split(","))
     {
         uavoHashArray.append(str.toInt(&ok,16));
     }
     QString gcsUavoHashStr;
     foreach(char i, uavoHashArray)
     {
         gcsUavoHashStr.append(QString::number(i,16).right(2));
     }
     uavoHashStr = tr("UAVO hash %1<br/>").arg(gcsUavoHashStr.left(8));
 #endif
     const QString version_name = tr("<h3><center>Tau Labs GCS<center></h3>"
                                     "<h4><center>%1: %2</center></h4>").arg(versionName, versionHash);
     const QString version_description = tr(
        "Based on Qt %1 (%2 bit)<br/>"
        "<br/>"
        "Built on %3 at %4<br />"
        "<br/>"
        "%5"
        "<br/>"
        "%6"
        "<br/>").arg(QLatin1String(QT_VERSION_STR), QString::number(QSysInfo::WordSize),
                     QLatin1String(__DATE__), QLatin1String(__TIME__), ideRev, uavoHashStr);

     QString copyright = tr(
        "Copyright 2012-%1 %2, 2010-2012 OpenPilot. All rights reserved.<br/>"
        "<br/>"
         "Between 2010 and 2012, a significant part of this application was designed "
         "and implemented within the OpenPilot project.<br/>"
         "This work was further based on work from the Nokia Corporation for Qt Creator.<br/>"
         "<br/>"
         "<small>This program is free software; you can redistribute it and/or modify"
         "it under the terms of the GNU General Public License as published by"
         "the Free Software Foundation; either version 3 of the License, or"
         "(at your option) any later version.<br/><br/>"
        "The program is provided AS IS with NO WARRANTY OF ANY KIND, "
        "INCLUDING THE WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A "
        "PARTICULAR PURPOSE.</small><br/>").arg(QLatin1String(GCS_YEAR), (QLatin1String(GCS_AUTHOR)));

    QLabel *versionNameLabel = new QLabel(version_name);
    QLabel *versionDescription = new QLabel(version_description);
    versionDescription->setWordWrap(true);
    versionDescription->setOpenExternalLinks(true);
    versionDescription->setTextInteractionFlags(Qt::TextBrowserInteraction);
    versionNameLabel->setWordWrap(true);
    versionNameLabel->setOpenExternalLinks(true);
    versionNameLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);

    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Close);
    QPushButton *closeButton = buttonBox->button(QDialogButtonBox::Close);
    QTC_ASSERT(closeButton, /**/);
    buttonBox->addButton(closeButton, QDialogButtonBox::ButtonRole(QDialogButtonBox::RejectRole | QDialogButtonBox::AcceptRole));
    connect(buttonBox , SIGNAL(rejected()), this, SLOT(reject()));

    QLabel *logoLabel = new QLabel;
    logoLabel->setPixmap(QPixmap(QLatin1String(":/core/images/taulabs_logo_128.png")));

    QLabel *copyRightLabel = new QLabel(copyright);
    copyRightLabel->setWordWrap(true);
    copyRightLabel->setOpenExternalLinks(true);
    copyRightLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);

    layout->addWidget(versionNameLabel , 0, 0, 1, 2);
    layout->addWidget(logoLabel , 1, 1, 1, 1);
    layout->addWidget(versionDescription , 1, 0, 1, 1);
    layout->addWidget(copyRightLabel, 3, 0, 1, 2);
    layout->addWidget(buttonBox, 5, 0, 1, 2);
}
