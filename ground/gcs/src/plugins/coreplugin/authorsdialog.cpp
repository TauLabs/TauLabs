/**
 ******************************************************************************
 *
 * @file       authorsdialog.cpp
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

#include "authorsdialog.h"

#include "coreconstants.h"
#include "icore.h"

#include <utils/qtcassert.h>

#include <QtCore/QDate>
#include <QtCore/QFile>
#include <QtCore/QSysInfo>

#include <QtGui/QDialogButtonBox>
#include <QtGui/QGridLayout>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QTextBrowser>

using namespace Core;
using namespace Core::Internal;
using namespace Core::Constants;

AuthorsDialog::AuthorsDialog(QWidget *parent)
    : QDialog(parent)
{
    // We need to set the window icon explicitly here since for some reason the
    // application icon isn't used when the size of the dialog is fixed (at least not on X11/GNOME)
    setWindowIcon(QIcon(":/core/images/taulabs_logo_32.png"));

    setWindowTitle(tr("About Tau Labs Authors"));
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
    QGridLayout *layout = new QGridLayout(this);
    layout->setSizeConstraint(QLayout::SetFixedSize);

    QString version = QLatin1String(GCS_VERSION_LONG);
    version += QDate(2007, 25, 10).toString(Qt::SystemLocaleDate);

#ifdef GCS_REVISION
     //: This gets conditionally inserted as argument %8 into the description string.
     ideRev = tr("From revision %1<br/>").arg(QString::fromLatin1(GCS_REVISION_STR).left(10));
#endif

     const QString description = tr(
        "Proudly brought to you by this fine team:<br/>"
        );

    QLabel *copyRightLabel = new QLabel(description);
    copyRightLabel->setWordWrap(true);
    copyRightLabel->setOpenExternalLinks(true);
    copyRightLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);

    QTextBrowser *creditsArea = new QTextBrowser(this);
    creditsArea->setSource(QUrl("qrc:core/CREDITS.html"));

    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Close);
    QPushButton *closeButton = buttonBox->button(QDialogButtonBox::Close);
    QTC_ASSERT(closeButton, /**/);
    buttonBox->addButton(closeButton, QDialogButtonBox::ButtonRole(QDialogButtonBox::RejectRole | QDialogButtonBox::AcceptRole));
    connect(buttonBox , SIGNAL(rejected()), this, SLOT(reject()));

    QLabel *logoLabel = new QLabel;
    logoLabel->setPixmap(QPixmap(QLatin1String(":/core/images/taulabs_logo_128.png")));
    layout->addWidget(logoLabel ,     0, 0, 1, 1);
    layout->addWidget(copyRightLabel, 0, 1, 2, 4);
    layout->addWidget(creditsArea,    3, 0, 2, 5);
    layout->addWidget(buttonBox,      6, 0, 1, 5);
}
