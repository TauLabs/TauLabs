/**
 ******************************************************************************
 *
 * @file       debuggadgetwidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup DebugGadgetPlugin Debug Gadget Plugin
 * @{
 * @brief A place holder gadget plugin
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
#include "debuggadgetwidget.h"

#include <QDebug>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include "debugengine.h"
#include <QFile>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollBar>
#include <QTime>

DebugGadgetWidget::DebugGadgetWidget(QWidget *parent) : QLabel(parent)
{
    m_config = new Ui_Form();
    m_config->setupUi(this);
    debugengine *de = debugengine::getInstance();

    connect(de, SIGNAL(debug(QString)), this, SLOT(dbgMsgDebug(QString)),Qt::QueuedConnection);
    connect(de, SIGNAL(warning(QString)), this, SLOT(dbgMsgWarning(QString)),Qt::QueuedConnection);
    connect(de, SIGNAL(critical(QString)), this, SLOT(dbgMsgCritical(QString)),Qt::QueuedConnection);
    connect(de, SIGNAL(fatal(QString)), this, SLOT(dbgMsgFatal(QString)),Qt::QueuedConnection);
    connect(m_config->saveToFile, SIGNAL(clicked()), this, SLOT(saveLog()));
    connect(m_config->clearLog, SIGNAL(clicked()), this, SLOT(clearLog()));
}

DebugGadgetWidget::~DebugGadgetWidget()
{
    // Do nothing
}

void DebugGadgetWidget::dbgMsgDebug(QString msg)
{
    m_config->plainTextEdit->setTextColor(Qt::blue);

    m_config->plainTextEdit->append(QString("%0[DEBUG]%1").arg(QTime::currentTime().toString()).arg(msg));

    QScrollBar *sb = m_config->plainTextEdit->verticalScrollBar();
    sb->setValue(sb->maximum());
}

void DebugGadgetWidget::dbgMsgWarning(QString msg)
{
    m_config->plainTextEdit->setTextColor(Qt::red);

    m_config->plainTextEdit->append(QString("%0[WARNING]%1").arg(QTime::currentTime().toString()).arg(msg));

    QScrollBar *sb = m_config->plainTextEdit->verticalScrollBar();
    sb->setValue(sb->maximum());
}

void DebugGadgetWidget::dbgMsgCritical(QString msg)
{
    m_config->plainTextEdit->setTextColor(Qt::red);

    m_config->plainTextEdit->append(QString("%0[CRITICAL]%1").arg(QTime::currentTime().toString()).arg(msg));

    QScrollBar *sb = m_config->plainTextEdit->verticalScrollBar();
    sb->setValue(sb->maximum());
}

void DebugGadgetWidget::dbgMsgFatal(QString msg)
{
    m_config->plainTextEdit->setTextColor(Qt::red);

    m_config->plainTextEdit->append(QString("%0[FATAL]%1").arg(QTime::currentTime().toString()).arg(msg));

    QScrollBar *sb = m_config->plainTextEdit->verticalScrollBar();
    sb->setValue(sb->maximum());
}

void DebugGadgetWidget::saveLog()
{
    QString fileName = QFileDialog::getSaveFileName(0, tr("Save log File As"), "");

    if (fileName.isEmpty()) {
        return;
    }

    QFile file(fileName);
    if (file.open(QIODevice::WriteOnly) &&
        (file.write(m_config->plainTextEdit->toHtml().toLatin1()) != -1)) {
        file.close();
    } else {
        QMessageBox::critical(0,
                              tr("Log Save"),
                              tr("Unable to save log: ") + fileName,
                              QMessageBox::Ok);
        return;
    }
}

void DebugGadgetWidget::clearLog()
{
    m_config->plainTextEdit->clear();
}
