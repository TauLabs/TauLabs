/**
 ******************************************************************************
 *
 * @file       mocapwidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup MoCapPlugin Motion Capture Plugin
 * @{
 * @brief The Hardware In The Loop plugin 
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
#include "mocapwidget.h"
#include "ui_mocapwidget.h"
#include "qxtlogger.h"
#include <QDebug>
#include <QFile>
#include <QDir>
#include <QDateTime>
#include <QThread>

#include <mocapplugin.h>
#include "export.h"
#include "uavobjectmanager.h"
#include "coreplugin/icore.h"
#include "coreplugin/threadmanager.h"


QStringList Export::instances;

MoCapWidget::MoCapWidget(QWidget *parent)
	: QWidget(parent),
    exporter(0)
{
    widget = new Ui_MoCapWidget();
	widget->setupUi(this);
	widget->startButton->setEnabled(true);
	widget->stopButton->setEnabled(false);

    greenColor = "rgb(35, 221, 35)"; //Change the green color in order to make it a bit more vibrant
    strStyleEnable = QString("QFrame{background-color: %1; color: white}").arg(greenColor);
    strStyleDisable = "QFrame{background-color: red; color: white}";

    strAutopilotDisconnected = " Autopilot OFF ";
    strExportDisconnected = " Export OFF ";
    strAutopilotConnected = " Autopilot ON ";

	widget->apLabel->setText(strAutopilotDisconnected);
    widget->mocapLabel->setText(strExportDisconnected);

	connect(widget->startButton, SIGNAL(clicked()), this, SLOT(startButtonClicked()));
	connect(widget->stopButton, SIGNAL(clicked()), this, SLOT(stopButtonClicked()));
	connect(widget->buttonClearLog, SIGNAL(clicked()), this, SLOT(buttonClearLogClicked()));
}

MoCapWidget::~MoCapWidget()
{
   delete widget;
}

void MoCapWidget::startButtonClicked()
{
        QThread* mainThread = QThread::currentThread();
	qDebug() << "Main Thread: "<< mainThread;

    //Allow only one instance per export
    if(Export::Instances().indexOf(settings.exportId) != -1)
	{
        widget->textBrowser->append(settings.exportId + " alreary started!");
		return;
	}

    if(!MoCapPlugin::typeMocaps.size())
	{
        qxtLog->info("There is no registered exports, add through MoCapPlugin::addExport");
		return;
	}

	// Stop running process if one is active
    if(exporter)
	{
        QMetaObject::invokeMethod(exporter, "onDeleteExport",Qt::QueuedConnection);
        exporter = NULL;
	}

    if(settings.hostAddress == "" || settings.inPort == 0)
	{
		widget->textBrowser->append("Before start, set UDP parameters in options page!");
		return;
	}

    MocapCreator* creator = MoCapPlugin::getMocapCreator(settings.exportId);
    exporter = creator->createExport(settings);

    // move to thread <--[BCH]
    exporter->setName(creator->Description());
    exporter->setExportId(creator->ClassId());

    connect(exporter, SIGNAL(processOutput(QString)), this, SLOT(onProcessOutput(QString)));

	// Setup process
	widget->textBrowser->append(QString("[%1] Starting %2... ").arg(QTime::currentTime().toString("hh:mm:ss")).arg(creator->Description()));
    qxtLog->info("MoCap: Starting " + creator->Description());

	// Start bridge
    bool ret = QMetaObject::invokeMethod(exporter, "setupProcess",Qt::QueuedConnection);
	if(ret)
	{
        Export::setInstance(settings.exportId);

        connect(this,SIGNAL(deleteExport()),exporter, SLOT(onDeleteExport()),Qt::QueuedConnection);

		widget->startButton->setEnabled(false);
		widget->stopButton->setEnabled(true);
        qxtLog->info("MoCap: Starting bridge, initializing flight export and Autopilot connections");

        connect(exporter, SIGNAL(autopilotConnected()), this, SLOT(onAutopilotConnect()),Qt::QueuedConnection);
        connect(exporter, SIGNAL(autopilotDisconnected()), this, SLOT(onAutopilotDisconnect()),Qt::QueuedConnection);
        connect(exporter, SIGNAL(exportConnected()), this, SLOT(onExportConnect()),Qt::QueuedConnection);
        connect(exporter, SIGNAL(exportDisconnected()), this, SLOT(onExportDisconnect()),Qt::QueuedConnection);

		// Initialize connection status
        if ( exporter->isAutopilotConnected() )
		{
			onAutopilotConnect();
		}
		else
		{
			onAutopilotDisconnect();
		}

        if ( exporter->isExportConnected() )
		{
            onExportConnect();
		}
		else
		{
            onExportDisconnect();
		}
	}
}


void MoCapWidget::stopButtonClicked()
{
    if(exporter)
        widget->textBrowser->append(QString("[%1] Terminate %2 ").arg(QTime::currentTime().toString("hh:mm:ss")).arg(exporter->Name()));

	widget->startButton->setEnabled(true);
	widget->stopButton->setEnabled(false);
    widget->apLabel->setStyleSheet(QString::fromUtf8("QFrame{background-color: transparent; color: white}"));
    widget->mocapLabel->setStyleSheet(QString::fromUtf8("QFrame{background-color: transparent; color: white}"));
    widget->apLabel->setText(strAutopilotDisconnected);
    widget->mocapLabel->setText(strExportDisconnected);
    if(exporter)
	{
        QMetaObject::invokeMethod(exporter, "onDeleteExport",Qt::QueuedConnection);
        exporter = NULL;
	}
}

void MoCapWidget::buttonClearLogClicked()
{
	widget->textBrowser->clear();
}

void MoCapWidget::onProcessOutput(QString text)
{
    widget->textBrowser->append(text);
}

void MoCapWidget::onAutopilotConnect()
{
    widget->apLabel->setStyleSheet(strStyleEnable);
    widget->apLabel->setText(strAutopilotConnected);
    qxtLog->info("MoCap: Motion capture connected.");
}

void MoCapWidget::onAutopilotDisconnect()
{
    widget->apLabel->setStyleSheet(strStyleDisable);
	widget->apLabel->setText(strAutopilotDisconnected);
	qxtLog->info(strAutopilotDisconnected);
}

void MoCapWidget::onExportConnect()
{
    widget->mocapLabel->setStyleSheet(strStyleEnable);
    widget->mocapLabel->setText(" " + exporter->Name() +" connected ");
    qxtLog->info(QString("MoCap: %1 connected").arg(exporter->Name()));
}

void MoCapWidget::onExportDisconnect()
{
    widget->mocapLabel->setStyleSheet(strStyleDisable);
    widget->mocapLabel->setText(" " + exporter->Name() +" disconnected ");
    qxtLog->info(QString("MoCap: %1 disconnected").arg(exporter->Name()));
}
