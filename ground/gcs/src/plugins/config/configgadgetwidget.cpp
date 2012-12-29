/**
 ******************************************************************************
 *
 * @file       configgadgetwidget.cpp
 * @author     E. Lafargue & The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief The Configuration Gadget used to update settings in the firmware
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

#include "configgadgetwidget.h"

#include "configvehicletypewidget.h"
#include "configccattitudewidget.h"
#include "configinputwidget.h"
#include "configoutputwidget.h"
#include "configstabilizationwidget.h"
#include "configautotunewidget.h"
#include "configcamerastabilizationwidget.h"
#include "configtxpidwidget.h"
#include "config_cc_hw_widget.h"
#include "configpipxtremewidget.h"
#include "configrevowidget.h"
#include "defaultattitudewidget.h"
#include "defaulthwsettingswidget.h"
#include "uavobjectutilmanager.h"

#include <QDebug>
#include <QStringList>
#include <QtGui/QWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QPushButton>



ConfigGadgetWidget::ConfigGadgetWidget(QWidget *parent) : QWidget(parent)
{
    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    ftw = new MyTabbedStackWidget(this, true, true);
    ftw->setIconSize(64);

    QVBoxLayout *layout = new QVBoxLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(ftw);
    setLayout(layout);

    // *********************
    QWidget *qwd;

    QIcon *icon = new QIcon();
    icon->addFile(":/configgadget/images/hardware_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/hardware_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    qwd = new DefaultHwSettingsWidget(this);
    ftw->insertTab(ConfigGadgetWidget::hardware, qwd, *icon, QString("Hardware"));

    icon = new QIcon();
    icon->addFile(":/configgadget/images/vehicle_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/vehicle_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    qwd = new ConfigVehicleTypeWidget(this);
    ftw->insertTab(ConfigGadgetWidget::aircraft, qwd, *icon, QString("Vehicle"));

    icon = new QIcon();
    icon->addFile(":/configgadget/images/input_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/input_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    qwd = new ConfigInputWidget(this);
    ftw->insertTab(ConfigGadgetWidget::input, qwd, *icon, QString("Input"));

    icon = new QIcon();
    icon->addFile(":/configgadget/images/output_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/output_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    qwd = new ConfigOutputWidget(this);
    ftw->insertTab(ConfigGadgetWidget::output, qwd, *icon, QString("Output"));

    icon = new QIcon();
    icon->addFile(":/configgadget/images/ins_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/ins_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    qwd = new DefaultAttitudeWidget(this);
    ftw->insertTab(ConfigGadgetWidget::sensors, qwd, *icon, QString("INS"));

    icon = new QIcon();
    icon->addFile(":/configgadget/images/stabilization_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/stabilization_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    qwd = new ConfigStabilizationWidget(this);
    ftw->insertTab(ConfigGadgetWidget::stabilization, qwd, *icon, QString("Stabilization"));

    icon = new QIcon();
    icon->addFile(":/configgadget/images/autotune_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/autotune_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    qwd = new ConfigAutotuneWidget(this);
    ftw->insertTab(ConfigGadgetWidget::autotune, qwd, *icon, QString("Autotune"));

    icon = new QIcon();
    icon->addFile(":/configgadget/images/camstab_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/camstab_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    qwd = new ConfigCameraStabilizationWidget(this);
    ftw->insertTab(ConfigGadgetWidget::camerastabilization, qwd, *icon, QString("Camera Stab"));

    icon = new QIcon();
    icon->addFile(":/configgadget/images/txpid_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/txpid_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    qwd = new ConfigTxPIDWidget(this);
    ftw->insertTab(ConfigGadgetWidget::txpid, qwd, *icon, QString("TxPID"));

    ftw->setCurrentIndex(ConfigGadgetWidget::hardware);
    // *********************
    // Listen to autopilot connection events

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    TelemetryManager* telMngr = pm->getObject<TelemetryManager>();
    connect(telMngr, SIGNAL(connected()), this, SLOT(onAutopilotConnect()));
    connect(telMngr, SIGNAL(disconnected()), this, SLOT(onAutopilotDisconnect()));

    // And check whether by any chance we are not already connected
    if (telMngr->isConnected()) {
        onAutopilotConnect();    
    }

    help = 0;
    connect(ftw,SIGNAL(currentAboutToShow(int,bool*)),this,SLOT(tabAboutToChange(int,bool*)));//,Qt::BlockingQueuedConnection);

    // Connect to the PipXStatus object updates
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    oplinkStatusObj = dynamic_cast<UAVDataObject*>(objManager->getObject("OPLinkStatus"));
    if (oplinkStatusObj != NULL ) {
        connect(oplinkStatusObj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(updateOPLinkStatus(UAVObject*)));
    } else {
	qDebug() << "Error: Object is unknown (OPLinkStatus).";
    }

    // Create the timer that is used to timeout the connection to the OPLink.
    oplinkTimeout = new QTimer(this);
    connect(oplinkTimeout, SIGNAL(timeout()), this, SLOT(onOPLinkDisconnect()));
    oplinkConnected = false;
}

ConfigGadgetWidget::~ConfigGadgetWidget()
{
    // TODO: properly delete all the tabs in ftw before exiting
}

void ConfigGadgetWidget::startInputWizard()
{
    ftw->setCurrentIndex(ConfigGadgetWidget::input);
    ConfigInputWidget* inputWidget = dynamic_cast<ConfigInputWidget*>(ftw->getWidget(ConfigGadgetWidget::input));
    Q_ASSERT(inputWidget);
    inputWidget->startInputWizard();
}

void ConfigGadgetWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
}

void ConfigGadgetWidget::onAutopilotDisconnect() {
    ftw->setCurrentIndex(ConfigGadgetWidget::hardware);
    ftw->removeTab(ConfigGadgetWidget::sensors);

    QIcon *icon = new QIcon();
    icon->addFile(":/configgadget/images/ins_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/ins_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    QWidget *qwd = new DefaultAttitudeWidget(this);
    ftw->insertTab(ConfigGadgetWidget::sensors, qwd, *icon, QString("INS"));
    ftw->removeTab(ConfigGadgetWidget::hardware);

    icon = new QIcon();
    icon->addFile(":/configgadget/images/hardware_normal.png", QSize(), QIcon::Normal, QIcon::Off);
    icon->addFile(":/configgadget/images/hardware_selected.png", QSize(), QIcon::Selected, QIcon::Off);
    qwd = new DefaultHwSettingsWidget(this);
    ftw->insertTab(ConfigGadgetWidget::hardware, qwd, *icon, QString("Hardware"));
    ftw->setCurrentIndex(ConfigGadgetWidget::hardware);

    emit autopilotDisconnected();
}

void ConfigGadgetWidget::onAutopilotConnect() {

    qDebug()<<"ConfigGadgetWidget onAutopilotConnect";
    // First of all, check what Board type we are talking to, and
    // if necessary, remove/add tabs in the config gadget:
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectUtilManager* utilMngr = pm->getObject<UAVObjectUtilManager>();
    if (utilMngr) {
        int board = utilMngr->getBoardModel();
        if ((board & 0xff00) == 1024) {
            // CopterControl family
            // Delete the INS panel, replace with CC Panel:
            ftw->setCurrentIndex(ConfigGadgetWidget::hardware);
            ftw->removeTab(ConfigGadgetWidget::sensors);

            QIcon *icon = new QIcon();
            icon->addFile(":/configgadget/images/ins_normal.png", QSize(), QIcon::Normal, QIcon::Off);
            icon->addFile(":/configgadget/images/ins_selected.png", QSize(), QIcon::Selected, QIcon::Off);
            QWidget *qwd = new ConfigCCAttitudeWidget(this);
            ftw->insertTab(ConfigGadgetWidget::sensors, qwd, *icon, QString("Attitude"));
            ftw->removeTab(ConfigGadgetWidget::hardware);

            icon = new QIcon();
            icon->addFile(":/configgadget/images/hardware_normal.png", QSize(), QIcon::Normal, QIcon::Off);
            icon->addFile(":/configgadget/images/hardware_selected.png", QSize(), QIcon::Selected, QIcon::Off);
            qwd = new ConfigCCHWWidget(this);
            ftw->insertTab(ConfigGadgetWidget::hardware, qwd, *icon, QString("Hardware"));
            ftw->setCurrentIndex(ConfigGadgetWidget::hardware);
        } else if ((board & 0xff00) == 0x0900) {
            // Revolution sensor calibration
            ftw->setCurrentIndex(ConfigGadgetWidget::hardware);
            ftw->removeTab(ConfigGadgetWidget::sensors);

            QIcon *icon = new QIcon();
            icon->addFile(":/configgadget/images/ins_normal.png", QSize(), QIcon::Normal, QIcon::Off);
            icon->addFile(":/configgadget/images/ins_selected.png", QSize(), QIcon::Selected, QIcon::Off);
            QWidget *qwd = new ConfigRevoWidget(this);
            ftw->insertTab(ConfigGadgetWidget::sensors, qwd, *icon, QString("INS"));
            ftw->removeTab(ConfigGadgetWidget::hardware);

            icon = new QIcon();
            icon->addFile(":/configgadget/images/hardware_normal.png", QSize(), QIcon::Normal, QIcon::Off);
            icon->addFile(":/configgadget/images/hardware_selected.png", QSize(), QIcon::Normal, QIcon::On);
            qwd = new DefaultHwSettingsWidget(this);
            ftw->insertTab(ConfigGadgetWidget::hardware, qwd, *icon, QString("Hardware"));
            ftw->setCurrentIndex(ConfigGadgetWidget::hardware);
        } else {
            //Unknown board
            qDebug() << "Unknown board " << board;
        }
    }
    emit autopilotConnected();
}

void ConfigGadgetWidget::tabAboutToChange(int i, bool * proceed)
{
    Q_UNUSED(i);
    *proceed = true;
    ConfigTaskWidget * wid = qobject_cast<ConfigTaskWidget *>(ftw->currentWidget());
    if(!wid) {
        return;
    }
    if(wid->isDirty())
    {
        int ans=QMessageBox::warning(this,tr("Unsaved changes"),tr("The tab you are leaving has unsaved changes,"
                                                           "if you proceed they will be lost."
                                                           "Do you still want to proceed?"),QMessageBox::Yes,QMessageBox::No);
        if(ans==QMessageBox::No) {
            *proceed=false;
        } else {
            wid->setDirty(false);
        }
    }
}

/*!
  \brief Called by updates to @OPLinkStatus
  */
void ConfigGadgetWidget::updateOPLinkStatus(UAVObject *)
{
    // Restart the disconnection timer.
    oplinkTimeout->start(5000);
    if (!oplinkConnected)
    {
        qDebug() << "ConfigGadgetWidget onOPLinkConnect";

        QIcon *icon = new QIcon();
        icon->addFile(":/configgadget/images/pipx-normal.png", QSize(), QIcon::Normal, QIcon::Off);
        icon->addFile(":/configgadget/images/pipx-selected.png", QSize(), QIcon::Selected, QIcon::Off);

        QWidget *qwd = new ConfigPipXtremeWidget(this);
        ftw->insertTab(ConfigGadgetWidget::oplink, qwd, *icon, QString("OPLink"));
        oplinkConnected = true;
    }
}

void ConfigGadgetWidget::onOPLinkDisconnect() {
	qDebug()<<"ConfigGadgetWidget onOPLinkDisconnect";
	oplinkTimeout->stop();
	ftw->removeTab(ConfigGadgetWidget::oplink);
	oplinkConnected = false;
}
