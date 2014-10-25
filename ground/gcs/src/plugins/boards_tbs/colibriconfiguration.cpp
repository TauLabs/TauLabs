/**
 ******************************************************************************
 *
 * @file       colibriconfiguration.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup Boards_TBS TBS boards support Plugin
 * @{
 * @brief Plugin to support boards by TBS
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

#include "smartsavebutton.h"
#include "colibriconfiguration.h"
#include "ui_colibriconfiguration.h"

#include "hwcolibri.h"

ColibriConfiguration::ColibriConfiguration(QWidget *parent) :
    ConfigTaskWidget(parent),
    ui(new Ui::ColibriConfiguration)
{
    ui->setupUi(this);

    addApplySaveButtons(ui->applySettings,ui->saveSettings);
    addUAVObjectToWidgetRelation("HwColibri","Frame",ui->cmbFrame);
    addUAVObjectToWidgetRelation("HwColibri","Uart2",ui->cmbUart2);
    addUAVObjectToWidgetRelation("HwColibri","Uart1",ui->cmbUart1);
    addUAVObjectToWidgetRelation("HwColibri","RcvrPort",ui->cmbRcvrPort);

    // Load UAVObjects to widget relations from UI file
    // using objrelation dynamic property
    autoLoadWidgets();

    connect(ui->help,SIGNAL(clicked()),this,SLOT(openHelp()));
    //enableControls(false);
    enableControls(true);
    populateWidgets();
    refreshWidgetsValues();
    forceConnectedState();
}

ColibriConfiguration::~ColibriConfiguration()
{
    delete ui;
}

void ColibriConfiguration::refreshValues()
{
}

/**
 * @brief ColibriConfiguration::widgetsContentsChanged
 * Verify that the configuration is valid for the frame type
 * selected
 */
void ColibriConfiguration::widgetsContentsChanged()
{

    ConfigTaskWidget::widgetsContentsChanged();

    bool valid = true;

    switch (ui->cmbFrame->currentIndex()) {
    case HwColibri::FRAME_GEMINI:

        frame = QPixmap(":/TBS/images/gemini.png");
        resizeEvent(NULL);

        // There are a few relevant points about the Gemini mainboard:
        // - it does not route out UART3/4
        // - it shares the PPM input pin with the UART2 RX pin
        // thus we have to ensure that only valid configurations
        // are used

        bool ppm_used = ui->cmbRcvrPort->currentIndex() == HwColibri::RCVRPORT_PPM ||
                ui->cmbRcvrPort->currentIndex() == HwColibri::RCVRPORT_PPMPWM ||
                ui->cmbRcvrPort->currentIndex() == HwColibri::RCVRPORT_PPMPWMADC ||
                ui->cmbRcvrPort->currentIndex() == HwColibri::RCVRPORT_PPMOUTPUTS ||
                ui->cmbRcvrPort->currentIndex() == HwColibri::RCVRPORT_PPMOUTPUTSADC;

        switch(ui->cmbUart2->currentIndex()) {
        case HwColibri::UART2_DEBUGCONSOLE:
        case HwColibri::UART2_COMBRIDGE:
        case HwColibri::UART2_MAVLINKTX:
        case HwColibri::UART2_MAVLINKTX_GPS_RX:
        case HwColibri::UART2_HOTTTELEMETRY:
        case HwColibri::UART2_FRSKYSENSORHUB:
        case HwColibri::UART2_LIGHTTELEMETRYTX:
            ui->lblConfigMessage->setText(QString(tr("Warning: Please do not enable options that require transmitting with UART2 on Gemini.")));
            valid = false;
            break;
        case HwColibri::UART2_DISABLED:
            // always valid;
            break;
        default:
            // Receiving with UART2 valid when PPM pin is not used
            valid = !ppm_used;
            if (ppm_used) {
                ui->lblConfigMessage->setText(QString(tr("Warning: Please do not UART2 and PPM at the same time.")));
                valid = false;
            }
        }

        break;
    }

    if (valid)
        ui->lblConfigMessage->setText(QString(tr("Configuration OK")));

    enableControls(valid);
}

void ColibriConfiguration::resizeEvent(QResizeEvent *event)
{
    Q_UNUSED(event);
    int w = ui->lblImage->width();
    int h = ui->lblImage->height();
    ui->lblImage->setPixmap(frame.scaled(w,h,Qt::KeepAspectRatio));
}

void ColibriConfiguration::openHelp()
{
    QDesktopServices::openUrl( QUrl("http://wiki.taulabs.org/OnlineHelp:-Hardware-Settings", QUrl::StrictMode) );
}
