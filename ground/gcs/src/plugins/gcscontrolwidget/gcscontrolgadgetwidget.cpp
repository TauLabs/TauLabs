/**
 ******************************************************************************
 *
 * @file       GCSControlgadgetwidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup GCSControlGadgetPlugin GCSControl Gadget Plugin
 * @{
 * @brief A gadget to control the UAV, either from the keyboard or a joystick
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
#include "gcscontrolgadgetwidget.h"
#include "ui_gcscontrol.h"

#include <QDebug>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>


#include "uavobject.h"
#include "uavobjectmanager.h"
#include "manualcontrolcommand.h"
#include "extensionsystem/pluginmanager.h"
#include "flightstatus.h"

GCSControlGadgetWidget::GCSControlGadgetWidget(QWidget *parent) : QLabel(parent)
{
    m_gcscontrol = new Ui_GCSControl();
    m_gcscontrol->setupUi(this);


    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();

    // Set up the drop down box for the flightmode
    FlightStatus* flightStatus = FlightStatus::GetInstance(objManager);;
    m_gcscontrol->comboBoxFlightMode->addItems(flightStatus->getField("FlightMode")->getOptions());

    // Set up slots and signals for joysticks
    connect(m_gcscontrol->widgetLeftStick,SIGNAL(positionClicked(double,double)),this,SLOT(leftStickClicked(double,double)));
    connect(m_gcscontrol->widgetRightStick,SIGNAL(positionClicked(double,double)),this,SLOT(rightStickClicked(double,double)));

    // Connect misc controls
    connect(m_gcscontrol->checkBoxGcsControl, SIGNAL(clicked(bool)), this, SLOT(toggleControl(bool)));
    connect(m_gcscontrol->comboBoxFlightMode, SIGNAL(currentIndexChanged(int)), this, SLOT(selectFlightMode(int)));

    connect(m_gcscontrol->checkBoxUDPControl, SIGNAL(stateChanged(int)),this,SLOT(toggleUDPControl(int))); //UDP control checkbox

    // Connect object updated event from UAVObject to also update check boxes and dropdown
    connect(flightStatus,SIGNAL(FlightModeChanged(quint8)), this, SLOT(flightModeChanged(quint8)));
    connect(flightStatus,SIGNAL(ArmedChanged(quint8)), this, SLOT(armedChanged(quint8)));

    leftX = 0;
    leftY = 0;
    rightX = 0;
    rightY = 0;
}

void GCSControlGadgetWidget::updateSticks(double nleftX, double nleftY, double nrightX, double nrightY) {
    leftX = nleftX;
    leftY = nleftY;
    rightX = nrightX;
    rightY = nrightY;
    m_gcscontrol->widgetLeftStick->changePosition(leftX,leftY);
    m_gcscontrol->widgetRightStick->changePosition(rightX,rightY);
}

void GCSControlGadgetWidget::leftStickClicked(double X, double Y) {
    leftX = X;
    leftY = Y;
    emit sticksChanged(leftX,leftY,rightX,rightY);
}

void GCSControlGadgetWidget::rightStickClicked(double X, double Y) {
    rightX = X;
    rightY = Y;
    emit sticksChanged(leftX,leftY,rightX,rightY);
}

/*!
  \brief Called when the gcs control is toggled
  */
void GCSControlGadgetWidget::toggleControl(bool checked)
{
    m_gcscontrol->checkBoxUDPControl->setEnabled(checked);
    emit controlEnabled(checked);
}

void GCSControlGadgetWidget::flightModeChanged(quint8 mode)
{
    m_gcscontrol->comboBoxFlightMode->setCurrentIndex(mode);
}

void GCSControlGadgetWidget::armedChanged(quint8 armed)
{
    m_gcscontrol->checkBoxArmed->setChecked(armed != 0);
}

void GCSControlGadgetWidget::toggleUDPControl(int state)
{
    if(state)
    {
        setUDPControl(true);
    }else{
        setUDPControl(false);
    }
}

/*!
  \brief Called when the flight mode drop down is changed and sets the ManualControlCommand->FlightMode accordingly
  */
void GCSControlGadgetWidget::selectFlightMode(int state)
{
    emit flightModeChangedLocaly((ManualControlSettings::FlightModePositionOptions)state);
}

//! Allow the GCS to take over control of UAV
void GCSControlGadgetWidget::allowGcsControl(bool allow)
{
    m_gcscontrol->checkBoxGcsControl->setEnabled(allow);
    if (allow)
        m_gcscontrol->checkBoxGcsControl->setToolTip(
                    tr("Take control of the board from GCS"));
    else
        m_gcscontrol->checkBoxGcsControl->setToolTip(
                    tr("Disabled for safety. Enable this in the options page."));
}

void GCSControlGadgetWidget::setGCSControl(bool newState)
{
    m_gcscontrol->checkBoxGcsControl->setChecked(newState);
}
bool GCSControlGadgetWidget::getGCSControl(void)
{
    return m_gcscontrol->checkBoxGcsControl->isChecked();
}

void GCSControlGadgetWidget::setUDPControl(bool newState)
{
    m_gcscontrol->checkBoxUDPControl->setChecked(newState);
}

bool GCSControlGadgetWidget::getUDPControl(void)
{
    return m_gcscontrol->checkBoxUDPControl->isChecked();
}


/**
  * @}
  * @}
  */

