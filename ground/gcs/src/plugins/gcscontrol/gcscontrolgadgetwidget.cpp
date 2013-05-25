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
#include <QtGui/QWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QPushButton>


#include "uavobject.h"
#include "uavobjectmanager.h"
#include "manualcontrolcommand.h"
#include "extensionsystem/pluginmanager.h"

GCSControlGadgetWidget::GCSControlGadgetWidget(QWidget *parent) : QLabel(parent)
{
    m_gcscontrol = new Ui_GCSControl();
    m_gcscontrol->setupUi(this);


    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* obj = dynamic_cast<UAVDataObject*>( objManager->getObject(QString("ManualControlCommand")) );
    UAVObject::Metadata mdata = obj->getMetadata();
    m_gcscontrol->checkBoxGcsControl->setChecked(UAVObject::GetFlightAccess(mdata) == UAVObject::ACCESS_READONLY);

    // Set up the drop down box for the flightmode
    UAVDataObject* flightStatus = dynamic_cast<UAVDataObject*>( objManager->getObject(QString("FlightStatus")) );
    m_gcscontrol->comboBoxFlightMode->addItems(flightStatus->getField("FlightMode")->getOptions());

    // Set up slots and signals for joysticks
    connect(m_gcscontrol->widgetLeftStick,SIGNAL(positionClicked(double,double)),this,SLOT(leftStickClicked(double,double)));
    connect(m_gcscontrol->widgetRightStick,SIGNAL(positionClicked(double,double)),this,SLOT(rightStickClicked(double,double)));

    // Connect misc controls
    connect(m_gcscontrol->checkBoxGcsControl, SIGNAL(stateChanged(int)), this, SLOT(toggleControl(int)));
    connect(m_gcscontrol->checkBoxArmed, SIGNAL(stateChanged(int)), this, SLOT(toggleArmed(int)));
    connect(m_gcscontrol->comboBoxFlightMode, SIGNAL(currentIndexChanged(int)), this, SLOT(selectFlightMode(int)));

    connect(m_gcscontrol->checkBoxUDPControl, SIGNAL(stateChanged(int)),this,SLOT(toggleUDPControl(int))); //UDP control checkbox

    // Connect object updated event from UAVObject to also update check boxes and dropdown
    connect(obj, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(mccChanged(UAVObject*)));




    leftX = 0;
    leftY = 0;
    rightX = 0;
    rightY = 0;

    // No point enabling OpenGL for the joysticks, and causes
    // issues on some computers:
//    m_gcscontrol->widgetLeftStick->enableOpenGL(true);
//    m_gcscontrol->widgetRightStick->enableOpenGL(true);
}

GCSControlGadgetWidget::~GCSControlGadgetWidget()
{
   // Do nothing
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
  \brief Called when the gcs control is toggled and enabled or disables flight write access to manual control command
  */
void GCSControlGadgetWidget::toggleControl(int state)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* obj = dynamic_cast<UAVDataObject*>( objManager->getObject(QString("ManualControlCommand")) );

    UAVObject::Metadata mdata = obj->getMetadata();
    if (state)
    {
        mccInitialData = mdata;
        UAVObject::SetFlightAccess(mdata, UAVObject::ACCESS_READONLY);
        UAVObject::SetFlightTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
        UAVObject::SetGcsTelemetryAcked(mdata, false);
        UAVObject::SetGcsTelemetryUpdateMode(mdata, UAVObject::UPDATEMODE_ONCHANGE);
        mdata.gcsTelemetryUpdatePeriod = 100;
        m_gcscontrol->checkBoxUDPControl->setEnabled(true);

    }
    else
    {
        mdata = mccInitialData;
        toggleUDPControl(false);
        m_gcscontrol->checkBoxUDPControl->setEnabled(false);
    }
    obj->setMetadata(mdata);
}

void GCSControlGadgetWidget::toggleArmed(int state)
{
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* obj = dynamic_cast<UAVDataObject*>( objManager->getObject(QString("FlightStatus")) );
    if(state)
        obj->getField("Armed")->setValue("Armed");
    else
        obj->getField("Armed")->setValue("Disarmed");
    obj->updated();
}

void GCSControlGadgetWidget::mccChanged(UAVObject * obj)
{
    Q_UNUSED(obj);

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* flightStatus = dynamic_cast<UAVDataObject*>( objManager->getObject(QString("FlightStatus")) );
    m_gcscontrol->comboBoxFlightMode->setCurrentIndex(m_gcscontrol->comboBoxFlightMode->findText(flightStatus->getField("FlightMode")->getValue().toString()));
    m_gcscontrol->checkBoxArmed->setChecked(flightStatus->getField("Armed")->getValue() == "Armed");
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
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* obj = dynamic_cast<UAVDataObject*>( objManager->getObject(QString("FlightStatus")) );
    UAVObjectField * field = obj->getField("FlightMode");
    field->setValue(field->getOptions()[state]);
    obj->updated();
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

