/**
 ******************************************************************************
 * @file       DefaultHwSettingsWidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Placeholder for attitude panel until board is connected.
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
#include "defaulthwsettingswidget.h"
#include "ui_defaultattitude.h"
#include "hwfieldselector.h"
#include <QMutexLocker>
#include <QErrorMessage>
#include <QDebug>

/**
 * @brief DefaultHwSettingsWidget::DefaultHwSettingsWidget Constructed when either a new
 * board connection is established or when there is no board
 * @param parent The main configuration widget
 */
DefaultHwSettingsWidget::DefaultHwSettingsWidget(QWidget *parent, bool autopilotConnected) :
        ConfigTaskWidget(parent),
        ui(new Ui_defaulthwsettings),
        hwSettingsObject(NULL),
        settingSelected(false)
{
    ui->setupUi(this);

    //TODO: This is a bit ugly. It sets up a form with no elements. The
    //result is that there is no formatting-- such as scrolling and stretching behavior--, so
    //this has to be manually added in the code.
    //Ideally, there would be a generic hardware page which is filled in either with a board-specific subform, or with
    //generic elements based on the hardware UAVO.
    fieldWidgets.clear();

    if (autopilotConnected){
        addApplySaveButtons(ui->applyButton,ui->saveButton);
        addReloadButton(ui->reloadButton, 0);

        //List of supported boards which do not currently have board-specific pages
        allHwSettings.append("HwFlyingF3");
        allHwSettings.append("HwFlyingF4");
        allHwSettings.append("HwFreedom");
        allHwSettings.append("HwRevolution");
        allHwSettings.append("HwRevoMini");
        allHwSettings.append("HwQuanton");

        //Connect all forms to slots
        foreach (QString str, allHwSettings) {
            UAVObject *obj = getObjectManager()->getObject(str);
            if (obj != NULL) {
                qDebug() << "Checking object " << obj->getName();
                connect(obj,SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(settingsUpdated(UAVObject*,bool)));
                obj->requestUpdate();
            }
        }
    }
    else{
        qDebug() << "No hardware attached. Showing placeholder text/graphic.";
        QLabel *label = new QLabel("  No board detected.\n  Hardware tab will refresh once board is detected.", this);
        label->resize(335,200);
    }
}

DefaultHwSettingsWidget::~DefaultHwSettingsWidget()
{
    delete ui;
}

void DefaultHwSettingsWidget::settingsUpdated(UAVObject *obj, bool success)
{
    if (success && !settingSelected) {
        qDebug() << "Selected object " << obj->getName();
        settingSelected = true;

        hwSettingsObject = obj;
        updateFields();

        QList<int> reloadGroups;
        reloadGroups << 0;

        addUAVObject(obj, &reloadGroups);
        refreshWidgetsValues();
    }
}

/**
 * @brief DefaultHwSettingsWidget::updateFields Update the list of fields and show all of them
 * on the UI.  Connect each to the smart save system.
 */
void DefaultHwSettingsWidget::updateFields()
{
    Q_ASSERT(settingSelected);
    Q_ASSERT(hwSettingsObject != NULL);

    QLayout *layout = ui->portSettingsFrame->layout();
    for (int i = 0; i < fieldWidgets.size(); i++)
        layout->removeWidget(fieldWidgets[i]);
    fieldWidgets.clear();

    QList <UAVObjectField*> fields = hwSettingsObject->getFields();
    for (int i = 0; i < fields.size(); i++) {
        if (fields[i]->getType() != UAVObjectField::ENUM)
            continue;
        HwFieldSelector *sel = new HwFieldSelector(this);
        layout->addWidget(sel);
        sel->setUavoField(fields[i]);
        fieldWidgets.append(sel);
        addUAVObjectToWidgetRelation(hwSettingsObject->getName(),fields[i]->getName(),sel->getCombo());
    }
}
