/**
 ******************************************************************************
 * @file       DefaultHwSettingsWidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
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
        defaultHWSettingsWidget(new Ui_defaulthwsettings),
        hwSettingsObject(NULL),
        settingSelected(false)
{
    defaultHWSettingsWidget->setupUi(this);

    //TODO: This is a bit ugly. It sets up a form with no elements. The
    //result is that there is no formatting-- such as scrolling and stretching behavior--, so
    //this has to be manually added in the code.
    //Ideally, there would be a generic hardware page which is filled in either with a board-specific subform, or with
    //generic elements based on the hardware UAVO.
    fieldWidgets.clear();

    bool unknown_board = true;
    if (autopilotConnected){
        addApplySaveButtons(defaultHWSettingsWidget->applyButton,defaultHWSettingsWidget->saveButton);
        addReloadButton(defaultHWSettingsWidget->reloadButton, 0);

        // Query the board plugin for the connected board to get the specific
        // hw settings object
        ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
        if (pm != NULL) {
             UAVObjectUtilManager* uavoUtilManager = pm->getObject<UAVObjectUtilManager>();
             Core::IBoardType* board = uavoUtilManager->getBoardType();
             if (board != NULL) {
                 QString hwSwettingsObject = board->getHwUAVO();

                 UAVObject *obj = getObjectManager()->getObject(hwSwettingsObject);
                 if (obj != NULL) {
                     unknown_board = false;
                     qDebug() << "Checking object " << obj->getName();
                     connect(obj,SIGNAL(transactionCompleted(UAVObject*,bool)), this, SLOT(settingsUpdated(UAVObject*,bool)));
                     obj->requestUpdate();
                 }
             }
        }
    }

    if (unknown_board) {
        QLabel *label = new QLabel("  No recognized board detected.\n  Hardware tab will refresh once a known board is detected.", this);
        label->resize(385, 200);
    }

    disableMouseWheelEvents();
}

DefaultHwSettingsWidget::~DefaultHwSettingsWidget()
{
    delete defaultHWSettingsWidget;
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

        // Have to force the form as clean (unedited by user) since refreshWidgetsValues forces it to dirty.
        setDirty(false);
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

    QLayout *layout = defaultHWSettingsWidget->portSettingsFrame->layout();
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

    QBoxLayout *boxLayout = dynamic_cast<QBoxLayout *>(layout);
    if (boxLayout) {
        boxLayout->addStretch();
    }

    // Prevent mouse wheel from changing items
    disableMouseWheelEvents();
}
