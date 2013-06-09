/**
 ******************************************************************************
 *
 * @file       modulesettingsform.h
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
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

#ifndef MODULESETTINGSFORM_H
#define MODULESETTINGSFORM_H

#include <QWidget>
//#include "configinputwidget.h"
#include "uavobjectwidgetutils/configtaskwidget.h"

namespace Ui {
    class Module_Settings_Widget;
}

class moduleSettingsForm : public ConfigTaskWidget
{
    Q_OBJECT

public:
    explicit moduleSettingsForm(QWidget *parent = 0);
    ~moduleSettingsForm();
    friend class ConfigInputWidget;
    void setName(QString &name);
private slots:
    void minMaxUpdated();
    void neutralUpdated(int);
    void groupUpdated();
    void channelDropdownUpdated(int);
    void channelNumberUpdated(int);

private:
    QVariant getVariantFromWidget(QWidget * widget, double scale);
    bool setWidgetFromVariant(QWidget *widget, QVariant value, double scale);

    static QString trueString;
    static QString falseString;

    Ui::Module_Settings_Widget *moduleSettingsWidget;
};

#endif // MODULESETTINGSFORM_H
