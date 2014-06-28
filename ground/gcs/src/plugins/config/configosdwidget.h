/**
 ******************************************************************************
 * @file       configosdwidget.h
 * @brief      Configure the OSD
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
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
#ifndef CONFIGOSDWIDGET_H
#define CONFIGOSDWIDGET_H

#include "ui_osd.h"

#include "uavobjectwidgetutils/configtaskwidget.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"

#include "onscreendisplaysettings.h"
#include "manualcontrolcommand.h"
#include "manualcontrolsettings.h"

namespace Ui {
    class Osd;
}

class ConfigOsdWidget: public ConfigTaskWidget
{
    Q_OBJECT

public:
        ConfigOsdWidget(QWidget *parent = 0);
        ~ConfigOsdWidget();

private slots:
    void movePageSlider();
    void updatePositionSlider();

private:
    quint8 scaleSwitchChannel(quint8 channelNumber, quint8 switchPositions);
    QVariant getVariantFromWidget(QWidget * widget, double scale);
    bool setWidgetFromVariant(QWidget *widget, QVariant value, double scale);

    static QString trueString;
    static QString falseString;

    Ui::Osd *ui;

    OnScreenDisplaySettings * osdSettingsObj;
    ManualControlSettings * manualSettingsObj;
    ManualControlSettings::DataFields manualSettingsData;
    ManualControlCommand * manualCommandObj;
    ManualControlCommand::DataFields manualCommandData;
protected:
    void resizeEvent(QResizeEvent *event);
    virtual void enableControls(bool enable);
};

#endif // CONFIGOSDWIDGET_H
 
