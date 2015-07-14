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

#include "onscreendisplaypagesettings.h"
#include "onscreendisplaypagesettings2.h"
#include "onscreendisplaypagesettings3.h"
#include "onscreendisplaypagesettings4.h"


namespace Ui {
    class Osd;
    class OsdPage;
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
    void handle_button_0_1();
    void handle_button_0_2();
    void handle_button_0_3();

    void handle_button_1_0();
    void handle_button_1_2();
    void handle_button_1_3();

    void handle_button_2_0();
    void handle_button_2_1();
    void handle_button_2_3();

    void handle_button_3_0();
    void handle_button_3_1();
    void handle_button_3_2();

    void setCustomText();
    void getCustomText();

private:
    void setupOsdPage(Ui::OsdPage * page, QWidget * page_widget, UAVObject * settings);
    void copyOsdPage(int to, int from);
    quint8 scaleSwitchChannel(quint8 channelNumber, quint8 switchPositions);
    QVariant getVariantFromWidget(QWidget * widget, double scale);
    bool setWidgetFromVariant(QWidget *widget, QVariant value, double scale);


    static QString trueString;
    static QString falseString;

    Ui::Osd *ui;
    Ui::OsdPage * ui_pages[4];
    QWidget *pages[4];

    OnScreenDisplaySettings * osdSettingsObj;
    OnScreenDisplayPageSettings * osdPageSettingsObj;
    OnScreenDisplayPageSettings2 * osdPageSettings2Obj;
    OnScreenDisplayPageSettings3 * osdPageSettings3Obj;
    OnScreenDisplayPageSettings4 * osdPageSettings4Obj;


    ManualControlSettings * manualSettingsObj;
    ManualControlSettings::DataFields manualSettingsData;
    ManualControlCommand * manualCommandObj;
    ManualControlCommand::DataFields manualCommandData;
protected:
    void resizeEvent(QResizeEvent *event);
    virtual void enableControls(bool enable);
};

#endif // CONFIGOSDWIDGET_H
 
