/**
 ******************************************************************************
 *
 * @file       configoutputwidget.h
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ConfigPlugin Config Plugin
 * @{
 * @brief Servo output configuration panel for the config gadget
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
#ifndef CONFIGOUTPUTWIDGET_H
#define CONFIGOUTPUTWIDGET_H

#include "ui_output.h"
#include "../uavobjectwidgetutils/configtaskwidget.h"
#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavobject.h"
#include "uavobjectutilmanager.h"
#include "cfg_vehicletypes/vehicleconfig.h"
#include <QWidget>
#include <QList>

class Ui_OutputWidget;
class OutputChannelForm;

class SignalSingleton : public QObject
{
    Q_OBJECT
public:
    static SignalSingleton* getInstance(QObject* parent=0) {
        if( !p_instance ) {
            p_instance = new SignalSingleton( parent );
        }
        return p_instance;
    }

    static void destroy(){
        if( p_instance ) {
            delete p_instance;
        }
    }

signals:
    void outputChannelsUpdated();

private:
    static SignalSingleton* p_instance;
    explicit SignalSingleton(QObject *parent = 0) :  QObject(parent) {
    }
};


class ConfigOutputWidget: public ConfigTaskWidget
{
	Q_OBJECT

public:
    ConfigOutputWidget(QWidget *parent = 0);
    ~ConfigOutputWidget();

private:
    Ui_OutputWidget *m_config;

    QList<QSlider> sliders;

    void updateChannelInSlider(QSlider *slider, QLabel *min, QLabel *max, QCheckBox *rev, int value);

    void assignChannel(UAVDataObject *obj, QString str);
    OutputChannelForm* getOutputChannelForm(const int index) const;
    int mccDataRate;

    //! List of dropdowns for the timer rate
    QList<QComboBox*> rateList;
    //! List of dropdowns for the timer resolution
    QList<QComboBox*> resList;
    //! List of timer grouping labels
    QList<QLabel*> lblList;

    // For naming custom rates and OneShot
    QString timerFreqToString(quint32) const;
    quint32 timerStringToFreq(QString) const;

    UAVObject::Metadata accInitialData;

    virtual void tabSwitchingAway();

private slots:
    void stopTests();
    virtual void refreshWidgetsValues(UAVObject * obj=NULL);
    void updateObjectsFromWidgets();
    void runChannelTests(bool state);
    void sendChannelTest(int index, int value);
    void startESCCalibration();
    void openHelp();
    void do_SetDirty();
    void assignOutputChannels();
    void refreshWidgetRanges();

protected:
    void enableControls(bool enable);
};

#endif
