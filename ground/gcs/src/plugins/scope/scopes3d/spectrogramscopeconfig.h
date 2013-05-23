/**
 ******************************************************************************
 *
 * @file       spectrogramplotconfig.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup ScopePlugin Scope Gadget Plugin
 * @{
 * @brief The scope Gadget, graphically plots the states of UAVObjects
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

#ifndef SPECTROGRAMSCOPECONFIG_H
#define SPECTROGRAMSCOPECONFIG_H

#include "scopes3d/scopes3dconfig.h"


/**
 * @brief The SpectrogramScopeConfig class The spectrogram scope configuration
 */
class SpectrogramScopeConfig : public Scopes3dConfig
{
    Q_OBJECT
public:
    /**
     * @brief The SpectrogramType enum Defines the different type of spectrogram plots.
     */
    enum SpectrogramType {
        VIBRATIONANALYSIS,
        CUSTOM_SPECTROGRAM
    };

    SpectrogramScopeConfig();
    SpectrogramScopeConfig(QSettings *qSettings);
    SpectrogramScopeConfig(Ui::ScopeGadgetOptionsPage *options_page);
    ~SpectrogramScopeConfig();

    virtual void saveConfiguration(QSettings* qSettings);
    void create(QSettings qSettings);

    QList<Plot3dCurveConfiguration*> getSpectrogramDataSource(){return m_spectrogramSourceConfigs;}
    void addSpectrogramDataSource(Plot3dCurveConfiguration* value){m_spectrogramSourceConfigs.append(value);}
    void replaceSpectrogramDataSource(QList<Plot3dCurveConfiguration*> spectrogramSourceConfigs);

    //Getter functions
    double getSamplingFrequency(){return samplingFrequency;}
    double getZMaximum(){return zMaximum;}
    unsigned int getWindowWidth(){return windowWidth;}
    double getTimeHorizon(){return timeHorizon;}
    virtual QList<Plot3dCurveConfiguration*> getDataSourceConfigs(){return m_spectrogramSourceConfigs;}
    virtual int getScopeType(){return SPECTROGRAM;}

    //Setter functions
    void setSamplingFrequency(double val){samplingFrequency = val;}
    void setZMaximum(double val){zMaximum = val;}
    void setWindowWidth(unsigned int val){windowWidth = val;}
    void setTimeHorizon(double val){timeHorizon = val;}
    virtual void setGuiConfiguration(Ui::ScopeGadgetOptionsPage *options_page);
    virtual ScopeConfig* cloneScope(ScopeConfig*);

    virtual void loadConfiguration(ScopeGadgetWidget *scopeGadgetWidget);
    virtual void preparePlot(ScopeGadgetWidget *);
    void configureAxes(ScopeGadgetWidget *);

    ColorMap::ColorMapType getColorMap(){return colorMapType;}

private slots:

private:
    SpectrogramType spectrogramType;

    double timeHorizon;
    QString units;

    QList<Plot3dCurveConfiguration*> m_spectrogramSourceConfigs;

    double samplingFrequency;
    unsigned int windowWidth;
    QString yAxisUnits;
    double zMaximum;

    ColorMap::ColorMapType colorMapType;
};

#endif // SPECTROGRAMSCOPECONFIG_H
