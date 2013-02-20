/**
 ******************************************************************************
 *
 * @file       scopegadgetoptionspage.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://www.taulabs.org Copyright (C) 2013.
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

#include "scopegadgetoptionspage.h"

#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavdataobject.h"

#include "scopes2d/histogramscopeconfig.h"
#include "scopes2d/scatterplotscopeconfig.h"
#include "scopes3d/spectrogramscopeconfig.h"

#include <QtGui/qpalette.h>
#include <QtGui/QMessageBox>


ScopeGadgetOptionsPage::ScopeGadgetOptionsPage(ScopeGadgetConfiguration *config, QObject *parent) :
        IOptionsPage(parent),
        m_config(config)
{
    //nothing to do here...
}

/**
 * @brief ScopeGadgetOptionsPage::createPage creates options page widget (uses the UI file)
 * @param parent Parent QWidghet
 * @return Returns options page widget
 */
QWidget* ScopeGadgetOptionsPage::createPage(QWidget *parent)
{
    Q_UNUSED(parent);

    options_page = new Ui::ScopeGadgetOptionsPage();
    //main widget
    QWidget *optionsPageWidget = new QWidget;
    //main layout
    options_page->setupUi(optionsPageWidget);

    //Set up 2D plots tab
    options_page->cmb2dPlotType->addItem("Scatter plot", SCATTERPLOT2D);
    options_page->cmb2dPlotType->addItem("Histogram", HISTOGRAM);
//    options_page->cmb2dPlotType->addItem("Polar plot", POLARPLOT);

    //Set up x-axis combo box
    options_page->cmbXAxisScatterplot2d->addItem("Series", SERIES2D);
    options_page->cmbXAxisScatterplot2d->addItem("Time series", TIMESERIES2D);


    //Set up 3D plots tab
//    options_page->cmb3dPlotType->addItem("Time series", TimeSeries3d);
    options_page->cmb3dPlotType->addItem("Spectrogram", SPECTROGRAM);

    options_page->cmbSpectrogramSource->addItem("Custom", CUSTOM);
    options_page->cmbSpectrogramSource->addItem("Vibration Test", VIBRATIONTEST);

    // Fills the combo boxes for the UAVObjects
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    QList< QList<UAVDataObject*> > objList = objManager->getDataObjects();
    foreach (QList<UAVDataObject*> list, objList) {
        foreach (UAVDataObject* obj, list) {
            if (obj->isSingleInstance())
            {
                options_page->cmbUAVObjects->addItem(obj->getName());
            }
            else if(obj->getName() != options_page->cmbUAVObjects->itemText(options_page->cmbUAVObjects->count()-1))
            { //Checks to see if we're duplicating UAVOs because of multiple instances
                options_page->cmbUAVObjects->addItem(obj->getName());
                options_page->cmbUAVObjectsSpectrogram->addItem(obj->getName());
            }
        }
    }

    QStringList mathFunctions;
    mathFunctions << "None" << "Boxcar average" << "Standard deviation";

    options_page->mathFunctionComboBox->addItems(mathFunctions);
    options_page->cmbMathFunctionSpectrogram->addItems(mathFunctions);

    // Check that an index is currently selected, and update if true
    if(options_page->cmbUAVObjects->currentIndex() >= 0){
        on_cmbUAVObjects_currentIndexChanged(options_page->cmbUAVObjects->currentText());
    }

    // Add scaling items for
    options_page->cmbScale->addItem("10^-9", -9);
    options_page->cmbScale->addItem("10^-6", -6);
    options_page->cmbScale->addItem("10^-5",-5);
    options_page->cmbScale->addItem("10^-4",-4);
    options_page->cmbScale->addItem("10^-3",-3);
    options_page->cmbScale->addItem("10^-2",-2);
    options_page->cmbScale->addItem("10^-1",-1);
    options_page->cmbScale->addItem("1",0);
    options_page->cmbScale->addItem("10^1",1);
    options_page->cmbScale->addItem("10^2",2);
    options_page->cmbScale->addItem("10^3",3);
    options_page->cmbScale->addItem("10^4",4);
    options_page->cmbScale->addItem("10^5",5);
    options_page->cmbScale->addItem("10^6",6);
    options_page->cmbScale->addItem("10^9",9);
    options_page->cmbScale->addItem("10^12",12);
    options_page->cmbScale->setCurrentIndex(7);

//    QStringList scaleTypes;
//    scaleTypes << "10^-9" << "10^-6" << "10^-5" << "10^-4" << "10^-3" << "10^-2" << "10^-1"
//               << "1" << "10^1" << "10^2" << "10^3" << "10^4" << "10^5" << "10^6" << "10^9" << "10^12";
//    options_page->cmbScale->addItems(scaleTypes);
//    options_page->cmbScale->setCurrentIndex(7);

    // Configure color button
    options_page->btnColor->setAutoFillBackground(true);

    //Connect signals to slots

    connect(options_page->cmbXAxisScatterplot2d, SIGNAL(currentIndexChanged(QString)), this, SLOT(on_cmbXAxisScatterplot2d_currentIndexChanged(QString)));
    connect(options_page->cmb2dPlotType, SIGNAL(currentIndexChanged(QString)), this, SLOT(on_cmb2dPlotType_currentIndexChanged(QString)));
    connect(options_page->cmb3dPlotType, SIGNAL(currentIndexChanged(QString)), this, SLOT(on_cmb3dPlotType_currentIndexChanged(QString)));
    connect(options_page->cmbUAVObjects, SIGNAL(currentIndexChanged(QString)), this, SLOT(on_cmbUAVObjects_currentIndexChanged(QString)));
    connect(options_page->cmbUAVObjectsSpectrogram, SIGNAL(currentIndexChanged(QString)), this, SLOT(on_cmbUAVObjectsSpectrogram_currentIndexChanged(QString)));
    connect(options_page->btnAdd2dCurve, SIGNAL(clicked()), this, SLOT(on_btnAdd2dCurve_clicked()));
    connect(options_page->btnApply2dCurve, SIGNAL(clicked()), this, SLOT(on_btnApply2dCurve_clicked()));
    connect(options_page->btnRemove2dCurve, SIGNAL(clicked()), this, SLOT(on_btnRemove2dCurve_clicked()));
    connect(options_page->lst2dCurves, SIGNAL(currentRowChanged(int)), this, SLOT(on_lst2dCurves_currentRowChanged(int)));
    connect(options_page->btnColor, SIGNAL(clicked()), this, SLOT(on_btnColor_clicked()));
    connect(options_page->mathFunctionComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(on_mathFunctionComboBox_currentIndexChanged(int)));
    connect(options_page->cmbSpectrogramSource, SIGNAL(currentIndexChanged(QString)), this, SLOT(on_cmbSpectrogramSource_currentIndexChanged(QString)));
    connect(options_page->tabWidget2d3d, SIGNAL(currentChanged(int)), this, SLOT(on_tabWidget2d3d_currentIndexChanged(int)));


    //Set widget elements to reflect plot configurations
    if(m_config->getScope()->getScopeDimensions() == PLOT2D)
    {
        //Set the tab widget to 2D
        options_page->tabWidget2d3d->setCurrentWidget(options_page->tabPlot2d);
        on_tabWidget2d3d_currentIndexChanged(options_page->tabWidget2d3d->currentIndex());

        //Set the plot type
        options_page->cmb2dPlotType->setCurrentIndex(options_page->cmb2dPlotType->findData(m_config->getScopeType()));
        on_cmb2dPlotType_currentIndexChanged(options_page->cmb2dPlotType->currentText());

        //Set widget values from settings

        //add the configured 2D curves
        options_page->lst2dCurves->clear();  //Clear list first
        if (m_config->getScopeType() == SCATTERPLOT2D){
            Scatterplot2dScope* configBob = (Scatterplot2dScope*) m_config->getScope();

            foreach (Plot2dCurveConfiguration* plotData, configBob->getScatterplotDataSource()) {
                options_page->cmbXAxisScatterplot2d->setCurrentIndex(configBob->getScatterplot2dType());
                options_page->spnDataSize->setValue(configBob->getTimeHorizon());

                QString uavObjectName = plotData->uavObjectName;
                QString uavFieldName = plotData->uavFieldName;
                int scale = plotData->yScalePower;
                int mean = plotData->yMeanSamples;
                QString mathFunction = plotData->mathFunction;
                QVariant varColor = plotData->color;

                addPlot2dCurveConfig(uavObjectName,uavFieldName,scale,mean,mathFunction,varColor);
            }
        }
        else if (m_config->getScopeType() == HISTOGRAM){
            HistogramScope* configBob = (HistogramScope*) m_config->getScope();

            foreach (Plot2dCurveConfiguration* dataSource,  configBob->getHistogramDataSource()) {
                options_page->spnMaxNumBins->setValue(configBob->getMaxNumberOfBins());
                options_page->spnBinWidth->setValue(configBob->getBinWidth());

                QString uavObjectName = dataSource->uavObjectName;
                QString uavFieldName = dataSource->uavFieldName;
                int scale = dataSource->yScalePower;
                int mean = dataSource->yMeanSamples;
                QString mathFunction = dataSource->mathFunction;
                QVariant varColor = dataSource->color;

                addPlot2dCurveConfig(uavObjectName,uavFieldName,scale,mean,mathFunction,varColor);
            }
        }
        else{
            Q_ASSERT(0);
        }

        //Select row 1st row in list
        options_page->lst2dCurves->setCurrentRow(0, QItemSelectionModel::ClearAndSelect);
    }
    else if(m_config->getScope()->getScopeDimensions() == PLOT3D)
    {
        //Set the tab widget to 3D
        options_page->tabWidget2d3d->setCurrentWidget(options_page->tabPlot3d);

        //Set the plot type
        options_page->cmb3dPlotType->setCurrentIndex(options_page->cmb3dPlotType->findData(m_config->getScope()->getScopeType()));

        if(m_config->getScopeType() == SPECTROGRAM){
            SpectrogramScope* configBob = (SpectrogramScope*) m_config->getScope();
            options_page->sbSpectrogramTimeHorizon->setValue(configBob->getTimeHorizon());
            options_page->sbSpectrogramFrequency->setValue(configBob->getSamplingFrequency());
            options_page->spnMaxSpectrogramZ->setValue(configBob->getZMaximum());

            foreach (Plot3dCurveConfiguration* plot3dData,  configBob->getSpectrogramDataSource()) {
                int uavoIdx= options_page->cmbUAVObjectsSpectrogram->findText(plot3dData->uavObjectName);
                options_page->cmbUAVObjectsSpectrogram->setCurrentIndex(uavoIdx);
                on_cmbUAVObjectsSpectrogram_currentIndexChanged(plot3dData->uavObjectName);
                options_page->sbSpectrogramWidth->setValue(configBob->getWindowWidth());

                int uavoFieldIdx= options_page->cmbUavoFieldSpectrogram->findText(plot3dData->uavFieldName);
                options_page->cmbUavoFieldSpectrogram->setCurrentIndex(uavoFieldIdx);
            }
        }


//        //Set widget values from settings

//        //add the configured 3D curves
//        foreach (Plot3dCurveConfiguration* plotData,  m_config->plot3dCurveConfigs()) {

//            QString uavObjectName = plotData->uavObjectName;
//            QString uavFieldName = plotData->uavFieldName;
//            int scale = plotData->yScalePower;
//            int mean = plotData->yMeanSamples;
//            QString mathFunction = plotData->mathFunction;
//            QVariant varColor = plotData->color;

//            addPlot3dCurveConfig(uavObjectName,uavFieldName,scale,mean,mathFunction,varColor);
//        }
    }

    //Disable mouse wheel events //TODO: DOES NOT WORK
    foreach( QSpinBox * sp, findChildren<QSpinBox*>() ) {
        sp->installEventFilter( this );
    }
    foreach( QDoubleSpinBox * sp, findChildren<QDoubleSpinBox*>() ) {
        sp->installEventFilter( this );
    }
    foreach( QSlider * sp, findChildren<QSlider*>() ) {
        sp->installEventFilter( this );
    }
    foreach( QComboBox * sp, findChildren<QComboBox*>() ) {
        sp->installEventFilter( this );
    }


    return optionsPageWidget;
}


/**
 * @brief ScopeGadgetOptionsPage::eventFilter Filters all wheel events.
 * @param obj
 * @param evt
 * @return
 */
bool ScopeGadgetOptionsPage::eventFilter( QObject * obj, QEvent * evt ) {
    //Filter all wheel events, and ignore them
    if ( evt->type() == QEvent::Wheel &&
         (qobject_cast<QAbstractSpinBox*>( obj ) ||
          qobject_cast<QComboBox*>( obj ) ||
          qobject_cast<QAbstractSlider*>( obj ) ))
    {
        evt->ignore();
        return true;
    }
    return ScopeGadgetOptionsPage::eventFilter( obj, evt );
}

void ScopeGadgetOptionsPage::on_mathFunctionComboBox_currentIndexChanged(int currentIndex){
    if (currentIndex > 0){
        options_page->spnMeanSamples->setEnabled(true);
    }
    else{
        options_page->spnMeanSamples->setEnabled(false);
    }

}

void ScopeGadgetOptionsPage::on_cmbSpectrogramSource_currentIndexChanged(QString currentText)
{
    if (currentText == "Vibration Test" ){
        int vibrationTestIdx = options_page->cmbUAVObjectsSpectrogram->findText("VibrationTestOutput");
        options_page->cmbUAVObjectsSpectrogram->setCurrentIndex(vibrationTestIdx);
        options_page->cmbUAVObjectsSpectrogram->setEnabled(false);
    }
    else{
        options_page->cmbUAVObjectsSpectrogram->setEnabled(true);
    }

}


/**
 * @brief ScopeGadgetOptionsPage::on_btnColor_clicked When clicked, open a color picker. If
 * a color is chosen, apply it to the QPushButton element
 */
void ScopeGadgetOptionsPage::on_btnColor_clicked()
 {
     QColor color = QColorDialog::getColor( QColor(options_page->btnColor->text()));
     if (color.isValid()) {
         setButtonColor(color);
     }
 }


/**
 * @brief ScopeGadgetOptionsPage::set2dYAxisWidgetFromDataSource Populate the widgets that
 * contain the configuration for the Y-Axis from the selected plot curve
 */
void ScopeGadgetOptionsPage::set2dYAxisWidgetFromDataSource()
{
    bool parseOK = false;
    QListWidgetItem* listItem = options_page->lst2dCurves->currentItem();

    if(listItem == 0)
        return;

    //TODO: WHAT IS UserRole DOING?
    int currentIndex = options_page->cmbUAVObjects->findText( listItem->data(Qt::UserRole + 0).toString());
    options_page->cmbUAVObjects->setCurrentIndex(currentIndex);

    currentIndex = options_page->cmbUAVField->findText( listItem->data(Qt::UserRole + 1).toString());
    options_page->cmbUAVField->setCurrentIndex(currentIndex);

    currentIndex = options_page->cmbScale->findData( listItem->data(Qt::UserRole + 2), Qt::UserRole, Qt::MatchExactly);
    options_page->cmbScale->setCurrentIndex(currentIndex);

    QVariant varColor  = listItem->data(Qt::UserRole + 3);
    int rgb = varColor.toInt(&parseOK);
    if (!parseOK)
        rgb = QColor(Qt::red).rgb();

    setButtonColor(QColor((QRgb) rgb));

    int mean = listItem->data(Qt::UserRole + 4).toInt(&parseOK);
    if(!parseOK)
        mean = 1;
    options_page->spnMeanSamples->setValue(mean);

    currentIndex = options_page->mathFunctionComboBox->findText( listItem->data(Qt::UserRole + 5).toString());
    options_page->mathFunctionComboBox->setCurrentIndex(currentIndex);
}


/**
 * @brief ScopeGadgetOptionsPage::setButtonColor Sets the Color picker button background
 * to the chosen color
 * @param color RGB color
 */
void ScopeGadgetOptionsPage::setButtonColor(const QColor &color)
{
    //TODO: Understand why this doesn't work when starting a new page. It only works when physically clicking on the button
    options_page->btnColor->setText(color.name());
    options_page->btnColor->setPalette(QPalette(color));

}


/**
 * @brief ScopeGadgetOptionsPage::on_cmbUAVObjects_currentIndexChanged When a new
 * UAVObject is selected, populate the UAVObject field combo box with the correct values.
 * @param val
 */
void ScopeGadgetOptionsPage::on_cmbUAVObjects_currentIndexChanged(QString val)
{
    options_page->cmbUAVField->clear();

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* objData = dynamic_cast<UAVDataObject*>( objManager->getObject(val) );

    if (objData == NULL)
        return; // Rare case: the config contained a UAVObject name which does not exist anymore.

    QList<UAVObjectField*> fieldList = objData->getFields();
    foreach (UAVObjectField* field, fieldList) {
        if(field->getType() == UAVObjectField::STRING || field->getType() == UAVObjectField::ENUM )
            continue;

        if(field->getElementNames().count() > 1)
        {
            foreach(QString elemName , field->getElementNames())
            {
                options_page->cmbUAVField->addItem(field->getName() + "-" + elemName);
            }
        }
        else
            options_page->cmbUAVField->addItem(field->getName());
    }
}


/**
 * @brief ScopeGadgetOptionsPage::on_cmbUAVObjectsSpectrogram_currentIndexChanged When a new
 * UAVObject is selected, populate the UAVObject field combo box with the correct values. Only
 * populate with UAVOs that have multiple instances.
 * @param val
 */
void ScopeGadgetOptionsPage::on_cmbUAVObjectsSpectrogram_currentIndexChanged(QString val)
{
    options_page->cmbUavoFieldSpectrogram->clear();

    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    UAVDataObject* objData = dynamic_cast<UAVDataObject*>( objManager->getObject(val) );

    if (objData == NULL)
        return; // Rare case: the config contained a UAVObject name which does not exist anymore.

    QList<UAVObjectField*> fieldList = objData->getFields();
    foreach (UAVObjectField* field, fieldList) {

        if(field->getType() == UAVObjectField::STRING || field->getType() == UAVObjectField::ENUM)
            continue;

        if(field->getElementNames().count() > 1)
        {
            foreach(QString elemName , field->getElementNames())
            {
                options_page->cmbUavoFieldSpectrogram->addItem(field->getName() + "-" + elemName);
            }
        }
        else
            options_page->cmbUavoFieldSpectrogram->addItem(field->getName());
    }

    // Get range from UAVO name
    unsigned int maxWidth = objManager->getNumInstances(objData->getObjID());
    options_page->sbSpectrogramWidth->setRange(0, maxWidth);
    options_page->sbSpectrogramWidth->setValue(maxWidth);

}


/**
 * @brief ScopeGadgetOptionsPage::apply Called when the user presses OK. Saves the current values
 */
void ScopeGadgetOptionsPage::apply()
{
    bool parseOK = false;

    //Apply configuration changes

    if(options_page->tabWidget2d3d->currentWidget() == options_page->tabPlot2d)
    {   //--- 2D ---//

        Plot2dType current2dType = (Plot2dType) options_page->cmb2dPlotType->itemData(options_page->cmb2dPlotType->currentIndex()).toUInt();
        m_config->getScope()->setScopeType(current2dType);
        m_config->getScope()->setScopeDimensions(PLOT2D);


        if (current2dType == HISTOGRAM){
            QList<Plot2dCurveConfiguration*> histogramConfigs;
            HistogramScope* configBob = (HistogramScope*) m_config;

            configBob->setBinWidth(options_page->spnBinWidth->value());
            configBob->setMaxNumberOfBins(options_page->spnMaxNumBins->value());

            //For each y-data source in the list
            for(int iIndex = 0; iIndex < options_page->lst2dCurves->count();iIndex++) {
                QListWidgetItem* listItem = options_page->lst2dCurves->item(iIndex);

                Plot2dCurveConfiguration* newPlotCurveConfigs = new Plot2dCurveConfiguration();
                newPlotCurveConfigs->uavObjectName = listItem->data(Qt::UserRole + 0).toString();
                newPlotCurveConfigs->uavFieldName  = listItem->data(Qt::UserRole + 1).toString();
                newPlotCurveConfigs->yScalePower  = listItem->data(Qt::UserRole + 2).toInt(&parseOK);
                if(!parseOK)
                    newPlotCurveConfigs->yScalePower = 0;

                QVariant varColor  = listItem->data(Qt::UserRole + 3);
                int rgb = varColor.toInt(&parseOK);
                if(!parseOK)
                    newPlotCurveConfigs->color = QColor(Qt::black).rgb();
                else
                    newPlotCurveConfigs->color = (QRgb)rgb;

                newPlotCurveConfigs->yMeanSamples = listItem->data(Qt::UserRole + 4).toInt(&parseOK);
                if(!parseOK)
                    newPlotCurveConfigs->yMeanSamples = 1;

                newPlotCurveConfigs->mathFunction  = listItem->data(Qt::UserRole + 5).toString();


                histogramConfigs.append(newPlotCurveConfigs);
            }

            configBob->replaceHistogramDataSource(histogramConfigs);
        }
        else if (current2dType == SCATTERPLOT2D){
            QList<Plot2dCurveConfiguration*> plot2dCurveConfigs;
            Scatterplot2dScope* configBob = (Scatterplot2dScope*) m_config;

            configBob->setTimeHorizon(options_page->spnDataSize->value());

            Scatterplot2dType currentScatterplotType = (Scatterplot2dType) options_page->cmbXAxisScatterplot2d->itemData(options_page->cmbXAxisScatterplot2d->currentIndex()).toUInt();
            configBob->setScatterplot2dType(currentScatterplotType);

            for(int iIndex = 0; iIndex < options_page->lst2dCurves->count();iIndex++) {
                QListWidgetItem* listItem = options_page->lst2dCurves->item(iIndex);

                Plot2dCurveConfiguration* newPlotCurveConfigs = new Plot2dCurveConfiguration();
                newPlotCurveConfigs->uavObjectName = listItem->data(Qt::UserRole + 0).toString();
                newPlotCurveConfigs->uavFieldName  = listItem->data(Qt::UserRole + 1).toString();
                newPlotCurveConfigs->yScalePower  = listItem->data(Qt::UserRole + 2).toInt(&parseOK);
                if(!parseOK)
                    newPlotCurveConfigs->yScalePower = 0;

                QVariant varColor  = listItem->data(Qt::UserRole + 3);
                int rgb = varColor.toInt(&parseOK);
                if(!parseOK)
                    newPlotCurveConfigs->color = QColor(Qt::black).rgb();
                else
                    newPlotCurveConfigs->color = (QRgb)rgb;

                newPlotCurveConfigs->yMeanSamples = listItem->data(Qt::UserRole + 4).toInt(&parseOK);
                if(!parseOK)
                    newPlotCurveConfigs->yMeanSamples = 1;

                newPlotCurveConfigs->mathFunction  = listItem->data(Qt::UserRole + 5).toString();


                plot2dCurveConfigs.append(newPlotCurveConfigs);
            }

            configBob->replaceScatterplotDataSource(plot2dCurveConfigs);
        }
        else{
            Q_ASSERT(0);
        }

    }
    else
    {   //--- 3D ---//
        QList<Plot3dCurveConfiguration*> plot3dCurveConfigs;

        //TODO: Investigage how the type can be decided ahead of time, before checking the Stack
        Plot3dType current3dType = (Plot3dType) options_page->cmb3dPlotType->itemData(options_page->cmb3dPlotType->currentIndex()).toUInt(); //[1]HUH?-->[2]
        m_config->getScope()->setScopeType(current3dType);
        m_config->getScope()->setScopeDimensions(PLOT3D);

        if (options_page->stackedWidget3dPlots->currentWidget() == options_page->sw3dSpectrogramStack)
        {
            SpectrogramScope* configBob = (SpectrogramScope*) m_config;

            configBob->setWindowWidth(options_page->sbSpectrogramWidth->value());
            configBob->setSamplingFrequency(options_page->sbSpectrogramFrequency->value());
            configBob->setTimeHorizon(options_page->sbSpectrogramTimeHorizon->value());
            configBob->setZMaximum(options_page->spnMaxSpectrogramZ->value());


            Plot3dCurveConfiguration* newPlotCurveConfigs = new Plot3dCurveConfiguration();
            newPlotCurveConfigs->uavObjectName = options_page->cmbUAVObjectsSpectrogram->currentText();
            newPlotCurveConfigs->uavFieldName  = options_page->cmbUavoFieldSpectrogram->currentText();
            newPlotCurveConfigs->yScalePower   = options_page->sbSpectrogramDataMultiplier->value();
            newPlotCurveConfigs->yMeanSamples  = options_page->spnMeanSamplesSpectrogram->value();
            newPlotCurveConfigs->mathFunction  = options_page->cmbMathFunctionSpectrogram->currentText();

            QVariant varColor = (int)QColor(options_page->btnColorSpectrogram->text()).rgb();
            int rgb = varColor.toInt(&parseOK);
            if(!parseOK)
                newPlotCurveConfigs->color = QColor(Qt::red).rgb();
            else
                newPlotCurveConfigs->color = (QRgb) rgb;

            plot3dCurveConfigs.append(newPlotCurveConfigs);

            configBob->replaceSpectrogramDataSource(plot3dCurveConfigs);
        }
        else if (options_page->stackedWidget3dPlots->currentWidget() == options_page->sw3dTimeSeriesStack)
        {
//            m_config->setPlot3dType(Scatterplot3d);

            for(int iIndex = 0; iIndex < options_page->lst2dCurves->count();iIndex++) {
                QListWidgetItem* listItem = options_page->lst2dCurves->item(iIndex);

                Plot3dCurveConfiguration* newPlotCurveConfigs = new Plot3dCurveConfiguration();
                newPlotCurveConfigs->uavObjectName = listItem->data(Qt::UserRole + 0).toString();
                newPlotCurveConfigs->uavFieldName  = listItem->data(Qt::UserRole + 1).toString();
                newPlotCurveConfigs->yScalePower   = listItem->data(Qt::UserRole + 2).toInt(&parseOK);
                if(!parseOK)
                    newPlotCurveConfigs->yScalePower = 0;

                QVariant varColor  = listItem->data(Qt::UserRole + 3);
                int rgb = varColor.toInt(&parseOK);
                if(!parseOK)
                    newPlotCurveConfigs->color = QColor(Qt::black).rgb();
                else
                    newPlotCurveConfigs->color = (QRgb)rgb;

                newPlotCurveConfigs->yMeanSamples = listItem->data(Qt::UserRole + 4).toInt(&parseOK);
                if(!parseOK)
                    newPlotCurveConfigs->yMeanSamples = 1;

                newPlotCurveConfigs->mathFunction  = listItem->data(Qt::UserRole + 5).toString();

                plot3dCurveConfigs.append(newPlotCurveConfigs);
            }
        }
        else{
            Q_ASSERT(0);
        }
    }
}


/**
 * @brief ScopeGadgetOptionsPage::on_btnAdd2dCurve_clicked Add a new data source to the 2D plot.
 */
void ScopeGadgetOptionsPage::on_btnAdd2dCurve_clicked()
{
    bool parseOK = false;
    QString uavObject = options_page->cmbUAVObjects->currentText();
    QString uavField = options_page->cmbUAVField->currentText();
    int scale = options_page->cmbScale->itemData(options_page->cmbScale->currentIndex()).toInt(&parseOK);

    if(!parseOK)
       scale = 0;

    int mean = options_page->spnMeanSamples->value();
    QString mathFunction = options_page->mathFunctionComboBox->currentText();

    QVariant varColor = (int)QColor(options_page->btnColor->text()).rgb();

    // Add curve
    addPlot2dCurveConfig(uavObject, uavField, scale, mean, mathFunction, varColor);
}


void ScopeGadgetOptionsPage::on_btnApply2dCurve_clicked()
{
    bool parseOK = false;
    QString uavObjectName = options_page->cmbUAVObjects->currentText();
    QString uavFieldName = options_page->cmbUAVField->currentText();
    int scale = options_page->cmbScale->itemData(options_page->cmbScale->currentIndex()).toInt(&parseOK);

    if(!parseOK)
       scale = 0;

    int mean = options_page->spnMeanSamples->value();
    QString mathFunction = options_page->mathFunctionComboBox->currentText();

    QVariant varColor = (int)QColor(options_page->btnColor->text()).rgb();

    // Apply curve settings
    QListWidgetItem *listWidgetItem = options_page->lst2dCurves->currentItem();
    if(listWidgetItem == NULL){
        //TODO: Replace the second and third [in eraseDone()] pop-up dialogs with a progress indicator,
        // counter, or infinite chain of `......` tied to the original dialog box
        QMessageBox msgBox;
        msgBox.setText(tr("No curve selected."));
        msgBox.setInformativeText(tr("Please generate a curve with the ""+"" symbol."));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();

        return;
    }

    qDebug() << uavObjectName << " " << uavFieldName << " " << scale << " " << mean << " " << mathFunction << " " << varColor << " ";

    setPlot2dCurveProperties(listWidgetItem, uavObjectName, uavFieldName, scale, mean, mathFunction, varColor);
}


/**
 * @brief ScopeGadgetOptionsPage::on_btnRemove2dCurve_clicked Remove a curve config from the plot.
 */
void ScopeGadgetOptionsPage::on_btnRemove2dCurve_clicked()
{
    options_page->lst2dCurves->takeItem(options_page->lst2dCurves->currentIndex().row());
}


//Add a new curve config to the list
void ScopeGadgetOptionsPage::addPlot2dCurveConfig(QString uavObjectName, QString uavFieldName, int scale, int mean, QString mathFunction, QVariant varColor)
{
    QString listItemDisplayText = uavObjectName + "." + uavFieldName; // Generate the name
    options_page->lst2dCurves->addItem(listItemDisplayText);  // Add the name to the list
    int itemIdx = options_page->lst2dCurves->count() - 1; // Get the index number for the new value
    QListWidgetItem *listWidgetItem = options_page->lst2dCurves->item(itemIdx); //Find the widget item

    //Apply all settings to curve
    setPlot2dCurveProperties(listWidgetItem, uavObjectName, uavFieldName, scale, mean, mathFunction, varColor);

    //Select the row with the new name
    options_page->lst2dCurves->setCurrentRow(itemIdx);

}


//Add a new curve config to the list
void ScopeGadgetOptionsPage::addPlot3dCurveConfig(QString uavObjectName, QString uavFieldName, int scale, int mean, QString mathFunction, QVariant varColor)
{
    // Do something here...
}


/**
 * @brief ScopeGadgetOptionsPage::setPlot2dCurveProperties Set the y-axis curve properties. Overwrites
 * the existing scope configuration.
 * @param listWidgetItem
 * @param uavObject UAVO name. If the name is empty, defaul to "New graph"
 * @param uavField
 * @param scale
 * @param mean
 * @param mathFunction
 * @param varColor
 */
void ScopeGadgetOptionsPage::setPlot2dCurveProperties(QListWidgetItem *listWidgetItem,QString uavObject, QString uavField, int scale, int mean, QString mathFunction, QVariant varColor)
{
    bool parseOK = false;
    QString listItemDisplayText;
    QRgb rgbColor;

    if(uavObject!="")
    {
        //Set the properties of the newly added list item
        listItemDisplayText = uavObject + "." + uavField;
        rgbColor = (QRgb)varColor.toInt(&parseOK);
        if(!parseOK)
            rgbColor = qRgb(255,0,0);
    }
    else{
        listItemDisplayText = "New graph";
        rgbColor = qRgb(255,0,0);
    }

    QColor color = QColor( rgbColor );
    listWidgetItem->setText(listItemDisplayText);
    listWidgetItem->setTextColor( color );

    //Store some additional data for the plot curve on the list item
    listWidgetItem->setData(Qt::UserRole + 0,QVariant(uavObject));
    listWidgetItem->setData(Qt::UserRole + 1,QVariant(uavField));
    listWidgetItem->setData(Qt::UserRole + 2,QVariant(scale));
    listWidgetItem->setData(Qt::UserRole + 3,varColor);
    listWidgetItem->setData(Qt::UserRole + 4,QVariant(mean));
    listWidgetItem->setData(Qt::UserRole + 5,QVariant(mathFunction));
}


void ScopeGadgetOptionsPage::finish()
{

}


/**
 * @brief ScopeGadgetOptionsPage::on_lst2dCurves_currentRowChanged When a different plot
 * curve config is selected, populate its values into the widgets.
 * @param currentRow
 */
void ScopeGadgetOptionsPage::on_lst2dCurves_currentRowChanged(int currentRow)
{
    Q_UNUSED(currentRow);
    set2dYAxisWidgetFromDataSource();
}

void ScopeGadgetOptionsPage::on_cmbXAxisScatterplot2d_currentIndexChanged(QString currentText)
{
    if (currentText == "Series"){
        options_page->spnDataSize->setSuffix(" samples");
    }
    else if (currentText == "Time series"){
        options_page->spnDataSize->setSuffix(" seconds");
    }
}

void ScopeGadgetOptionsPage::on_cmb2dPlotType_currentIndexChanged(QString currentText)
{
    if (currentText == "Polar plot"){
//        options_page->spnDataSize->setSuffix(" samples");
//        options_page->sw2dXAxis->setCurrentWidget(options_page->sw2dSeriesStack);
    }
    else if (currentText == "Scatter plot"){
        options_page->sw2dXAxis->setCurrentWidget(options_page->sw2dSeriesStack);
        on_cmbXAxisScatterplot2d_currentIndexChanged(options_page->cmbXAxisScatterplot2d->currentText());
    }
    else if (currentText == "Histogram"){
        options_page->spnMaxNumBins->setSuffix(" bins");
        options_page->sw2dXAxis->setCurrentWidget(options_page->sw2dHistogramStack);
    }
}


void ScopeGadgetOptionsPage::on_cmb3dPlotType_currentIndexChanged(QString currentText)
{
    if (currentText == "Spectrogram"){
        options_page->stackedWidget3dPlots->setCurrentWidget(options_page->sw3dSpectrogramStack);

        //Set the spectrogram source combobox to vibration test by default
        options_page->cmbSpectrogramSource->setCurrentIndex(options_page->cmbSpectrogramSource->findData(VIBRATIONTEST));
    }
    else if (currentText == "Time series"){
        options_page->stackedWidget3dPlots->setCurrentWidget(options_page->sw3dTimeSeriesStack);
    }
}


void ScopeGadgetOptionsPage::on_tabWidget2d3d_currentIndexChanged(int)
{

}
