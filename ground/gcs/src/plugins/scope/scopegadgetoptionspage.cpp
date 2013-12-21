/**
 ******************************************************************************
 *
 * @file       scopegadgetoptionspage.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
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

#include "scopegadgetoptionspage.h"

#include "extensionsystem/pluginmanager.h"
#include "uavobjectmanager.h"
#include "uavdataobject.h"

#include "vibrationanalysissettings.h"
#include "vibrationanalysisoutput.h"

#include "scopes2d/histogramscopeconfig.h"
#include "scopes2d/scatterplotscopeconfig.h"
#include "scopes3d/spectrogramscopeconfig.h"

#include <qpalette.h>
#include <QMessageBox>


ScopeGadgetOptionsPage::ScopeGadgetOptionsPage(ScopeGadgetConfiguration *config, QObject *parent) :
        IOptionsPage(parent),
        m_config(config),
        selectedItem(0)
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

    // main widget
    QWidget *optionsPageWidget = new QWidget;

    // Generate UI layout
    options_page = new Ui::ScopeGadgetOptionsPage();
    options_page->setupUi(optionsPageWidget);

    // Set up 2D plots tab
    options_page->cmb2dPlotType->addItem("Scatter plot", Scopes2dConfig::SCATTERPLOT2D);
    options_page->cmb2dPlotType->addItem("Histogram", Scopes2dConfig::HISTOGRAM);

    // Set up x-axis combo box
    options_page->cmbXAxisScatterplot2d->addItem("Series", Scatterplot2dScopeConfig::SERIES2D);
    options_page->cmbXAxisScatterplot2d->addItem("Time series", Scatterplot2dScopeConfig::TIMESERIES2D);


    // Set up 3D plots tab
    options_page->cmb3dPlotType->addItem("Spectrogram", Scopes3dConfig::SPECTROGRAM);

    options_page->cmbSpectrogramSource->addItem("Custom", SpectrogramScopeConfig::CUSTOM_SPECTROGRAM);
    options_page->cmbSpectrogramSource->addItem("Vibration Analysis", SpectrogramScopeConfig::VIBRATIONANALYSIS);

    // Populate colormap combobox.
    options_page->cmbColorMapSpectrogram->addItem("Standard", ColorMap::STANDARD);
    options_page->cmbColorMapSpectrogram->addItem("Jet", ColorMap::JET);

    // Fills the combo boxes for the UAVObjects
    ExtensionSystem::PluginManager *pm = ExtensionSystem::PluginManager::instance();
    UAVObjectManager *objManager = pm->getObject<UAVObjectManager>();
    QVector< QVector<UAVDataObject*> > objList = objManager->getDataObjectsVector();
    foreach (QVector<UAVDataObject*> list, objList) {
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

    // Add scaling items
    options_page->cmbScale->addItem("10^-9", -9);
    options_page->cmbScale->addItem("10^-6", -6);
    options_page->cmbScale->addItem("10^-5",-5);
    options_page->cmbScale->addItem("10^-4",-4);
    options_page->cmbScale->addItem("10^-3",-3);
    options_page->cmbScale->addItem(".01",-2);
    options_page->cmbScale->addItem(".1",-1);
    options_page->cmbScale->addItem("1",0);
    options_page->cmbScale->addItem("10",1);
    options_page->cmbScale->addItem("100",2);
    options_page->cmbScale->addItem("10^3",3);
    options_page->cmbScale->addItem("10^4",4);
    options_page->cmbScale->addItem("10^5",5);
    options_page->cmbScale->addItem("10^6",6);
    options_page->cmbScale->addItem("10^9",9);
    options_page->cmbScale->addItem("10^12",12);

    // Set default scaling to 10^0
    options_page->cmbScale->setCurrentIndex(options_page->cmbScale->findData(0));

    // Configure color button
    options_page->btnColor->setAutoFillBackground(true);

    // Generate style sheet for data sources list
    dataSourceStyleSheetTemplate = "QListView::item:selected {border: 2px solid white; background: rgba(255,255,255,50); selection-color: rgba(%1,%2,%3,255) }";


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
    connect(options_page->lst2dCurves, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(on_lst2dItem_clicked(QListWidgetItem *)));

    // Configuration the GUI elements to reflect the scope settings
    if(m_config)
        m_config->getScope()->setGuiConfiguration(options_page);

    // Cascading update on the UI elements
    emit on_cmb2dPlotType_currentIndexChanged(options_page->cmb2dPlotType->currentText());
    emit on_cmb3dPlotType_currentIndexChanged(options_page->cmb3dPlotType->currentText());
    emit on_cmbUAVObjectsSpectrogram_currentIndexChanged(options_page->cmbUAVObjectsSpectrogram->currentText());


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


/**
 * @brief ScopeGadgetOptionsPage::on_cmbSpectrogramSource_currentIndexChanged Handles special data sources for the spectrogram
 * @param currentText
 */
void ScopeGadgetOptionsPage::on_cmbSpectrogramSource_currentIndexChanged(QString currentText)
{
    if (currentText == options_page->cmbSpectrogramSource->itemText(options_page->cmbSpectrogramSource->findData(SpectrogramScopeConfig::VIBRATIONANALYSIS))){
        int vibrationTestIdx = options_page->cmbUAVObjectsSpectrogram->findText("VibrationTestOutput");
        options_page->cmbUAVObjectsSpectrogram->setCurrentIndex(vibrationTestIdx);
        options_page->cmbUAVObjectsSpectrogram->setEnabled(false);

        // Load UAVO
        ExtensionSystem::PluginManager* pm = ExtensionSystem::PluginManager::instance();
        UAVObjectManager* objManager = pm->getObject<UAVObjectManager>();
        VibrationAnalysisOutput* vibrationAnalysisOutput = VibrationAnalysisOutput::GetInstance(objManager);
        VibrationAnalysisSettings* vibrationAnalysisSettings = VibrationAnalysisSettings::GetInstance(objManager);
        VibrationAnalysisSettings::DataFields vibrationAnalysisSettingsData = vibrationAnalysisSettings->getData();

        // Set combobox field to UAVO name
        options_page->cmbUAVObjectsSpectrogram->setCurrentIndex(options_page->cmbUAVObjectsSpectrogram->findText(vibrationAnalysisOutput->getName()));
        // Get the window size
        int fftWindowSize;
        switch(vibrationAnalysisSettingsData.FFTWindowSize)
        {
        default:
        case VibrationAnalysisSettings::FFTWINDOWSIZE_16 :
            fftWindowSize = 16;
            break;
        case VibrationAnalysisSettings::FFTWINDOWSIZE_64 :
            fftWindowSize = 64;
            break;
        case VibrationAnalysisSettings::FFTWINDOWSIZE_256 :
            fftWindowSize = 256;
            break;
        case VibrationAnalysisSettings::FFTWINDOWSIZE_1024 :
            fftWindowSize = 1024;
            break;
        }

        // Set spinbox range before setting value
        options_page->sbSpectrogramWidth->setRange(0, fftWindowSize / 2);

        // Set values to UAVO
        options_page->sbSpectrogramWidth->setValue(fftWindowSize / 2);
        options_page->sbSpectrogramFrequency->setValue(1000.0f/vibrationAnalysisSettingsData.SampleRate); // Sample rate is in ms

        options_page->sbSpectrogramFrequency->setEnabled(false);
        options_page->sbSpectrogramWidth->setEnabled(false);

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

    // Fetch data from teh listItem. The data is stored by user role + offset
    int currentIndex = options_page->cmbUAVObjects->findText( listItem->data(Qt::UserRole + UR_UAVOBJECT).toString());
    options_page->cmbUAVObjects->setCurrentIndex(currentIndex);

    currentIndex = options_page->cmbUAVField->findText( listItem->data(Qt::UserRole + UR_UAVFIELD).toString());
    options_page->cmbUAVField->setCurrentIndex(currentIndex);

    currentIndex = options_page->cmbScale->findData( listItem->data(Qt::UserRole + UR_SCALE), Qt::UserRole, Qt::MatchExactly);
    options_page->cmbScale->setCurrentIndex(currentIndex);

    // Get graph color
    QVariant varColor  = listItem->data(Qt::UserRole + UR_COLOR);
    int rgb = varColor.toInt(&parseOK);
    if (!parseOK)
        rgb = QColor(Qt::red).rgb();

    // Set button color
    setButtonColor(QColor((QRgb) rgb));

    // Set selected color
    QString styleSheet = dataSourceStyleSheetTemplate.arg(QColor((QRgb) rgb).red()).arg(QColor((QRgb) rgb).green()).arg(QColor((QRgb) rgb).blue());
    options_page->lst2dCurves->setStyleSheet(styleSheet);

    unsigned int mean = listItem->data(Qt::UserRole + UR_MEAN).toUInt(&parseOK);
    if(!parseOK)
        mean = 1;
    options_page->spnMeanSamples->setValue(mean);

    currentIndex = options_page->mathFunctionComboBox->findText( listItem->data(Qt::UserRole + UR_MATHFUNCTION).toString());
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
        return;

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
        return;

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
 * @brief ScopeGadgetOptionsPage::apply Called when the user presses OK. Applies the current values to the scope.
 */
void ScopeGadgetOptionsPage::apply()
{
    //The GUI settings are read by the appropriate subclass
    if(m_config)
        m_config->applyGuiConfiguration(options_page);
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

    unsigned int mean = options_page->spnMeanSamples->value();
    QString mathFunction = options_page->mathFunctionComboBox->currentText();

    QVariant varColor = (int)QColor(options_page->btnColor->text()).rgb();

    // Add curve
    addPlot2dCurveConfig(uavObject, uavField, scale, mean, mathFunction, varColor);
}


/**
 * @brief ScopeGadgetOptionsPage::on_btnApply2dCurve_clicked Creates new data sources
 */
void ScopeGadgetOptionsPage::on_btnApply2dCurve_clicked()
{
    bool parseOK = false;
    QString uavObjectName = options_page->cmbUAVObjects->currentText();
    QString uavFieldName = options_page->cmbUAVField->currentText();
    int scale = options_page->cmbScale->itemData(options_page->cmbScale->currentIndex()).toInt(&parseOK);

    if(!parseOK)
       scale = 0;

    unsigned int mean = options_page->spnMeanSamples->value();
    QString mathFunction = options_page->mathFunctionComboBox->currentText();

    QVariant varColor = (int)QColor(options_page->btnColor->text()).rgb();

    // Apply curve settings
    QListWidgetItem *listWidgetItem = options_page->lst2dCurves->currentItem();
    if(listWidgetItem == NULL){
        QMessageBox msgBox;
        msgBox.setText(tr("No curve selected."));
        msgBox.setInformativeText(tr("Please select a curve or generate one with the ""+"" symbol."));
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


/**
 * @brief ScopeGadgetOptionsPage::addPlot2dCurveConfig Add a new curve config to the list
 * @param uavObjectName UAVO name
 * @param uavFieldName UAVO fiel
 * @param scale Scale multiplier
 * @param mean Number of samples in mean
 * @param mathFunction Math function to be performed on data
 * @param varColor Plotted color
 */
void ScopeGadgetOptionsPage::addPlot2dCurveConfig(QString uavObjectName, QString uavFieldName, int scale, unsigned int mean, QString mathFunction, QVariant varColor)
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
void ScopeGadgetOptionsPage::setPlot2dCurveProperties(QListWidgetItem *listWidgetItem, QString uavObject, QString uavField, int scale, unsigned int mean, QString mathFunction, QVariant varColor)
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

    // Set text
    listWidgetItem->setText(listItemDisplayText);

    // Set selected color. Both color sets are necessary.
    QColor color = QColor( rgbColor );
    QString styleSheet = dataSourceStyleSheetTemplate.arg(QColor(rgbColor).red()).arg(QColor(rgbColor).green()).arg(QColor(rgbColor).blue());
    listWidgetItem->setTextColor( color ); // This one sets the text color when unselected...
    options_page->lst2dCurves->setStyleSheet(styleSheet); //.. and this one sets the text color when selected

    //Store some additional data for the plot curve on the list item
    listWidgetItem->setData(Qt::UserRole + UR_UAVOBJECT,QVariant(uavObject));
    listWidgetItem->setData(Qt::UserRole + UR_UAVFIELD,QVariant(uavField));
    listWidgetItem->setData(Qt::UserRole + UR_SCALE,QVariant(scale));
    listWidgetItem->setData(Qt::UserRole + UR_COLOR,varColor);
    listWidgetItem->setData(Qt::UserRole + UR_MEAN,QVariant(mean));
    listWidgetItem->setData(Qt::UserRole + UR_MATHFUNCTION,QVariant(mathFunction));
}


//TODO: Document why finish() is here and what it's supposed to do. Clearly nothing right now.
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

/**
 * @brief ScopeGadgetOptionsPage::on_cmbXAxisScatterplot2d_currentIndexChanged Updates the combobox text
 * @param currentText
 */
void ScopeGadgetOptionsPage::on_cmbXAxisScatterplot2d_currentIndexChanged(QString currentText)
{
    if (currentText == "Series"){
        options_page->spnDataSize->setSuffix(" samples");
    }
    else if (currentText == "Time series"){
        options_page->spnDataSize->setSuffix(" seconds");
    }
}


/**
 * @brief ScopeGadgetOptionsPage::on_cmb2dPlotType_currentIndexChanged Updates the combobox text
 * @param currentText
 */
void ScopeGadgetOptionsPage::on_cmb2dPlotType_currentIndexChanged(QString currentText)
{
    if (currentText == "Scatter plot"){
        options_page->sw2dXAxis->setCurrentWidget(options_page->sw2dSeriesStack);
        on_cmbXAxisScatterplot2d_currentIndexChanged(options_page->cmbXAxisScatterplot2d->currentText());
    }
    else if (currentText == "Histogram"){
        options_page->spnMaxNumBins->setSuffix(" bins");
        options_page->sw2dXAxis->setCurrentWidget(options_page->sw2dHistogramStack);
    }
}


/**
 * @brief ScopeGadgetOptionsPage::on_lst2dItem_clicked If a 2d data source is clicked in the listview, toggle its selected state
 * @param listItem
 */
void ScopeGadgetOptionsPage::on_lst2dItem_clicked(QListWidgetItem * listItem)
{
    if (listItem == selectedItem)
    {
        listItem->setSelected(false);
        options_page->lst2dCurves->setCurrentRow(-1);
        selectedItem = 0;
    }
    else{
        selectedItem = listItem;
    }
}


/**
 * @brief ScopeGadgetOptionsPage::on_cmb3dPlotType_currentIndexChanged
 * @param currentText
 */
void ScopeGadgetOptionsPage::on_cmb3dPlotType_currentIndexChanged(QString currentText)
{
    if (currentText == "Spectrogram"){
        options_page->stackedWidget3dPlots->setCurrentWidget(options_page->sw3dSpectrogramStack);

        //Set the spectrogram source combobox to custom spectrogram by default
        options_page->cmbSpectrogramSource->setCurrentIndex(options_page->cmbSpectrogramSource->findData(SpectrogramScopeConfig::CUSTOM_SPECTROGRAM));
    }
    else if (currentText == "Time series"){
        options_page->stackedWidget3dPlots->setCurrentWidget(options_page->sw3dTimeSeriesStack);
    }
}
