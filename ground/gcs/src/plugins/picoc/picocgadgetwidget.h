/**
 ******************************************************************************
 * @file       picocgadgetwidget.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2014
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup PicoCGadgetPlugin PicoC Gadget Plugin
 * @{
 * @brief A gadget to edit PicoC scripts
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

#ifndef PICOCGADGETWIDGET_H_
#define PICOCGADGETWIDGET_H_

#include <QLabel>
#include "ui_picoc.h"
#include "picocstatus.h"
#include "picocsettings.h"
#include "picoc_global.h"

class PICOC_EXPORT PicoCGadgetWidget : public QLabel
{
    Q_OBJECT

public:
    PicoCGadgetWidget(QWidget *parent = 0);
    ~PicoCGadgetWidget();

private:
    Ui_PicoCWidget      *ui;
    UAVObjectManager    *objManager;
    PicoCStatus         *pcStatus;
    PicoCSettings       *pcSettings;

    void fetchScript();
    void sendScript();
    bool waitForExecution();
    void sleep(int);
    void setProgress(int);

private slots:
    void on_tbNewFile_clicked();
    void on_tbLoadFromFile_clicked();
    void on_tbSaveToFile_clicked();
    void on_tbFetchFromUAV_clicked();
    void on_tbSendToUAV_clicked();
    void on_tbFetchFromUAVROM_clicked();
    void on_tbSendToUAVROM_clicked();
    void on_tbEraseFromUAVROM_clicked();
    void on_tbStartScript_clicked();
    void updatePicoCStatus(UAVObject *);
    void on_tbTestValueSend_clicked();
};

#endif /* PICOCGADGETWIDGET_H_ */
