/**
 ******************************************************************************
 * @file       mainwindow.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 * @addtogroup crashreporterapp
 * @{
 * @addtogroup MainWindow
 * @{
 * @brief Sends a crash report to the forum
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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFile>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void setDumpFile(QString filename);

private:
    Ui::MainWindow *ui;
    QString dumpFile;
private slots:
    void onShowReport();
    void onSaveReport();
    void onOkSend();
    void onSendReport();
    void onCancelSend();
};

#endif // MAINWINDOW_H
