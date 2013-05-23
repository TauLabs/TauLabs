/**
 ******************************************************************************
 *
 * @file       mainwindow.h
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup Tools
 * @{
 * @brief A Makefile GUI tool
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
#include <QProcess>
#include <QMap>
#include <QScrollArea>
#include <QDir>
#include <QList>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
    struct makefileOptions
    {
        QString brief;
        QStringList option;
        QStringList defaultValue;
    };

    struct makefileGroup
    {
        QString brief;
        QString title;
        QList<makefileOptions*> options;
    };

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_tool_installPB_clicked();
    void onProcStandardOutputAvail();
    void onProcErrorOutputAvail();
    void on_tool_cleanPB_clicked();
    bool eventFilter(QObject *o, QEvent *e);
    void on_targetsCB_currentIndexChanged(int index);
    void on_buildPB_clicked();
    void on_cleanPB_clicked();
    void on_programPB_clicked();
    void on_actionSave_triggered();
    void on_actionLoad_triggered();
    void onVerbosityAction(bool);
    void onJobsAction(bool);
    void onProcStarted();
    void onProcFinish(int, QProcess::ExitStatus exitStatus);
    void on_pushButton_clicked();
    void on_buildAllPB_clicked();
    void on_cleanAllPB_clicked();
    void on_gcsBuildPB_clicked();
    void on_gcsCleanPB_clicked();
    void on_cancelProcessPB_clicked();
    void startProcess(QStringList arguments);
private:
    void processButtonsSetEnabled(bool val);
    void loadFlightGui();
    void loadFlightTargets();
    void parseMakefile(QString path, QString target);
    QList<makefileOptions *> parseMakefileOptions(QStringList::const_iterator start,QStringList::const_iterator end);
    QDir toolPath;
    QDir projectRoot;
    QDir flightTargetsPath;
    QDir toolMakePath;
    Ui::MainWindow *ui;
    void loadToolTargets();
    QProcess * proc;
    QString parseMakefileTag(QString str, QString tag);
    QMap<QString,QList<makefileGroup*> *>makefileGroups;
    QMap<QWidget *,QScrollArea *> scrollHandle;
    QList<QAction*> verbosityActions;
    QList<QAction*> jobsActions;
    QString jobsStr;
    QString verbStr;
};

#endif // MAINWINDOW_H
