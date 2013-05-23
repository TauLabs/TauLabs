/**
 ******************************************************************************
 *
 * @file       mainwindow.cpp
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

#include <mainwindow.h>
#include <QApplication>
#include <QTextStream>
#include <QCheckBox>
#include "ui_mainwindow.h"
#include <QScrollBar>
#include <QSettings>
#include <QFileDialog>
#include <QGroupBox>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    toolPath=QDir(QApplication::applicationDirPath());
    projectRoot=toolPath;
    projectRoot.cdUp();
    projectRoot.cdUp();
#ifdef Q_WS_MAC
    projectRoot.cdUp();
    projectRoot.cdUp();
    projectRoot.cdUp();
#endif
    flightTargetsPath=projectRoot;
    flightTargetsPath.cd("flight");
    flightTargetsPath.cd("targets");
    toolMakePath=projectRoot;
    toolMakePath.cd("make");
    ui->setupUi(this);
    verbosityActions<<ui->action0<<ui->action1<<ui->action2;
    jobsActions<<ui->action1_2<<ui->action2_2<<ui->action3<<ui->action4<<ui->action5<<ui->action6<<ui->action7<<ui->action8;
    foreach(QAction * act,verbosityActions)
        connect(act,SIGNAL(toggled(bool)),this,SLOT(onVerbosityAction(bool)));
    foreach(QAction * act,jobsActions)
        connect(act,SIGNAL(toggled(bool)),this,SLOT(onJobsAction(bool)));
    ui->action0->setChecked(true);
    ui->action1_2->setChecked(true);
    proc=new QProcess(this);
    connect(proc,SIGNAL(started()),this,SLOT(onProcStarted()));
    connect(proc,SIGNAL(finished(int,QProcess::ExitStatus)),this,SLOT(onProcFinish(int,QProcess::ExitStatus)));
    proc->setWorkingDirectory(projectRoot.absolutePath());
    connect(proc,SIGNAL(readyReadStandardError()),this,SLOT(onProcErrorOutputAvail()));
    connect(proc,SIGNAL(readyReadStandardOutput()),this,SLOT(onProcStandardOutputAvail()));
    loadToolTargets();
    loadFlightTargets();
    QFont font("Monospace");
    font.setStyleHint(QFont::TypeWriter);
    ui->debugOutput->setFont(font);
}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::on_tool_installPB_clicked()
{
    QStringList arguments;
    foreach(QCheckBox * box,ui->toolsTab->findChildren<QCheckBox *>())
    {
        if(box->isChecked())
        {
            arguments<<box->text()+"_install";
        }
    }
    arguments<<verbStr<<jobsStr;
    startProcess(arguments);
}

void MainWindow::startProcess(QStringList arguments)
{
    QString temp;
    foreach(QString str, arguments)
        temp +=str + " ";
    temp += "\n\r";
    ui->debugOutput->insertPlainText("make " + temp);
    QTextCursor c =  ui->debugOutput->textCursor();
    c.movePosition(QTextCursor::End);
    ui->debugOutput->setTextCursor(c);
    proc->start("make",arguments);
}

void MainWindow::onProcStandardOutputAvail()
{
    QString str=proc->readAllStandardOutput();
    ui->debugOutput->insertPlainText(str);
    QTextCursor c =  ui->debugOutput->textCursor();
    c.movePosition(QTextCursor::End);
    ui->debugOutput->setTextCursor(c);
}


void MainWindow::onProcErrorOutputAvail()
{
    QString str=proc->readAllStandardError();
    QColor backup=ui->debugOutput->textColor();
    ui->debugOutput->setTextColor(Qt::red);
    ui->debugOutput->insertPlainText(str);
    ui->debugOutput->setTextColor(backup);
    ui->debugOutput->insertPlainText("\r");
    QTextCursor c =  ui->debugOutput->textCursor();
    c.movePosition(QTextCursor::End);
    ui->debugOutput->setTextCursor(c);
}


void MainWindow::on_tool_cleanPB_clicked()
{
    QStringList arguments;
    foreach(QCheckBox * box,ui->toolsTab->findChildren<QCheckBox *>())
    {
        if(box->isChecked())
        {
            arguments<<box->text()+"_clean";
        }
    }
    arguments<<verbStr<<jobsStr;
    startProcess(arguments);
}


void MainWindow::loadFlightTargets()
{
    QStringList list;
    list=flightTargetsPath.entryList();
    list.removeAll("..");
    list.removeAll(".");
    list.removeAll("board_hw_defs");
    list.removeAll("Bootloaders");
    list.removeAll("EntireFlash");
    list.removeAll("UAVObjects");
    list.removeAll("UAVTalk");
    list.removeAll("Project");
    foreach(QString str,list)
    {
        ui->targetsCB->addItem(str);
    }
    foreach(QString str,list)
    {
        QDir dir=flightTargetsPath;
        dir.cd(str);
        parseMakefile(dir.absoluteFilePath("Makefile"),str.toLower());
    }
    loadFlightGui();

}


void MainWindow::parseMakefile(QString path, QString target)
{
    QList<makefileGroup*> * newGroup=new QList<makefileGroup*>();
    makefileGroups.insert(target,newGroup);

    QFile file(path);
    file.open(QIODevice::ReadOnly);
    QTextStream in(&file);
    QString str;
    QStringList list;
    do
    {
        str=in.readLine();
        list<<str;

    }while (str!=QString::null);
    list.removeLast();
    QStringList::const_iterator startGroupIt;
    for (startGroupIt = list.constBegin(); startGroupIt != list.constEnd();++startGroupIt)
    {
        QString groupID=*startGroupIt;
        if(groupID.contains("@startgroup"))
        {
            QString groupStr=parseMakefileTag(groupID,"@startgroup");
            QStringList::const_iterator endGroupIt;
            bool endGroupFound=false;
            QString groupBrief;
            for(endGroupIt=startGroupIt;endGroupIt!=list.constEnd();++endGroupIt)
            {
                QString temp=*endGroupIt;
                if(temp.contains("@groupbrief"))
                {
                    groupBrief=parseMakefileTag(temp,"@groupbrief");
                }
                if(temp.contains("@endgroup") && temp.contains(groupStr))
                {
                    endGroupFound=true;
                    makefileGroup * group=new makefileGroup;
                    group->title=groupStr;
                    group->brief=groupBrief;
                    group->options=parseMakefileOptions(startGroupIt,endGroupIt);
                    makefileGroups.value(target)->append(group);
                }
            }
            if(!endGroupFound)
            {
                QColor backup=ui->debugOutput->textColor();
                ui->debugOutput->setTextColor(Qt::red);
                ui->debugOutput->append(QString("Problem parsing %0 @startgroup %1 tag found without @endgroup").arg(path).arg(groupID));
                ui->debugOutput->setTextColor(backup);
                ui->debugOutput->insertPlainText("\r");
            }
        }
    }

}

QList<MainWindow::makefileOptions *> MainWindow::parseMakefileOptions(QStringList::const_iterator start, QStringList::const_iterator end)
{
    QList<MainWindow::makefileOptions *> optionsList;
    QStringList::const_iterator i;
    QString lastBrief;
    for(i=start;i!=end;++i)
    {
        QString temp=*i;
        if(temp.contains("@optbrief"))
            lastBrief=parseMakefileTag(temp,"@optbrief");
        else if(temp.length()==temp.count(" "))
            lastBrief="";
        else if(temp.contains("?="))
        {
            bool found=false;
            foreach(makefileOptions * option,optionsList)
            {
                if(option->brief==lastBrief)
                {
                    found=true;
                    option->option.append(temp.split("?=").at(0).trimmed());
                    option->defaultValue.append(temp.split("?=").at(1).trimmed());
                }
            }
            if(!found)
            {
                makefileOptions * opt=new makefileOptions;
                opt->brief=lastBrief;
                opt->defaultValue.append(temp.split("?=").at(1).trimmed());
                opt->option.append(temp.split("?=").at(0).trimmed());
                optionsList.append(opt);
            }
        }
    }
    return optionsList;
}

void MainWindow::on_targetsCB_currentIndexChanged(int index)
{
    ui->stackedWidget->setCurrentIndex(index);
}

void MainWindow::loadFlightGui()
{
    foreach(QString target,makefileGroups.keys())
    {
        QWidget * page = new QWidget();
        page->setObjectName(target+QString::fromUtf8("page"));
        QVBoxLayout *pageVerticalLayout = new QVBoxLayout(page);
        pageVerticalLayout->setSpacing(6);
        pageVerticalLayout->setContentsMargins(11, 11, 11, 11);
        pageVerticalLayout->setObjectName(target+QString::fromUtf8("pageVerticalLayout"));
        QTabWidget * tab = new QTabWidget(page);
        tab->setObjectName(target+QString::fromUtf8("tab"));
        QList<makefileGroup*> * groups=makefileGroups.value(target);
        foreach(makefileGroup * group,*groups)
        {
            QWidget * tabWidget = new QWidget();
            tabWidget->setObjectName(target+group->title+QString::fromUtf8("TabWidget"));
            QVBoxLayout * tabWidgetVerticalLayout = new QVBoxLayout(tabWidget);
            tabWidgetVerticalLayout->setSpacing(6);
            tabWidgetVerticalLayout->setContentsMargins(11, 11, 11, 11);
            tabWidgetVerticalLayout->setObjectName(target+group->title+QString::fromUtf8("tabWidgetVerticalLayout"));
            QScrollArea * scrollArea = new QScrollArea(tabWidget);
            scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
            scrollArea->setObjectName(target+group->title+QString::fromUtf8("scrollArea"));
            scrollArea->setWidgetResizable(true);
            QWidget * scrollAreaWidgetContents = new QWidget();
            scrollHandle.insert(scrollAreaWidgetContents,scrollArea);
            scrollAreaWidgetContents->installEventFilter(this);
            scrollAreaWidgetContents->setObjectName(target+group->title+QString::fromUtf8("scrollAreaWidgetContents"));
            scrollAreaWidgetContents->setGeometry(QRect(0, 0, 346, 153));
            QVBoxLayout * scrollAreaVerticalLayout = new QVBoxLayout(scrollAreaWidgetContents);
            scrollAreaVerticalLayout->setSpacing(6);
            scrollAreaVerticalLayout->setContentsMargins(11, 11, 11, 11);
            scrollAreaVerticalLayout->setObjectName(target+QString::fromUtf8("scrollAreaVerticalLayout"));
            foreach(makefileOptions * option,group->options)
            {
                QGroupBox * groupBox = new QGroupBox(scrollAreaWidgetContents);
                groupBox->setTitle(option->brief);
                groupBox->setObjectName(target+group->title+option->brief+QString::fromUtf8("groupBox"));
                QVBoxLayout * groupBoxVerticalLayout = new QVBoxLayout(groupBox);
                groupBoxVerticalLayout->setSpacing(6);
                groupBoxVerticalLayout->setContentsMargins(11, 11, 11, 11);
                groupBoxVerticalLayout->setObjectName(target+group->title+option->brief+QString::fromUtf8("verticalLayout"));
                foreach(QString str,option->option)
                {
                    QCheckBox * checkBox = new QCheckBox(groupBox);
                    checkBox->setObjectName(target+group->title+option->brief+str+QString::fromUtf8("checkBox"));
                    groupBoxVerticalLayout->addWidget(checkBox);
                    checkBox->setText(str);
                    checkBox->setChecked(option->defaultValue.at(option->option.indexOf(str))=="YES"?true:false);
                }
                scrollAreaVerticalLayout->addWidget(groupBox);
            }
            scrollArea->setWidget(scrollAreaWidgetContents);
            tabWidgetVerticalLayout->addWidget(scrollArea);
            tabWidget->setToolTip(group->brief);
            tab->addTab(tabWidget, group->title);
        }
        pageVerticalLayout->addWidget(tab);
        ui->stackedWidget->addWidget(page);
    }
}


void MainWindow::on_buildPB_clicked()
{
    QString target=ui->targetsCB->currentText().toLower();
    QStringList arguments;
    arguments<<target;
    foreach(QCheckBox * box,ui->stackedWidget->currentWidget()->findChildren<QCheckBox *>())
    {
        arguments<<QString("%0=%1").arg(box->text()).arg(box->isChecked()?"YES":"NO");
    }
    arguments<<verbStr<<jobsStr;
    startProcess(arguments);
}


void MainWindow::loadToolTargets()
{
    QFile file(toolMakePath.absoluteFilePath("tools.mk"));
    file.open(QIODevice::ReadOnly);
    QTextStream in(&file);
    QString str;
    QStringList list;
    do
    {
        str=in.readLine();
        if(str.contains(".PHONY:"))
        {
            str=str.split(" ").at(1);
            str.remove("_install");
            str.remove("_clean");
            list<<str.trimmed();
        }
    }
    while(str!=QString::null);
    list.removeDuplicates();
    int col=0;
    int row=0;
    foreach(str,list)
    {
        QCheckBox * box=new QCheckBox(ui->toolsTab);
        box->setObjectName(str+QString::fromUtf8("CheckBox"));
        box->setText(str);
        ui->toolsLayout->addWidget(box,row,col,1,1);
        if(col==0)
            col=1;
        else
        {
            col=0;
            ++row;
        }
    }
}


QString MainWindow::parseMakefileTag(QString str, QString tag)
{
    return str.remove(0,str.indexOf(" ",str.indexOf(tag))).trimmed();
}

bool MainWindow::eventFilter(QObject *o, QEvent *e)
{
    QWidget * widget=qobject_cast<QWidget *>(o);
    if(o == widget && e->type() == QEvent::Resize)
        scrollHandle.value(widget)->setMinimumWidth(widget->minimumSizeHint().width() + scrollHandle.value(widget)->verticalScrollBar()->width());

    return false;
}

void MainWindow::on_cleanPB_clicked()
{
    QString target=ui->targetsCB->currentText().toLower();
    QStringList arguments;
    arguments<<target+"_clean";
    arguments<<verbStr<<jobsStr;
    startProcess(arguments);
}

void MainWindow::on_programPB_clicked()
{
    QString target=ui->targetsCB->currentText().toLower();
    QStringList arguments;
    arguments<<"fw_"+target+"_program";
    foreach(QCheckBox * box,ui->stackedWidget->currentWidget()->findChildren<QCheckBox *>())
    {
        arguments<<QString("%0=%1").arg(box->text()).arg(box->isChecked()?"YES":"NO");
    }
    arguments<<verbStr<<jobsStr;
    startProcess(arguments);
}

void MainWindow::on_actionSave_triggered()
{
    QCoreApplication::setOrganizationName("TauLabs");
    QCoreApplication::setOrganizationDomain("taulabs.org");
    QCoreApplication::setApplicationName("Make GUI");
    QString file=QFileDialog::getSaveFileName(this,"Save template");
    QSettings settings(file,
                       QSettings::NativeFormat);
    foreach(QCheckBox * box,this->findChildren<QCheckBox*>())
    {
        settings.setValue(box->objectName(),box->isChecked());
    }
    foreach(QAction * action,this->findChildren<QAction*>())
    {
        if(action->isCheckable())
            settings.setValue(action->objectName(),action->isChecked());
    }
}

void MainWindow::on_actionLoad_triggered()
{
    QString file=QFileDialog::getOpenFileName(this,"Open template");
    QSettings settings(file,
                       QSettings::NativeFormat);
    foreach(QCheckBox * box,this->findChildren<QCheckBox*>())
    {
        box->setChecked(settings.value(box->objectName()).toBool());
    }
    foreach(QAction * action,this->findChildren<QAction*>())
    {
        if(action->isCheckable())
            action->setChecked(settings.value(action->objectName()).toBool());
    }

}

void MainWindow::onVerbosityAction(bool)
{
    QAction * act=qobject_cast<QAction *>(sender());
    if(!act->isChecked())
    {
        verbosityActions.at(0)->setChecked(true);
        verbStr="V=0";
        return;
    }
    foreach(QAction * action,verbosityActions)
    {
        if(action!=act)
        {
            action->blockSignals(true);
            action->setChecked(false);
            action->blockSignals(false);
        }
    }
    verbStr=QString("V=%0").arg(verbosityActions.indexOf(act));
}

void MainWindow::onJobsAction(bool)
{
    QAction * act=qobject_cast<QAction *>(sender());
    if(!act->isChecked())
    {
        jobsActions.at(0)->setChecked(true);
        jobsStr="-j1";
        return;
    }
    else
    {
        foreach(QAction * action,jobsActions)
        {
            if(action!=act)
            {
                action->blockSignals(true);
                action->setChecked(false);
                action->blockSignals(false);
            }
        }
    }
    jobsStr=QString("-j%0").arg(jobsActions.indexOf(act)+1);
}

void MainWindow::processButtonsSetEnabled(bool val)
{
    ui->buildPB->setEnabled(val);
    ui->cleanPB->setEnabled(val);
    ui->programPB->setEnabled(val);
    ui->tool_installPB->setEnabled(val);
    ui->tool_cleanPB->setEnabled(val);
    ui->buildAllPB->setEnabled(val);
    ui->cleanAllPB->setEnabled(val);
    ui->gcsBuildPB->setEnabled(val);
    ui->gcsCleanPB->setEnabled(val);
}

void MainWindow::onProcStarted()
{
    processButtonsSetEnabled(false);
}

void MainWindow::onProcFinish(int result,QProcess::ExitStatus exitStatus)
{
    QColor backup=ui->debugOutput->textColor();
    if(exitStatus == QProcess::CrashExit)
    {
        ui->debugOutput->setTextColor(Qt::red);
        ui->debugOutput->insertPlainText("BUILD CANCELED");
    }
    else if(result == 0)
    {
        ui->debugOutput->insertPlainText("BUILD SUCCEEDED");
    }
    else
    {
        ui->debugOutput->setTextColor(Qt::red);
        ui->debugOutput->insertPlainText("BUILD FAILED");
    }
    ui->debugOutput->setTextColor(backup);
    ui->debugOutput->insertPlainText("\r");
    QTextCursor c =  ui->debugOutput->textCursor();
    c.movePosition(QTextCursor::End);
    ui->debugOutput->setTextCursor(c);
    processButtonsSetEnabled(true);
}

void MainWindow::on_pushButton_clicked()
{
    QStringList arguments;
    arguments<<"me a sandwich";
    startProcess(arguments);
}

void MainWindow::on_buildAllPB_clicked()
{
    QString target="all";
    QStringList arguments;
    arguments<<target;
    arguments<<verbStr<<jobsStr;
    startProcess(arguments);
}

void MainWindow::on_cleanAllPB_clicked()
{
    QString target="all_clean";
    QStringList arguments;
    arguments<<target;
    arguments<<verbStr<<jobsStr;
    startProcess(arguments);
}

void MainWindow::on_gcsBuildPB_clicked()
{
    QString target="gcs";
    QStringList arguments;
    arguments<<target;
    arguments<<QString("GCS_BUILD_CONF=%0").arg(ui->gcsReleaseRB->isChecked()?"release":"debug");
    arguments<<verbStr<<jobsStr;
    startProcess(arguments);
}

void MainWindow::on_gcsCleanPB_clicked()
{
    QString target="gcs_clean";
    QStringList arguments;
    arguments<<target;
    arguments<<QString("GCS_BUILD_CONF=%0").arg(ui->gcsReleaseRB->isChecked()?"release":"debug");
    startProcess(arguments);
}

void MainWindow::on_cancelProcessPB_clicked()
{
    proc->kill();
}
