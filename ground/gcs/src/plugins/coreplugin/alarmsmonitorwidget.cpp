/**
 ******************************************************************************
 * @file       alarmsmonitorwidget.cpp
 * @author     PhoenixPilot, http://github.com/PhoenixPilot, Copyright (C) 2012
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief Provides a compact summary of alarms on the tool bar
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
#include "alarmsmonitorwidget.h"
#include <QFont>
#include <coreplugin/icore.h>
#include <globalmessaging.h>

#define DIMMED_SYMBOL 0.1

AlarmsMonitorWidget::AlarmsMonitorWidget():hasErrors(false),hasWarnings(false),hasInfos(false),needsUpdate(false)
{
}

void AlarmsMonitorWidget::init(QSvgRenderer *renderer,QGraphicsSvgItem *graph)
{
    error_sym=new QGraphicsSvgItem();
    error_sym->setSharedRenderer(renderer);
    error_sym->setElementId("error_sym");
    error_sym->setParentItem(graph);
    error_txt = new QGraphicsTextItem();
    error_txt->setDefaultTextColor(Qt::white);
    error_txt->setFont(QFont("Helvetica",22,2));
    error_txt->setParentItem(graph);
    error_txt->setPlainText("0");

    QRectF orig=renderer->boundsOnElement("error_sym");
    QMatrix Matrix = renderer->matrixForElement("error_sym");
    orig=Matrix.mapRect(orig);
    QTransform trans;
    trans.translate(orig.x(),orig.y());
    error_sym->setTransform(trans,false);
    trans.reset();
    int refY=orig.y();
    trans.translate(orig.x()+orig.width()-5,refY);
    error_txt->setTransform(trans,false);
    trans.reset();

    info_sym=new QGraphicsSvgItem();
    info_sym->setSharedRenderer(renderer);
    info_sym->setElementId("info_sym");
    info_sym->setParentItem(graph);
    info_txt = new QGraphicsTextItem();
    info_txt->setDefaultTextColor(Qt::white);
    info_txt->setFont(QFont("Helvetica",22,2));
    info_txt->setParentItem(graph);
    info_txt->setPlainText("0");
    orig=renderer->boundsOnElement("info_sym");
    Matrix = renderer->matrixForElement("info_sym");
    orig=Matrix.mapRect(orig);
    trans.translate(orig.x(),orig.y());
    info_sym->setTransform(trans,false);
    trans.reset();
    trans.translate(orig.x()-5+orig.width(),refY);
    info_txt->setTransform(trans,false);
    trans.reset();

    warning_sym=new QGraphicsSvgItem();
    warning_sym->setSharedRenderer(renderer);
    warning_sym->setElementId("warning_sym");
    warning_sym->setParentItem(graph);
    warning_txt = new QGraphicsTextItem();
    warning_txt->setDefaultTextColor(Qt::white);
    warning_txt->setFont(QFont("Helvetica",22,2));
    warning_txt->setParentItem(graph);
    warning_txt->setPlainText("0");
    orig=renderer->boundsOnElement("warning_sym");
    Matrix = renderer->matrixForElement("warning_sym");
    orig=Matrix.mapRect(orig);
    trans.translate(orig.x(),orig.y());
    warning_sym->setTransform(trans,false);
    trans.reset();
    trans.translate(orig.x()+orig.width()-20,refY);
    warning_txt->setTransform(trans,false);
    trans.reset();
    error_sym->setOpacity(0.1);
    warning_sym->setOpacity(0.1);
    info_sym->setOpacity(0.1);
    error_txt->setOpacity(0.1);
    warning_txt->setOpacity(0.1);
    info_txt->setOpacity(0.1);

    connect(&alertTimer,SIGNAL(timeout()),this,SLOT(processAlerts()));
    connect(Core::ICore::instance()->globalMessaging(),SIGNAL(newMessage(GlobalMessage*)),this,SLOT(updateMessages()));
    connect(Core::ICore::instance()->globalMessaging(),SIGNAL(deletedMessage()),this,SLOT(updateMessages()));
    connect(Core::ICore::instance()->globalMessaging(),SIGNAL(changedMessage(GlobalMessage*)),this,SLOT(updateNeeded()));

    alertTimer.start(1000);

}


void AlarmsMonitorWidget::updateMessages()
{
    QString error;
    error.append("<html><head/><body>");
    foreach(Core::GlobalMessage * msg,Core::ICore::instance()->globalMessaging()->getActiveErrors())
    {
        QString temp;
        temp=QString("<p><span style=' font-size:11pt; font-weight:600;'>%0</span></p>").arg(msg->getBrief());
        error.append(temp);
        temp=QString("<p><span style=' font-style:italic;'>%0</span></p>").arg(msg->getDescription());
        error.append(temp);
    }
    error.append("</body></html>");
    QString warning;
    warning.append("<html><head/><body>");
    foreach(Core::GlobalMessage * msg,Core::ICore::instance()->globalMessaging()->getActiveWarnings())
    {
        QString temp;
        temp=QString("<p><span style=' font-size:11pt; font-weight:600;'>%0</span></p>").arg(msg->getBrief());
        warning.append(temp);
        temp=QString("<p><span style=' font-style:italic;'>%0</span></p>").arg(msg->getDescription());
        warning.append(temp);
    }
    warning.append("</body></html>");
    QString info;
    info.append("<html><head/><body>");
    foreach(Core::GlobalMessage * msg,Core::ICore::instance()->globalMessaging()->getActiveInfos())
    {
        QString temp;
        temp=QString("<p><span style=' font-size:11pt; font-weight:600;'>%0</span></p>").arg(msg->getBrief());
        info.append(temp);
        temp=QString("<p><span style=' font-style:italic;'>%0</span></p>").arg(msg->getDescription());
        info.append(temp);
    }
    info.append("</body></html>");
    error_txt->setPlainText(QString::number(Core::ICore::instance()->globalMessaging()->getActiveErrors().length()));
    if(Core::ICore::instance()->globalMessaging()->getActiveErrors().length()>0)
    {
        error_txt->setOpacity(1);
        hasErrors=true;
    }
    else
    {
        error="No errors";
        error_txt->setOpacity(DIMMED_SYMBOL);
        hasErrors=false;
    }
    warning_txt->setPlainText(QString::number(Core::ICore::instance()->globalMessaging()->getActiveWarnings().length()));
    if(Core::ICore::instance()->globalMessaging()->getActiveWarnings().length()>0)
    {
        warning_txt->setOpacity(1);
        hasWarnings=true;
    }
    else
    {
        warning="No warnings";
        warning_txt->setOpacity(DIMMED_SYMBOL);
        hasWarnings=false;
    }
    info_txt->setPlainText(QString::number(Core::ICore::instance()->globalMessaging()->getActiveInfos().length()));
    if(Core::ICore::instance()->globalMessaging()->getActiveInfos().length()>0)
    {
        info_txt->setOpacity(1);
        hasInfos=true;
    }
    else
    {
        info="No info";
        info_txt->setOpacity(DIMMED_SYMBOL);
        hasInfos=false;
    }
    error_sym->setToolTip(error);
    warning_sym->setToolTip(warning);
    info_sym->setToolTip(info);
}

void AlarmsMonitorWidget::updateNeeded()
{
    needsUpdate=true;
}

void AlarmsMonitorWidget::processAlerts()
{
    if(needsUpdate)
        updateMessages();;
    needsUpdate=false;
    static bool flag=true;
    flag = flag ^ true;
    if(hasErrors)
    {
        if(flag)
            error_sym->setOpacity(1);
        else
            error_sym->setOpacity(DIMMED_SYMBOL);
    }
    else
        error_sym->setOpacity(DIMMED_SYMBOL);
    if(hasWarnings)
    {
        if(flag)
            warning_sym->setOpacity(1);
        else
            warning_sym->setOpacity(DIMMED_SYMBOL);
    }
    else
        warning_sym->setOpacity(DIMMED_SYMBOL);
    if(hasInfos)
    {
        if(flag)
            info_sym->setOpacity(1);
        else
            info_sym->setOpacity(DIMMED_SYMBOL);
    }
    else
        info_sym->setOpacity(DIMMED_SYMBOL);
}
