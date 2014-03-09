/**
 ******************************************************************************
 * @file       telemetrymonitorwidget.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2011-2012.
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief Provides a compact summary of telemetry on the tool bar
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
#include "telemetrymonitorwidget.h"

#include <QObject>
#include <QtGui>
#include <QFont>
#include <QDebug>

TelemetryMonitorWidget::TelemetryMonitorWidget(QWidget *parent) : QGraphicsView(parent)
{
    setMinimumSize(200,parent->height());
    setMaximumSize(200,parent->height());
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setAlignment(Qt::AlignCenter);
    setFrameStyle(QFrame::NoFrame);
    setStyleSheet("QGraphicsView{background:transparent;}");

    setAttribute(Qt::WA_TranslucentBackground);
    setWindowFlags(Qt::FramelessWindowHint);

    // This comes from the size of the image
    QGraphicsScene *scene = new QGraphicsScene(0,0,1040,110, this);

    renderer = new QSvgRenderer();
    if (renderer->load(QString(":/core/images/tx-rx.svg"))) {
        graph = new QGraphicsSvgItem();
        graph->setSharedRenderer(renderer);
        graph->setElementId("txrxBackground");

        QString name;
        QGraphicsSvgItem* pt;

        QRectF orig;
        QMatrix Matrix;
        QTransform trans;

        for (int i=0; i<NODE_NUMELEM; i++) {
            name = QString("tx%0").arg(i);
            if (renderer->elementExists(name)) {
                trans.reset();
                pt = new QGraphicsSvgItem();
                pt->setSharedRenderer(renderer);
                pt->setElementId(name);
                pt->setParentItem(graph);
                orig=renderer->boundsOnElement(name);
                Matrix = renderer->matrixForElement(name);
                orig=Matrix.mapRect(orig);
                trans.translate(orig.x(),orig.y());
                pt->setTransform(trans,false);
                txNodes.append(pt);
            }

            name = QString("rx%0").arg(i);
            if (renderer->elementExists(name)) {
                trans.reset();
                pt = new QGraphicsSvgItem();
                pt->setSharedRenderer(renderer);
                pt->setElementId(name);
                pt->setParentItem(graph);
                orig=renderer->boundsOnElement(name);
                Matrix = renderer->matrixForElement(name);
                orig=Matrix.mapRect(orig);
                trans.translate(orig.x(),orig.y());
                pt->setTransform(trans,false);
                rxNodes.append(pt);
            }
        }
        scene->addItem(graph);
        txSpeed = new QGraphicsTextItem();
        txSpeed->setDefaultTextColor(Qt::white);
        txSpeed->setFont(QFont("Helvetica",22,2));
        txSpeed->setParentItem(graph);
        orig=renderer->boundsOnElement("txPH");
        Matrix = renderer->matrixForElement("txPH");
        orig=Matrix.mapRect(orig);
        trans.reset();
        trans.translate(orig.x(),orig.y());
        txSpeed->setTransform(trans,false);
        rxSpeed = new QGraphicsTextItem();
        rxSpeed->setDefaultTextColor(Qt::white);
        rxSpeed->setFont(QFont("Helvetica",22,2));
        rxSpeed->setParentItem(graph);
        trans.reset();
        orig=renderer->boundsOnElement("rxPH");
        Matrix = renderer->matrixForElement("rxPH");
        orig=Matrix.mapRect(orig);
        trans.translate(orig.x(),orig.y());
        rxSpeed->setTransform(trans,false);

        scene->setSceneRect(graph->boundingRect());
        setScene(scene);
    }

    m_connected = false;
    txValue = 0.0;
    rxValue = 0.0;

    setMin(0.0);
    setMax(1200.0);

    showTelemetry();
}

TelemetryMonitorWidget::~TelemetryMonitorWidget()
{
    while (!txNodes.isEmpty())
        delete txNodes.takeFirst();

    while (!rxNodes.isEmpty())
        delete rxNodes.takeFirst();
}

void TelemetryMonitorWidget::connected()
{
    m_connected = true;

    //flash the lights
    updateTelemetry(maxValue, maxValue);
}

void TelemetryMonitorWidget::disconnect()
{
    //flash the lights
    updateTelemetry(maxValue, maxValue);

    m_connected = false;
    updateTelemetry(0.0,0.0);
}

/**
 * @brief Called by the UAVObject which got updated
 * Updates the numeric value and/or the icon if the dial wants this.
 */
void TelemetryMonitorWidget::updateTelemetry(double txRate, double rxRate)
{
    txValue = txRate;
    rxValue = rxRate;

    showTelemetry();
}

/** Converts the value into an percentage:
 * this enables smooth movement in moveIndex below
 */
void TelemetryMonitorWidget::showTelemetry()
{
    txIndex = (txValue-minValue)/(maxValue-minValue) * NODE_NUMELEM;
    rxIndex = (rxValue-minValue)/(maxValue-minValue) * NODE_NUMELEM;

    if (m_connected)
        this->setToolTip(QString("Tx: %0 bytes/sec\nRx: %1 bytes/sec").arg(txValue).arg(rxValue));
    else
        this->setToolTip(QString("Disconnected"));

    int i;
    QGraphicsItem* node;

    for (i=0; i < txNodes.count(); i++) {
        node = txNodes.at(i);
        node->setVisible(m_connected && i < txIndex);
        node->update();
    }

    for (i=0; i < rxNodes.count(); i++) {
        node = rxNodes.at(i);
        node->setVisible(m_connected && i < rxIndex);
        node->update();
    }

    txSpeed->setPlainText(QString("%0").arg(txValue));
    txSpeed->setVisible(m_connected);

    rxSpeed->setPlainText(QString("%0").arg(rxValue));
    rxSpeed->setVisible(m_connected);

    update();
}

void TelemetryMonitorWidget::showEvent(QShowEvent *event)
{
    Q_UNUSED(event);

    fitInView(graph, Qt::KeepAspectRatio);
}

void TelemetryMonitorWidget::resizeEvent(QResizeEvent* event)
{
    Q_UNUSED(event);

    // This offset is required because the widget is not centered while
    // it is a child of the connection selector widget
    graph->setPos(0,-20);
    fitInView(graph, Qt::KeepAspectRatio);
}

