/**
 ******************************************************************************
 *
 * @file       connectiondiagram.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2012.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @see        The GNU Public License (GPL) Version 3
 *
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup SetupWizard Setup Wizard
 * @{
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

#include <QDebug>
#include <QFile>
#include <QFileDialog>
#include "connectiondiagram.h"
#include "ui_connectiondiagram.h"

ConnectionDiagram::ConnectionDiagram(QWidget *parent, VehicleConfigurationSource *configSource) :
    QDialog(parent), ui(new Ui::ConnectionDiagram), m_configSource(configSource), m_background(0)
{
    ui->setupUi(this);
    setWindowTitle(tr("Connection Diagram"));
    setupGraphicsScene();
}

ConnectionDiagram::~ConnectionDiagram()
{
    delete ui;
}

void ConnectionDiagram::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);

    ui->connectionDiagram->fitInView(m_background, Qt::KeepAspectRatio);
}

void ConnectionDiagram::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);

    ui->connectionDiagram->fitInView(m_background, Qt::KeepAspectRatio);
}

void ConnectionDiagram::setupGraphicsScene()
{
    Core::IBoardType *board = m_configSource->getControllerType();
    if (!board)
        return;
    QString diagram = board->getConnectionDiagram();
    m_renderer = new QSvgRenderer();
    if (QFile::exists(diagram) && m_renderer->load(diagram) && m_renderer->isValid()) {

        m_scene = new QGraphicsScene(this);
        ui->connectionDiagram->setScene(m_scene);

        m_background = new QGraphicsSvgItem();
        m_background->setSharedRenderer(m_renderer);
        m_background->setElementId("background");
        m_background->setOpacity(0);
        m_background->setZValue(-1);
        m_scene->addItem(m_background);

        QList<QString> elementsToShow;

        Core::IBoardType* type = m_configSource->getControllerType();
        if (type != NULL)
            elementsToShow << QString("controller-").append(type->shortName().toLower());

        switch (m_configSource->getVehicleType()) {
        case VehicleConfigurationSource::VEHICLE_MULTI:
            switch (m_configSource->getVehicleSubType()) {
            case VehicleConfigurationSource::MULTI_ROTOR_TRI_Y:
                elementsToShow << "tri";
                break;
            case VehicleConfigurationSource::MULTI_ROTOR_QUAD_X:
                elementsToShow << "quad-x";
                break;
            case VehicleConfigurationSource::MULTI_ROTOR_QUAD_PLUS:
                elementsToShow << "quad-p";
                break;
            case VehicleConfigurationSource::MULTI_ROTOR_HEXA:
                elementsToShow << "hexa";
                break;
            case VehicleConfigurationSource::MULTI_ROTOR_HEXA_COAX_Y:
                elementsToShow << "hexa-y";
                break;
            case VehicleConfigurationSource::MULTI_ROTOR_HEXA_H:
                elementsToShow << "hexa-h";
                break;
            default:
                break;
            }
            break;
        case VehicleConfigurationSource::VEHICLE_FIXEDWING:
        case VehicleConfigurationSource::VEHICLE_HELI:
        case VehicleConfigurationSource::VEHICLE_SURFACE:
        default:
            break;
        }

        switch (m_configSource->getInputType()) {
        case Core::IBoardType::INPUT_TYPE_PWM:
            elementsToShow << "pwm";
            break;
        case Core::IBoardType::INPUT_TYPE_PPM:
            elementsToShow << "ppm";
            break;
        case Core::IBoardType::INPUT_TYPE_SBUS:
            elementsToShow << "sbus";
            break;
        case Core::IBoardType::INPUT_TYPE_DSMX10BIT:
        case Core::IBoardType::INPUT_TYPE_DSMX11BIT:
        case Core::IBoardType::INPUT_TYPE_DSM2:
            elementsToShow << "satellite";
            break;
        case Core::IBoardType::INPUT_TYPE_HOTTSUMD:
        case Core::IBoardType::INPUT_TYPE_HOTTSUMH:
            elementsToShow << "HoTT";
            break;
        default:
            break;
        }

        setupGraphicsSceneItems(elementsToShow);

        ui->connectionDiagram->setSceneRect(m_background->boundingRect());
        ui->connectionDiagram->fitInView(m_background, Qt::KeepAspectRatio);

        qDebug() << "Scene complete";
    }
}

void ConnectionDiagram::setupGraphicsSceneItems(QList<QString> elementsToShow)
{
    qreal z = 0;

    foreach(QString elementId, elementsToShow) {
        if (m_renderer->elementExists(elementId)) {
            QGraphicsSvgItem *element = new QGraphicsSvgItem();
            element->setSharedRenderer(m_renderer);
            element->setElementId(elementId);
            element->setZValue(z++);
            element->setOpacity(1.0);

            QMatrix matrix = m_renderer->matrixForElement(elementId);
            QRectF orig    = matrix.mapRect(m_renderer->boundsOnElement(elementId));
            element->setPos(orig.x(), orig.y());

            m_scene->addItem(element);
            qDebug() << "Adding " << elementId << " to scene at " << element->pos();
        } else {
            qDebug() << "Element with id: " << elementId << " not found.";
        }
    }
}

void ConnectionDiagram::on_saveButton_clicked()
{
    QImage image(2200, 1100, QImage::Format_ARGB32);

    image.fill(0);
    QPainter painter(&image);
    m_scene->render(&painter);
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"), "", tr("Images (*.png *.xpm *.jpg)"));
    if (!fileName.isEmpty()) {
        image.save(fileName);
    }
}
