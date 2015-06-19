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

#ifndef PFDQMLGADGETWIDGET_H_
#define PFDQMLGADGETWIDGET_H_

#include "pfdqmlgadgetconfiguration.h"
#include <QtQuick/QQuickView>

class UAVObjectManager;

class PfdQmlGadgetWidget : public QQuickView
{
    Q_OBJECT

public:
    PfdQmlGadgetWidget(QWindow *parent = 0);
   ~PfdQmlGadgetWidget();
    void setQmlFile(QString fn);

public slots:
    void setSettingsMap(const QVariantMap &settings);

protected:
    void mouseReleaseEvent(QMouseEvent *event);

private:
    QStringList objectsToExport;
    QString m_qmlFileName;

    UAVObjectManager *m_objManager;
    void exportUAVOInstance(const QString &objectName, int instId);
    void resetUAVOExport(const QString &objectName, int instId);
};

#endif /* PFDQMLGADGETWIDGET_H_ */
