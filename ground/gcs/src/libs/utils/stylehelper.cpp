/**
 ******************************************************************************
 *
 * @file       stylehelper.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
 * @brief      
 * @see        The GNU Public License (GPL) Version 3
 * @defgroup   
 * @{
 * 
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

#include "stylehelper.h"

#include <QPixmapCache>
#include <QWidget>
#include <QRect>
#include <QPainter>
#include <QApplication>
#include <QPalette>
#include <QStyleOption>
#include <QObject>


// Clamps float color values within (0, 255)
static int clamp(float x)
{
    const int val = x > 255 ? 255 : static_cast<int>(x);
    return val < 0 ? 0 : val;
}

// Clamps float color values within (0, 255)
/*
static int range(float x, int min, int max)
{
    int val = x > max ? max : x;
    return val < min ? min : val;
}
*/

namespace Utils {

QColor StyleHelper::mergedColors(const QColor &colorA, const QColor &colorB, int factor)
{
    const int maxFactor = 100;
    QColor tmp = colorA;
    tmp.setRed((tmp.red() * factor) / maxFactor + (colorB.red() * (maxFactor - factor)) / maxFactor);
    tmp.setGreen((tmp.green() * factor) / maxFactor + (colorB.green() * (maxFactor - factor)) / maxFactor);
    tmp.setBlue((tmp.blue() * factor) / maxFactor + (colorB.blue() * (maxFactor - factor)) / maxFactor);
    return tmp;
}

qreal StyleHelper::sidebarFontSize()
{
#if defined(Q_WS_MAC)
    return 9;
#else
    return 7.5;
#endif
}

QPalette StyleHelper::sidebarFontPalette(const QPalette &original)
{
    QPalette palette = original;
    palette.setColor(QPalette::Active, QPalette::Text, panelTextColor());
    palette.setColor(QPalette::Active, QPalette::WindowText, panelTextColor());
    palette.setColor(QPalette::Inactive, QPalette::Text, panelTextColor().darker());
    palette.setColor(QPalette::Inactive, QPalette::WindowText, panelTextColor().darker());
    return palette;
}

QColor StyleHelper::panelTextColor(bool lightColored)
{
    if (!lightColored)
        return Qt::white;
    else
        return Qt::black;
}

QColor StyleHelper::m_baseColor(0x666666);

QColor StyleHelper::baseColor()
{
    return m_baseColor;
}

QColor StyleHelper::highlightColor()
{
    QColor result = baseColor();
    result.setHsv(result.hue(),
                  clamp(result.saturation()),
                  clamp(result.value() * 1.16));
    return result;
}

QColor StyleHelper::shadowColor()
{
    QColor result = baseColor();
    result.setHsv(result.hue(),
                  clamp(result.saturation() * 1.1),
                  clamp(result.value() * 0.70));
    return result;
}

QColor StyleHelper::borderColor()
{
    QColor result = baseColor();
    result.setHsv(result.hue(),
                  result.saturation(),
                  result.value() / 2);
    return result;
}

void StyleHelper::setBaseColor(const QColor &color)
{
    if (color.isValid() && color != m_baseColor) {
        m_baseColor = color;
        foreach (QWidget *w, QApplication::topLevelWidgets())
            w->update();
    }
}

static void verticalGradientHelper(QPainter *p, const QRect &spanRect, const QRect &rect)
{
    QColor base = StyleHelper::baseColor();
    QLinearGradient grad(spanRect.topRight(), spanRect.topLeft());
    grad.setColorAt(0, StyleHelper::highlightColor());
    grad.setColorAt(0.301, base);
    grad.setColorAt(1, StyleHelper::shadowColor());
    p->fillRect(rect, grad);

    QColor light(255, 255, 255, 80);
    p->setPen(light);
    p->drawLine(rect.topRight() - QPoint(1, 0), rect.bottomRight() - QPoint(1, 0));
}

void StyleHelper::verticalGradient(QPainter *painter, const QRect &spanRect,
                                   const QRect &clipRect, bool lightColored)
{
    if (StyleHelper::usePixmapCache()) {
        QString key;
        key.sprintf("mh_vertical %d %d %d %d %d",
            spanRect.width(), spanRect.height(), clipRect.width(),
            clipRect.height(), StyleHelper::baseColor().rgb());;

        QPixmap pixmap;
        if (!QPixmapCache::find(key, pixmap)) {
            pixmap = QPixmap(clipRect.size());
            QPainter p(&pixmap);
            QRect rect(0, 0, clipRect.width(), clipRect.height());
            verticalGradientHelper(&p, spanRect, rect);
            p.end();
            QPixmapCache::insert(key, pixmap);
        }

        painter->drawPixmap(clipRect.topLeft(), pixmap);
    } else {
        verticalGradientHelper(painter, spanRect, clipRect);
    }
}

static void horizontalGradientHelper(QPainter *p, const QRect &spanRect,
                                     const QRect &rect)
{
    QColor base = StyleHelper::baseColor();
    QLinearGradient grad(rect.topLeft(), rect.bottomLeft());
    grad.setColorAt(0, StyleHelper::highlightColor().lighter(120));
    if (rect.height() == StyleHelper::navigationWidgetHeight()) {
        grad.setColorAt(0.4, StyleHelper::highlightColor());
        grad.setColorAt(0.401, base);
    }
    grad.setColorAt(1, StyleHelper::shadowColor());
    p->fillRect(rect, grad);

    QLinearGradient shadowGradient(spanRect.topLeft(), spanRect.topRight());
    shadowGradient.setColorAt(0, QColor(0, 0, 0, 30));
    QColor highlight = StyleHelper::highlightColor().lighter(130);
    highlight.setAlpha(100);
    shadowGradient.setColorAt(0.7, highlight);
    shadowGradient.setColorAt(1, QColor(0, 0, 0, 40));
    p->fillRect(rect, shadowGradient);

}

void StyleHelper::horizontalGradient(QPainter *painter, const QRect &spanRect,
                                     const QRect &clipRect, bool lightColored)
{
    if (StyleHelper::usePixmapCache()) {
        QString key;
        key.sprintf("mh_horizontal %d %d %d %d %d",
            spanRect.width(), spanRect.height(), clipRect.width(),
            clipRect.height(), StyleHelper::baseColor().rgb());

        QPixmap pixmap;
        if (!QPixmapCache::find(key, pixmap)) {
            pixmap = QPixmap(clipRect.size());
            QPainter p(&pixmap);
            QRect rect = QRect(0, 0, clipRect.width(), clipRect.height());
            horizontalGradientHelper(&p, spanRect, rect);
            p.end();
            QPixmapCache::insert(key, pixmap);
        }

        painter->drawPixmap(clipRect.topLeft(), pixmap);

    } else {
        horizontalGradientHelper(painter, spanRect, clipRect);
    }
}

static void menuGradientHelper(QPainter *p, const QRect &spanRect, const QRect &rect)
{
    QLinearGradient grad(spanRect.topLeft(), spanRect.bottomLeft());
    QColor menuColor = StyleHelper::mergedColors(StyleHelper::baseColor(), QColor(244, 244, 244), 25);
    grad.setColorAt(0, menuColor.lighter(112));
    grad.setColorAt(1, menuColor);
    p->fillRect(rect, grad);
}

void StyleHelper::drawArrow(QStyle::PrimitiveElement element, QPainter *painter, const QStyleOption *option)
{
    // From windowsstyle but modified to enable AA
    if (option->rect.width() <= 1 || option->rect.height() <= 1)
        return;

    QRect r = option->rect;
    int size = qMin(r.height(), r.width());
    QPixmap pixmap;
    QString pixmapName;
    pixmapName.sprintf("arrow-%s-%d-%d-%d-%lld",
                       "$qt_ia",
                       uint(option->state), element,
                       size, option->palette.cacheKey());
    if (!QPixmapCache::find(pixmapName, pixmap)) {
        int border = size/5;
        int sqsize = 2*(size/2);
        QImage image(sqsize, sqsize, QImage::Format_ARGB32);
        image.fill(Qt::transparent);
        QPainter imagePainter(&image);
        imagePainter.setRenderHint(QPainter::Antialiasing, true);
        imagePainter.translate(0.5, 0.5);
        QPolygon a;
        switch (element) {
            case QStyle::PE_IndicatorArrowUp:
                a.setPoints(3, border, sqsize/2,  sqsize/2, border,  sqsize - border, sqsize/2);
                break;
            case QStyle::PE_IndicatorArrowDown:
                a.setPoints(3, border, sqsize/2,  sqsize/2, sqsize - border,  sqsize - border, sqsize/2);
                break;
            case QStyle::PE_IndicatorArrowRight:
                a.setPoints(3, sqsize - border, sqsize/2,  sqsize/2, border,  sqsize/2, sqsize - border);
                break;
            case QStyle::PE_IndicatorArrowLeft:
                a.setPoints(3, border, sqsize/2,  sqsize/2, border,  sqsize/2, sqsize - border);
                break;
            default:
                break;
        }

        int bsx = 0;
        int bsy = 0;

        if (option->state & QStyle::State_Sunken) {
            bsx = qApp->style()->pixelMetric(QStyle::PM_ButtonShiftHorizontal);
            bsy = qApp->style()->pixelMetric(QStyle::PM_ButtonShiftVertical);
        }

        QRect bounds = a.boundingRect();
        int sx = sqsize / 2 - bounds.center().x() - 1;
        int sy = sqsize / 2 - bounds.center().y() - 1;
        imagePainter.translate(sx + bsx, sy + bsy);

        if (!(option->state & QStyle::State_Enabled)) {
            QColor foreGround(150, 150, 150, 150);
            imagePainter.setBrush(option->palette.mid().color());
            imagePainter.setPen(option->palette.mid().color());
        } else {
            QColor shadow(0, 0, 0, 100);
            imagePainter.translate(0, 1);
            imagePainter.setPen(shadow);
            imagePainter.setBrush(shadow);
            QColor foreGround(255, 255, 255, 210);
            imagePainter.drawPolygon(a);
            imagePainter.translate(0, -1);
            imagePainter.setPen(foreGround);
            imagePainter.setBrush(foreGround);
        }
        imagePainter.drawPolygon(a);
        imagePainter.end();
        pixmap = QPixmap::fromImage(image);
        QPixmapCache::insert(pixmapName, pixmap);
    }
    int xOffset = r.x() + (r.width() - size)/2;
    int yOffset = r.y() + (r.height() - size)/2;
    painter->drawPixmap(xOffset, yOffset, pixmap);
}

void StyleHelper::menuGradient(QPainter *painter, const QRect &spanRect, const QRect &clipRect)
{
    if (StyleHelper::usePixmapCache()) {
        QString key;
        key.sprintf("mh_menu %d %d %d %d %d",
            spanRect.width(), spanRect.height(), clipRect.width(),
            clipRect.height(), StyleHelper::baseColor().rgb());

        QPixmap pixmap;
        if (!QPixmapCache::find(key, pixmap)) {
            pixmap = QPixmap(clipRect.size());
            QPainter p(&pixmap);
            QRect rect = QRect(0, 0, clipRect.width(), clipRect.height());
            menuGradientHelper(&p, spanRect, rect);
            p.end();
            QPixmapCache::insert(key, pixmap);
        }

        painter->drawPixmap(clipRect.topLeft(), pixmap);
    } else {
        menuGradientHelper(painter, spanRect, clipRect);
    }
}

//! Draws a CSS-like border image where the defined borders are not stretched
void StyleHelper::drawCornerImage(const QImage &img, QPainter *painter, QRect rect,
                                  int left, int top, int right, int bottom)
{
    QSize size = img.size();
    if (top > 0) { //top
        painter->drawImage(QRect(rect.left() + left, rect.top(), rect.width() -right - left, top), img,
                           QRect(left, 0, size.width() -right - left, top));
        if (left > 0) //top-left
            painter->drawImage(QRect(rect.left(), rect.top(), left, top), img,
                               QRect(0, 0, left, top));
        if (right > 0) //top-right
            painter->drawImage(QRect(rect.left() + rect.width() - right, rect.top(), right, top), img,
                               QRect(size.width() - right, 0, right, top));
    }
    //left
    if (left > 0)
        painter->drawImage(QRect(rect.left(), rect.top()+top, left, rect.height() - top - bottom), img,
                           QRect(0, top, left, size.height() - bottom - top));
    //center
    painter->drawImage(QRect(rect.left() + left, rect.top()+top, rect.width() -right - left,
                             rect.height() - bottom - top), img,
                       QRect(left, top, size.width() -right -left,
                             size.height() - bottom - top));
    if (right > 0) //right
        painter->drawImage(QRect(rect.left() +rect.width() - right, rect.top()+top, right, rect.height() - top - bottom), img,
                           QRect(size.width() - right, top, right, size.height() - bottom - top));
    if (bottom > 0) { //bottom
        painter->drawImage(QRect(rect.left() +left, rect.top() + rect.height() - bottom,
                                 rect.width() - right - left, bottom), img,
                           QRect(left, size.height() - bottom,
                                 size.width() - right - left, bottom));
    if (left > 0) //bottom-left
        painter->drawImage(QRect(rect.left(), rect.top() + rect.height() - bottom, left, bottom), img,
                           QRect(0, size.height() - bottom, left, bottom));
    if (right > 0) //bottom-right
        painter->drawImage(QRect(rect.left() + rect.width() - right, rect.top() + rect.height() - bottom, right, bottom), img,
                           QRect(size.width() - right, size.height() - bottom, right, bottom));
    }
}

} // namespace Utils
