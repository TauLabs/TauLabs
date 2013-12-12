/**
******************************************************************************
*
* @file       textbubbleslider.h
* @author     Tau Labs, http://taulabs.org Copyright (C) 2013.
* @brief      Creates a slider with a text bubble showing the slider value
* @see        The GNU Public License (GPL) Version 3
* @defgroup   Config
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

#include <QDebug>
#include <QPainter>
#include <QDebug>
#include <qmath.h>

#include "textbubbleslider.h"

/**
 * @brief TextBubbleSlider::TextBubbleSlider Constructs a regular text-bubble slider
 * @param parent
 */
TextBubbleSlider::TextBubbleSlider(QWidget *parent) :
    QSlider(parent)
{
    construct();
}

/**
 * @brief TextBubbleSlider::TextBubbleSlider Constructs a text-bubble slider that copys
 * all relevant data from the previous slider
 * @param copySlider
 * @param parent
 */
TextBubbleSlider::TextBubbleSlider(QSlider *copySlider, QWidget *parent) :
    QSlider(parent)
{
    construct();

    // Copy settings
    setSizePolicy(copySlider->sizePolicy());
    setMinimumSize(copySlider->minimumSize());
    setMaximumSize(copySlider->maximumSize());
    setFocusPolicy(copySlider->focusPolicy());
    setOrientation(copySlider->orientation());
    setMaximum(copySlider->maximum());
    setMinimum(copySlider->minimum());
    setToolTip(copySlider->toolTip());
}


/**
 * @brief TextBubbleSlider::construct This function needs to be called from all constructors. It
 * provides a single point where settings can be changed.
 */
void TextBubbleSlider::construct()
{
    font = QFont("Arial", 13);
    slideHandleMargin = 2; // This is a dubious way to set the margin. In reality, it should be read from the style sheet.
}

TextBubbleSlider::~TextBubbleSlider()
{
}


/**
 * @brief numIntegerDigits Counts the number of digits in an integer
 * @param number Input integer
 * @return Number of digits in input integer
 */
unsigned int numIntegerDigits(int number)
{
    unsigned int digits = 0;

    // If there is a negative sign, be sure to include it in digit count
    if (number < 0)
        digits = 1;

    while (number) {
        number /= 10;
        digits++;
    }

    return digits;
}


/**
 * @brief TextBubbleSlider::setMaxPixelWidth Sets maximum pixel width for slider handle
 */
void TextBubbleSlider::setMaxPixelWidth()
{
    // Calculate maximum number of digits possible in string
    int maxNumDigits = numIntegerDigits(maximum()) > numIntegerDigits(minimum()) ? numIntegerDigits(maximum()) : numIntegerDigits(minimum());

    // Generate string with maximum pixel width. Suppose that "0" is
    // the widest number in pixels.
    QString maximumWidthString;
    for (int i=0; i<maxNumDigits; i++) {
        maximumWidthString.append("0");
    }

    // Calculate maximum possible pixel width for string.
    QFontMetrics fontMetrics(font);
    maximumFontWidth = fontMetrics.width(QString("%1").arg(maximumWidthString));
    maximumFontHeight = fontMetrics.height();

    // Override stylesheet slider handle width
    slideHandleWidth = maximumFontWidth + 6;
    setStyleSheet(QString("QSlider::handle:horizontal { width: %1px; margin: -5px 0;}").arg(slideHandleWidth));
}


/**
 * @brief TextBubbleSlider::setMinimum Reimplements setMinimum. Ensures that the slider
 * handle is the correct size for the text field.
 * @param max maximum
 */
void TextBubbleSlider::setMinimum(int max)
{
    // Pass value on to QSlider
    QSlider::setMinimum(max);

    // Reset handler size
    setMaxPixelWidth();
}


/**
 * @brief TextBubbleSlider::setMaximum Reimplements setMaximum. Ensures that the slider
 * handle is the correct size for the text field.
 * @param max maximum
 */
void TextBubbleSlider::setMaximum(int max)
{
    // Pass value on to QSlider
    QSlider::setMaximum(max);

    // Reset handler size
    setMaxPixelWidth();
}


/**
 * @brief TextBubbleSlider::paintEvent Reimplements QSlider::paintEvent.
 * @param bob
 */
void TextBubbleSlider::paintEvent(QPaintEvent *paintEvent)
{
    // Pass paint event on to QSlider
    QSlider::paintEvent(paintEvent);

    /* Add numbers on top of handler */

    // Calculate pixel position for text.
    int sliderWidth = width();
    int sliderHeight = height();
    double valuePos;

    if (!invertedAppearance()) {
        valuePos = (slideHandleWidth - maximumFontWidth)/2 + slideHandleMargin + // First part centers text in handle...
                (value()-minimum())/(double)(maximum()-minimum()) * (sliderWidth - (slideHandleWidth + slideHandleMargin) - 1); //... and second part moves text with handle
    } else {
        valuePos = (slideHandleWidth - maximumFontWidth)/2 + slideHandleMargin + // First part centers text in handle...
                (maximum()-value())/(double)(maximum()-minimum()) * (sliderWidth - (slideHandleWidth + slideHandleMargin) - 1); //... and second part moves text with handle
    }

    // Create painter and set font
    QPainter painter(this);
    painter.setFont(font);

    // Draw neutral value text. Verically center it in the handle
    QString neutralStringWidth = QString("%1").arg(value());
    QFontMetrics fontMetrics(font);
    int textWidth = fontMetrics.width(neutralStringWidth);
    painter.drawText(QRectF(valuePos + maximumFontWidth - textWidth, ceil((sliderHeight - maximumFontHeight)/2.0), textWidth, maximumFontHeight),
                     neutralStringWidth);
}
