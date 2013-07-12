/**
 ******************************************************************************
 * @file       customsplash.cpp
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @addtogroup app
 * @{
 * @addtogroup CustomSplash
 * @{
 * @brief A custom splashscreen class with transparent background and progress bar
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

#include "customsplash.h"
#include <QStyleOptionProgressBarV2>
#include <QPainter>
#include <QDebug>

#define PROGRESS_BAR_WIDTH  100
#define PROGRESS_BAR_HEIGHT 12
#define TEXT_BACKGROUND_WIDTH  480
#define TEXT_BACKGROUND_HEIGHT 20

/**
* @brief Class constructor
* @param pixmap picture to show on splashscreen, should be png transparent background.
* @param f window flags.
*/
CustomSplash::CustomSplash(const QPixmap &pixmap, Qt::WindowFlags f):
    QSplashScreen(pixmap,f),m_progress(0),m_progress_bar_color(QColor(13, 125, 102, 255)),message_number(0)
{
    QPixmap original_scaled = pixmap.scaledToHeight(180);
    QPixmap pix(500,180);
    pix.fill(QColor(255,0,0,0));
    QPainter p(&pix);
    p.drawPixmap(pix.rect().center().x()-original_scaled.width()/2,0,original_scaled);
    p.end();
    this->setPixmap(pix);
    this->setCursor(Qt::BusyCursor);
    resize(500,240);
    time.start();
    settings.beginGroup("splashscreen");
}

void CustomSplash::drawContents(QPainter *painter)
{
    painter->setBrush(QBrush(QColor(222, 222, 222, 200)));
    painter->drawRoundedRect(QRect(this->rect().center().x()-PROGRESS_BAR_WIDTH/2,this->rect().bottom()-PROGRESS_BAR_HEIGHT-30,PROGRESS_BAR_WIDTH,PROGRESS_BAR_HEIGHT), 5,5);
    painter->drawRoundedRect(QRect(this->rect().center().x()-TEXT_BACKGROUND_WIDTH/2,this->rect().bottom()-2-TEXT_BACKGROUND_HEIGHT,TEXT_BACKGROUND_WIDTH,TEXT_BACKGROUND_HEIGHT), 5,5);
    QSplashScreen::drawContents(painter);
    painter->setBrush(QBrush(m_progress_bar_color));
    painter->drawRoundedRect(QRect(this->rect().center().x()-PROGRESS_BAR_WIDTH/2,this->rect().bottom()-PROGRESS_BAR_HEIGHT-30,PROGRESS_BAR_WIDTH * m_progress / 100 ,PROGRESS_BAR_HEIGHT), 5,5);

}

/**
* @brief Gets the fill color of the progress bar
* @return The fill color of the progress bar
*/
QColor CustomSplash::progressBarColor() const
{
    return m_progress_bar_color;
}

/**
* @brief Sets the fill color of the progress bar
* @param progressBarColor The fill color of the progress bar
*/
void CustomSplash::setprogressBarColor(const QColor &progressBarColor)
{
    m_progress_bar_color = progressBarColor;
}

/**
* @brief Sets the message to display below the progress bar
* @param message text to display
* @param alignment alignment of the text
* @param color color of the text
*/
void CustomSplash::showMessage(const QString &message, int alignment, const QColor &color)
{
    QSplashScreen::showMessage(message,alignment,color);
    int last_duration = settings.value(QString::number(message_number),0).toInt();
    if(last_duration == 0)
        setProgress(progress()+10);
    else
    {
        int total_duration = settings.value("total",5000).toInt();
        int value = last_duration * 100 / total_duration;
        setProgress(value);
    }
    settings.setValue(QString::number(message_number),time.elapsed());
    ++message_number;
}
/**
 * @brief Closes the splashscreen
 */
void CustomSplash::close()
{
    setProgress(100);
    settings.setValue(QString::number(message_number),time.elapsed());
    settings.setValue("total",time.elapsed());
    settings.endGroup();
    QSplashScreen::close();
}
