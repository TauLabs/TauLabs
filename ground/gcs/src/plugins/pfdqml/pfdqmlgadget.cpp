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

#include "pfdqmlgadget.h"
#include "pfdqmlgadgetwidget.h"
#include "pfdqmlgadgetconfiguration.h"

#include "utils/pathutils.h"

#ifdef USE_OSG
#include <osgEarth/Registry>
#include <osgEarth/Cache>
#include <osgEarth/CachePolicy>
#include <osgEarthDrivers/cache_filesystem/FileSystemCache>
#endif

PfdQmlGadget::PfdQmlGadget(QString classId, PfdQmlGadgetWidget *widget, QWidget *parent) :
        IUAVGadget(classId, parent),
        m_widget(widget)
{
    m_container = NULL;
    m_parent = parent;
}

PfdQmlGadget::~PfdQmlGadget()
{
    delete m_widget;
}

/*
  This is called when a configuration is loaded, and updates the plugin's settings.
  Careful: the plugin is already drawn before the loadConfiguration method is called the
  first time, so you have to be careful not to assume all the plugin values are initialized
  the first time you use them
 */
void PfdQmlGadget::loadConfiguration(IUAVGadgetConfiguration* config)
{
    PfdQmlGadgetConfiguration *m = qobject_cast<PfdQmlGadgetConfiguration*>(config);
    m_widget->setOpenGLEnabled(m->openGLEnabled());
    m_widget->setQmlFile(m->qmlFile());
    m_widget->setEarthFile(m->earthFile());
    m_widget->setTerrainEnabled(m->terrainEnabled());
    m_widget->setActualPositionUsed(m->actualPositionUsed());
    m_widget->setLatitude(m->latitude());
    m_widget->setLongitude(m->longitude());
    m_widget->setAltitude(m->altitude());
    m_widget->setSettingsMap(m->settings());

#ifdef USE_OSG
    //setup terrain caching
    QString cacheDir = Utils::PathUtils().GetStoragePath()+QLatin1String("osgEarth_cache");
    osgEarth::Drivers::FileSystemCacheOptions cacheOptions;
    cacheOptions.rootPath() = cacheDir.toStdString();

    osgEarth::Cache *cache = osgEarth::Drivers::CacheFactory::create(cacheOptions);
    if (cache) {
        osgEarth::CachePolicy policy = m->cacheOnly() ?
                    osgEarth::CachePolicy::USAGE_CACHE_ONLY :
                    osgEarth::CachePolicy::USAGE_READ_WRITE;
        osgEarth::Registry::instance()->setDefaultCachePolicy(policy);
        osgEarth::Registry::instance()->setCache(cache);
    }
#endif //USE_OSG
}
