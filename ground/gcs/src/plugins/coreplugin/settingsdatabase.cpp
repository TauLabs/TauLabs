/**
 ******************************************************************************
 *
 * @file       settingsdatabase.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 *             Parts by Nokia Corporation (qt-info@nokia.com) Copyright (C) 2009.
 * @addtogroup GCSPlugins GCS Plugins
 * @{
 * @addtogroup CorePlugin Core Plugin
 * @{
 * @brief The Core GCS plugin
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

#include "settingsdatabase.h"

#include <QtCore/QDir>
#include <QtCore/QMap>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QVariant>

#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlError>
#include <QtSql/QSqlQuery>
#include <QDebug>

/*!
    \class Core::SettingsDatabase
    \brief An alternative to the application-wide QSettings that is more
    suitable for storing large amounts of data.

    The settings database is SQLite based, and lazily retrieves data when it
    is asked for. It also does incremental updates of the database rather than
    rewriting the whole file each time one of the settings change.

    The SettingsDatabase API mimics that of QSettings.
*/

using namespace Core;
using namespace Core::Internal;

enum { debug_settings = 0 };

namespace Core {
namespace Internal {

typedef QMap<QString, QVariant> SettingsMap;

class SettingsDatabasePrivate
{
public:
    QString effectiveGroup() const
    {
        return m_groups.join(QLatin1String("/"));
    }

    QString effectiveKey(const QString &key) const
    {
        QString g = effectiveGroup();
        if (!g.isEmpty() && !key.isEmpty())
            g += QLatin1Char('/');
        g += key;
        return g;
    }

    SettingsMap m_settings;

    QStringList m_groups;
    QStringList m_dirtyKeys;

    QSqlDatabase m_db;
};

} // namespace Internal
} // namespace Core

SettingsDatabase::SettingsDatabase(const QString &path,
                                   const QString &application,
                                   QObject *parent)
    : QObject(parent)
    , d(new SettingsDatabasePrivate)
{
    const QLatin1Char slash('/');

    // TODO: Don't rely on a path, but determine automatically
    QDir pathDir(path);
    if (!pathDir.exists())
        pathDir.mkpath(pathDir.absolutePath());

    QString fileName = path;
    if (!fileName.endsWith(slash))
        fileName += slash;
    fileName += application;
    fileName += QLatin1String(".db");

    d->m_db = QSqlDatabase::addDatabase("QSQLITE", QLatin1String("settings"));
    d->m_db.setDatabaseName(fileName);
    if (!d->m_db.open()) {
        qWarning().nospace() << "Warning: Failed to open settings database at " << fileName << " ("
                             << d->m_db.lastError().driverText() << ")";
    } else {
        // Create the settings table if it doesn't exist yet
        QSqlQuery query(d->m_db);
        query.prepare(QLatin1String("CREATE TABLE IF NOT EXISTS settings ("
                                    "key PRIMARY KEY ON CONFLICT REPLACE, "
                                    "value)"));
        if (!query.exec())
            qWarning().nospace() << "Warning: Failed to prepare settings database! ("
                                 << query.lastError().driverText() << ")";

        // Retrieve all available keys (values are retrieved lazily)
        if (query.exec(QLatin1String("SELECT key FROM settings"))) {
            while (query.next()) {
                d->m_settings.insert(query.value(0).toString(), QVariant());
            }
        }
    }
}

SettingsDatabase::~SettingsDatabase()
{
    sync();

    delete d;
    QSqlDatabase::removeDatabase(QLatin1String("settings"));
}

void SettingsDatabase::setValue(const QString &key, const QVariant &value)
{
    const QString effectiveKey = d->effectiveKey(key);

    // Add to cache
    d->m_settings.insert(effectiveKey, value);

    if (!d->m_db.isOpen())
        return;

    // Instant apply (TODO: Delay writing out settings)
    QSqlQuery query(d->m_db);
    query.prepare(QLatin1String("INSERT INTO settings VALUES (?, ?)"));
    query.addBindValue(effectiveKey);
    query.addBindValue(value);
    query.exec();

    if (debug_settings)
        qDebug() << "Stored:" << effectiveKey << "=" << value;
}

QVariant SettingsDatabase::value(const QString &key, const QVariant &defaultValue) const
{
    const QString effectiveKey = d->effectiveKey(key);
    QVariant value = defaultValue;

    SettingsMap::const_iterator i = d->m_settings.constFind(effectiveKey);
    if (i != d->m_settings.constEnd() && i.value().isValid()) {
        value = i.value();
    } else if (d->m_db.isOpen()) {
        // Try to read the value from the database
        QSqlQuery query(d->m_db);
        query.prepare(QLatin1String("SELECT value FROM settings WHERE key = ?"));
        query.addBindValue(effectiveKey);
        query.exec();
        if (query.next()) {
            value = query.value(0);

            if (debug_settings)
                qDebug() << "Retrieved:" << effectiveKey << "=" << value;
        }

        // Cache the result
        d->m_settings.insert(effectiveKey, value);
    }

    return value;
}

bool SettingsDatabase::contains(const QString &key) const
{
    return d->m_settings.contains(d->effectiveKey(key));
}

void SettingsDatabase::remove(const QString &key)
{
    const QString effectiveKey = d->effectiveKey(key);

    // Remove keys from the cache
    foreach (const QString &k, d->m_settings.keys()) {
        // Either it's an exact match, or it matches up to a /
        if (k.startsWith(effectiveKey)
            && (k.length() == effectiveKey.length()
                || k.at(effectiveKey.length()) == QLatin1Char('/')))
        {
            d->m_settings.remove(k);
        }
    }

    if (!d->m_db.isOpen())
        return;

    // Delete keys from the database
    QSqlQuery query(d->m_db);
    query.prepare(QLatin1String("DELETE FROM settings WHERE key = ? OR key LIKE ?"));
    query.addBindValue(effectiveKey);
    query.addBindValue(QString(effectiveKey + QLatin1String("/%")));
    query.exec();
}

void SettingsDatabase::beginGroup(const QString &prefix)
{
    d->m_groups.append(prefix);
}

void SettingsDatabase::endGroup()
{
    d->m_groups.removeLast();
}

QString SettingsDatabase::group() const
{
    return d->effectiveGroup();
}

QStringList SettingsDatabase::childKeys() const
{
    QStringList childs;

    const QString g = group();
    QMapIterator<QString, QVariant> i(d->m_settings);
    while (i.hasNext()) {
        const QString &key = i.next().key();
        if (key.startsWith(g) && key.indexOf(QLatin1Char('/'), g.length() + 1) == -1) {
            childs.append(key.mid(g.length() + 1));
        }
    }

    return childs;
}

void SettingsDatabase::sync()
{
    // TODO: Delay writing of dirty keys and save them here
}
