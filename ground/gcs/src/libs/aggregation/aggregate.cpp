/**
 ******************************************************************************
 *
 * @file       aggregate.cpp
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

#include "aggregate.h"

#include <QtCore/QWriteLocker>

/*!
    \namespace Aggregation
    \brief The Aggregation namespace contains support for bundling related components,
           such that each component exposes the properties and behavior of the
           other components to the outside.

    Components that are bundled to an Aggregate can be "cast" to each other
    and have a coupled life cycle. See the documentation of Aggregation::Aggregate for
    details and examples.
*/

/*!
    \class Aggregation::Aggregate
    \mainclass
    \threadsafe

    \brief Defines a collection of related components that can be viewed as a unit.

    An Aggregate is a collection of components that are handled as a unit,
    such that each component exposes the properties and behavior of the
    other components in the Aggregate to the outside.
    Specifically that means:
    \list
    \o They can be "cast" to each other (using query and query_all methods).
    \o Their life cycle is coupled, i.e. whenever one is deleted all of them are.
    \endlist
    Components can be of any QObject derived type.

    You can use an Aggregate to simulate multiple inheritance by aggregation. Assume we have
    \code
        using namespace Aggregation;
        class MyInterface : public QObject { ........ };
        class MyInterfaceEx : public QObject { ........ };
        [...]
        MyInterface *object = new MyInterface; // this is single inheritance
    \endcode
    The query method works like a qobject_cast with normal objects:
    \code
        Q_ASSERT(query<MyInterface>(object) == object);
        Q_ASSERT(query<MyInterfaceEx>(object) == 0);
    \endcode
    If we want 'object' to also implement the class MyInterfaceEx,
    but don't want to or cannot use multiple inheritance, we can do it
    at any point using an Aggregate:
    \code
        MyInterfaceEx *objectEx = new MyInterfaceEx;
        Aggregate *aggregate = new Aggregate;
        aggregate->add(object);
        aggregate->add(objectEx);
    \endcode
    The Aggregate bundles the two objects together.
    If we have any part of the collection we get all parts:
    \code
        Q_ASSERT(query<MyInterface>(object) == object);
        Q_ASSERT(query<MyInterfaceEx>(object) == objectEx);
        Q_ASSERT(query<MyInterface>(objectEx) == object);
        Q_ASSERT(query<MyInterfaceEx>(objectEx) == objectEx);
    \endcode
    The following deletes all three: object, objectEx and aggregate:
    \code
        delete objectEx;
        // or delete object;
        // or delete aggregate;
    \endcode

    Aggregation aware code never uses qobject_cast, but always uses
    Aggregation::query which behaves like a qobject_cast as a fallback.
*/

/*!
    \fn T *Aggregate::component()

    Template method that returns the component with the given type, if there is one.
    If there are multiple components with that type a random one is returned.

    \sa Aggregate::components()
    \sa Aggregate::add()
*/

/*!
    \fn QList<T *> Aggregate::components()

    Template method that returns all components with the given type, if there are any.

    \sa Aggregate::component()
    \sa Aggregate::add()
*/

/*!
    \fn T *Aggregation::query<T *>(Aggregate *obj)
    \internal
*/

/*!
    \fn QList<T *> Aggregation::query_all<T *>(Aggregate *obj)
    \internal
*/

/*!
    \relates Aggregation::Aggregate
    \fn T *Aggregation::query<T *>(QObject *obj)

    Performs a dynamic cast that is aware of a possible Aggregate that \a obj
    might belong to. If \a obj itself is of the requested type then it is simply cast
    and returned. Otherwise, if \a obj belongs to an Aggregate all its components are
    checked, or if it doesn't belong to an Aggregate null is returned.

    \sa Aggregate::component()
*/

/*!
    \relates Aggregation::Aggregate
    \fn QList<T *> Aggregation::query_all<T *>(QObject *obj)

    If \a obj belongs to an Aggregate, all components that can be cast to the given
    type are returned. Otherwise, \a obj is returned if it is of the requested type.

    \sa Aggregate::components()
*/

using namespace Aggregation;

/*!
    \fn Aggregate *Aggregate::parentAggregate(QObject *obj)

    Returns the Aggregate object of \a obj if there is one. Otherwise returns 0.
*/
Aggregate *Aggregate::parentAggregate(QObject *obj)
{
    QReadLocker locker(&lock());
    return aggregateMap().value(obj);
}

QHash<QObject *, Aggregate *> &Aggregate::aggregateMap()
{
    static QHash<QObject *, Aggregate *> map;
    return map;
}

/*!
    \fn QReadWriteLock &Aggregate::lock()
    \internal
*/
QReadWriteLock &Aggregate::lock()
{
    static QReadWriteLock lock;
    return lock;
}

/*!
    \fn Aggregate::Aggregate(QObject *parent)

    Creates a new Aggregate with the given \a parent.
    The \a parent is passed directly passed to the QObject part
    of the class and is not used beside that.
*/
Aggregate::Aggregate(QObject *parent)
    : QObject(parent)
{
    QWriteLocker locker(&lock());
    aggregateMap().insert(this, this);
}

/*!
    \fn Aggregate::~Aggregate()

    Deleting the aggregate automatically deletes all its components.
*/
Aggregate::~Aggregate()
{
    QWriteLocker locker(&lock());
    foreach (QObject *component, m_components) {
        disconnect(component, SIGNAL(destroyed(QObject*)), this, SLOT(deleteSelf(QObject*)));
        aggregateMap().remove(component);
    }
    qDeleteAll(m_components);
    m_components.clear();
    aggregateMap().remove(this);
}

void Aggregate::deleteSelf(QObject *obj)
{
    {
        QWriteLocker locker(&lock());
        aggregateMap().remove(obj);
        m_components.removeAll(obj);
    }
    delete this;
}

/*!
    \fn void Aggregate::add(QObject *component)

    Adds the \a component to the aggregate.

    \sa Aggregate::remove()
*/
void Aggregate::add(QObject *component)
{
    if (!component)
        return;
    QWriteLocker locker(&lock());
    Aggregate *parentAggregation = aggregateMap().value(component);
    if (parentAggregation == this)
        return;
    if (parentAggregation)
        parentAggregation->remove(component);
    m_components.append(component);
    connect(component, SIGNAL(destroyed(QObject*)), this, SLOT(deleteSelf(QObject*)));
    aggregateMap().insert(component, this);
}

/*!
    \fn void Aggregate::remove(QObject *component)

    Removes the \a component from the aggregate.

    \sa Aggregate::add()
*/
void Aggregate::remove(QObject *component)
{
    if (!component)
        return;
    QWriteLocker locker(&lock());
    aggregateMap().remove(component);
    m_components.removeAll(component);
    disconnect(component, SIGNAL(destroyed(QObject*)), this, SLOT(deleteSelf(QObject*)));
}
