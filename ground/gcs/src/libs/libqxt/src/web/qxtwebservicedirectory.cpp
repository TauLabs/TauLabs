/****************************************************************************
 **
 ** Copyright (C) Qxt Foundation. Some rights reserved.
 **
 ** This file is part of the QxtWeb module of the Qxt library.
 **
 ** This library is free software; you can redistribute it and/or modify it
 ** under the terms of the Common Public License, version 1.0, as published
 ** by IBM, and/or under the terms of the GNU Lesser General Public License,
 ** version 2.1, as published by the Free Software Foundation.
 **
 ** This file is provided "AS IS", without WARRANTIES OR CONDITIONS OF ANY
 ** KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT LIMITATION, ANY
 ** WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY OR
 ** FITNESS FOR A PARTICULAR PURPOSE.
 **
 ** You should have received a copy of the CPL and the LGPL along with this
 ** file. See the LICENSE file and the cpl1.0.txt/lgpl-2.1.txt files
 ** included with the source distribution for more information.
 ** If you did not receive a copy of the licenses, contact the Qxt Foundation.
 **
 ** <http://libqxt.org>  <foundation@libqxt.org>
 **
 ****************************************************************************/

/*!
\class QxtWebServiceDirectory

\inmodule QxtWeb

\brief The QxtWebServiceDirectory class provides Path-based web service dispatcher

QxtWebServiceDirectory allows multiple services to be associated with a single
session. Selection between services is determined by the first path component
in the URL. For example, the URL "/site/request?param=true" would relay the
URL "/request?param=true" to the service named "site".

This class can be used recursively to declare a hierarchy of services. For
example:
\code
QxtWebServiceDirectory* top = new QxtWebServiceDirectory(sm, sm);
QxtWebServiceDirectory* service1 = new QxtWebServiceDirectory(sm, top);
QxtWebServiceDirectory* service2 = new QxtWebServiceDirectory(sm, top);
QxtWebServiceDirectory* service1a = new QxtWebServiceDirectory(sm, service1);
QxtWebServiceDirectory* service1b = new QxtWebServiceDirectory(sm, service1);
top->addService("1", service1);
top->addService("2", service2);
service1->addService("a", service1a);
service1->addService("b", service1b);
\endcode
This accepts the URLs "/1/a/", "/1/b/", and "/2/".
*/

#include "qxtwebservicedirectory.h"
#include "qxtwebservicedirectory_p.h"
#include "qxtwebevent.h"
#include <QUrl>
#include <QtDebug>

#ifndef QXT_DOXYGEN_RUN
QxtWebServiceDirectoryPrivate::QxtWebServiceDirectoryPrivate() : QObject(0)
{
    // initializers only
}

void QxtWebServiceDirectoryPrivate::serviceDestroyed()
{
    QxtAbstractWebService* service = qobject_cast<QxtAbstractWebService*>(sender());
    if (!service) return; // this shouldn't happen
    QString path;
    while (!(path = services.key(service)).isNull())
    {
        services.remove(path);
    }
}
#endif

/*!
 * Constructs a QxtWebServiceDirectory object with the specified session manager \a sm and \a parent.
 *
 * Often, the session manager will also be the parent, but this is not a requirement.
 */
QxtWebServiceDirectory::QxtWebServiceDirectory(QxtAbstractWebSessionManager* sm, QObject* parent) : QxtAbstractWebService(sm, parent)
{
    QXT_INIT_PRIVATE(QxtWebServiceDirectory);
}

/*!
 * Adds a \a service to the directory at the given \a path.
 * \sa removeService(), service()
 */
void QxtWebServiceDirectory::addService(const QString& path, QxtAbstractWebService* service)
{
    if (qxt_d().services.contains(path))
    {
        qWarning() << "QxtWebServiceDirectory::addService:" << path << "already registered";
    }

    qxt_d().services[path] = service;
    if (qxt_d().defaultRedirect.isEmpty())
        setDefaultRedirect(path);
    connect(service, SIGNAL(destroyed()), &qxt_d(), SLOT(serviceDestroyed()));
}

/*!
 * Removes the service at the given \a path.
 *
 * Note that the service object is not destroyed.
 */
void QxtWebServiceDirectory::removeService(const QString& path)
{
    if (!qxt_d().services.contains(path))
    {
        qWarning() << "QxtWebServiceDirectory::removeService:" << path << "not registered";
    }
    else
    {
        qxt_d().services.remove(path);
    }
}

/*!
 * Returns the service at the given \a path.
 */
QxtAbstractWebService* QxtWebServiceDirectory::service(const QString& path) const
{
    if (!qxt_d().services.contains(path))
        return 0;
    return qxt_d().services[path];
}

/*!
 * \internal
 * Returns the first path segment from the URL in the \a event object.
 * (i.e. "a" from "/a/b/c") This also removes the path segment from the
 * event object. (in the previous example, the event's URL is now "/b/c")
 */
static QString extractPathLevel(QxtWebRequestEvent* event)
{
    QString path = event->url.path();
    int pos = path.indexOf("/", 1); // the path always starts with /
    if (pos == -1)
        event->url.setPath(""); // cue to redirect to /service/
    else
        event->url.setPath(path.mid(pos));
    return path.mid(1, pos - 1);
}

/*!
 * \reimp
 */
void QxtWebServiceDirectory::pageRequestedEvent(QxtWebRequestEvent* event)
{
    QString path = extractPathLevel(event);
    if (path.isEmpty())
    {
        indexRequested(event);
    }
    else if (!qxt_d().services.contains(path))
    {
        unknownServiceRequested(event, path);
    }
    else if (event->url.path().isEmpty())
    {
        postEvent(new QxtWebRedirectEvent(event->sessionID, event->requestID, path + '/', 307));
    }
    else
    {
        qxt_d().services[path]->pageRequestedEvent(event);
    }
}

/*
 * \reimp unimplemented
 */
/*
void QxtWebServiceDirectory::functionInvokedEvent(QxtWebRequestEvent* event) {
    QString path = extractPathLevel(event);
    if(path == "") {
        indexRequested(event);
    } else if(!qxt_d().services.contains(path)) {
        unknownServiceRequested(event, path);
    } else {
        qxt_d().services[path]->functionInvokedEvent(event);
    }
}
*/

/*!
 * This \a event handler is called whenever the URL requests a service with \a name that has
 * not been added to the directory.
 *
 * The default implementation returns a 404 "Service not known" error.
 * Subclasses may reimplement this event handler to customize this behavior.
 */
void QxtWebServiceDirectory::unknownServiceRequested(QxtWebRequestEvent* event, const QString& name)
{
    postEvent(new QxtWebErrorEvent(event->sessionID, event->requestID, 404, ("Service &quot;" + QString(name).replace('<', "&lt") + "&quot; not known").toUtf8()));
}

/*!
 * This \a event handler is called whenever the URL does not contain a path, that
 * is, the URL is "/" or empty.
 *
 * The default implementation redirects to the service specified by
 * setDefaultRedirect(), or invokes unknownServiceRequested() if no default
 * redirect has been set.
 */
void QxtWebServiceDirectory::indexRequested(QxtWebRequestEvent* event)
{
    if (defaultRedirect().isEmpty())
    {
        unknownServiceRequested(event, "/");
    }
    else
    {
        postEvent(new QxtWebRedirectEvent(event->sessionID, event->requestID, defaultRedirect() + '/', 307));
    }
}

/*!
 * Returns the path that will be used by default by the indexRequested event.
 * \sa indexRequested(), setDefaultRedirect()
 */
QString QxtWebServiceDirectory::defaultRedirect() const
{
    return qxt_d().defaultRedirect;
}

/*!
 * Sets the \a path that will be used by default by the indexRequested event.
 * \sa indexRequested(), defaultRedirect()
 */
void QxtWebServiceDirectory::setDefaultRedirect(const QString& path)
{
    if (!qxt_d().services.contains(path))
        qWarning() << "QxtWebServiceDirectory::setDefaultRedirect:" << path << "not registered";
    qxt_d().defaultRedirect = path;
}
