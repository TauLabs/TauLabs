/**
 ******************************************************************************
 *
 * @file       wrap_helpers.h
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

#ifndef WRAP_HELPERS_H
#define WRAP_HELPERS_H

#include <QtScript/QScriptEngine>
#include <QtScript/QScriptContext>
#include <QtScript/QScriptValue>

namespace SharedTools {

// Strip a const ref from a type via template specialization trick.
// Use for determining function call args

template <class T>
    struct RemoveConstRef {
        typedef T Result;
    };

template <class T>
    struct RemoveConstRef<const T &> {
        typedef T Result;
    };

// Template that retrieves a QObject-derived class from a QScriptValue.

template <class QObjectDerived>
    QObjectDerived *qObjectFromScriptValue(const QScriptValue &v)
{
     if (!v.isQObject())
        return 0;
    QObject *o = v.toQObject();
    return qobject_cast<QObjectDerived *>(o);
}

// Template that retrieves a wrapped object from a QScriptValue.
// The wrapped object is accessed through an accessor of
// the  QObject-derived wrapper.

template <class  Wrapper, class Wrapped>
    Wrapped *wrappedFromScriptValue(const QScriptValue &v,
                                    Wrapped * (Wrapper::*wrappedAccessor)  () const)
{
    Wrapper *wrapper = qObjectFromScriptValue<Wrapper>(v);
    if (!wrapper)
        return 0;
    return (wrapper->*wrappedAccessor)();
}

// Template that retrieves a wrapped object from
// a QObject-derived script wrapper object that is set as 'this' in
// a script context via accessor.

template <class  Wrapper, class Wrapped>
    static inline Wrapped *wrappedThisFromContext(QScriptContext *context,
                                                Wrapped * (Wrapper::*wrappedAccessor)  () const)
{
    Wrapped *wrapped = wrappedFromScriptValue(context->thisObject(), wrappedAccessor);
    Q_ASSERT(wrapped);
    return wrapped;
}

// Template that retrieves an object contained in a wrapped object
// in a script getter call (namely the interfaces returned by
// the core interface accessors). Mangles out the wrapper object from
// thisObject(), accesses the wrapped object and returns the contained object.

template <class Contained, class  Wrapper, class Wrapped>
    static inline Contained *containedFromContext(QScriptContext *context,
                                                  Wrapped *   (Wrapper::*wrappedAccessor)  () const,
                                                  Contained * (Wrapped::*containedAccessor)() const)
{
    Wrapped *wrapped = wrappedThisFromContext(context, wrappedAccessor);
    return (wrapped->*containedAccessor)();
}

// Template that retrieves a contained QObject-type object
// in a script getter call and creates a new script-object via engine->newQObject().
// To be called from a script getter callback.

template <class Contained, class  Wrapper, class Wrapped>
    static inline QScriptValue containedQObjectFromContextToScriptValue(QScriptContext *context, QScriptEngine *engine,
                                                                        Wrapped *   (Wrapper::*wrappedAccessor)  () const,
                                                                        Contained * (Wrapped::*containedAccessor)() const)
{
    return engine->newQObject(containedFromContext(context, wrappedAccessor, containedAccessor));
}

// Template that retrieves a contained Non-QObject-type object
// in a script getter call and creates a new script-object by wrapping it into
// a new instance of ContainedWrapper (which casts to QScriptValue).
// To be called from a script getter callback.

template <class ContainedWrapper, class Contained, class  Wrapper, class Wrapped>
    static inline QScriptValue wrapContainedFromContextAsScriptValue(QScriptContext *context, QScriptEngine *engine,
                                                                     Wrapped *   (Wrapper::*wrappedAccessor)  () const,
                                                                     Contained * (Wrapped::*containedAccessor)() const)
{
    Contained *c = containedFromContext(context, wrappedAccessor, containedAccessor);
    if (!c)
        return QScriptValue(engine, QScriptValue::NullValue);

    ContainedWrapper *cw = new ContainedWrapper(*engine, c);
    return *cw; // cast to QScriptValue
}

// Template that retrieves a wrapped object from context (this)
// and calls a const-member function with no parameters.
// To be called from a script getter callback.

template <class Ret, class  Wrapper, class Wrapped>
    static inline QScriptValue scriptCallConstMember_0(QScriptContext *context, QScriptEngine *engine,
                                                       Wrapped *   (Wrapper::*wrappedAccessor)  () const,
                                                       Ret  (Wrapped::*member)() const)
{
    Wrapped *wrapped = wrappedThisFromContext(context, wrappedAccessor);
    return engine->toScriptValue( (wrapped->*member)() );
}

// Ditto for non-const

template <class Ret, class  Wrapper, class Wrapped>
    static inline QScriptValue scriptCallMember_0(QScriptContext *context, QScriptEngine *engine,
                                                       Wrapped *   (Wrapper::*wrappedAccessor)  () const,
                                                       Ret  (Wrapped::*member)())
{
    Wrapped *wrapped = wrappedThisFromContext(context, wrappedAccessor);
    return engine->toScriptValue( (wrapped->*member)() );
}

// Template that retrieves a wrapped object from context (this)
// and calls a const-member function with 1 parameter on it.
// To be called from a script getter callback.

template <class Ret, class Argument, class  Wrapper, class Wrapped>
    static inline QScriptValue scriptCallConstMember_1(QScriptContext *context, QScriptEngine *engine,
                                                       Wrapped *   (Wrapper::*wrappedAccessor)  () const,
                                                       Ret  (Wrapped::*member)(Argument a1) const)
{
    const int argumentCount = context->argumentCount();
    if ( argumentCount != 1)
        return QScriptValue (engine, QScriptValue::NullValue);

    Wrapped *wrapped = wrappedThisFromContext(context, wrappedAccessor);
    // call member. If the argument is a const ref, strip it.
    typedef typename RemoveConstRef<Argument>::Result ArgumentBase;
    ArgumentBase a = qscriptvalue_cast<ArgumentBase>(context->argument(0));
    return engine->toScriptValue( (wrapped->*member)(a) );
}

// Template that retrieves a wrapped object
// and calls a member function with 1 parameter on it.
// To be called from a script getter callback.

template <class Ret, class Argument, class  Wrapper, class Wrapped>
    static inline QScriptValue scriptCallMember_1(QScriptContext *context, QScriptEngine *engine,
                                                  Wrapped *   (Wrapper::*wrappedAccessor)  () const,
                                                  Ret  (Wrapped::*member)(Argument a1))
{
    const int argumentCount = context->argumentCount();
    if ( argumentCount != 1)
        return QScriptValue (engine, QScriptValue::NullValue);

    Wrapped *wrapped = wrappedThisFromContext(context, wrappedAccessor);
    // call member. If the argument is a const ref, strip it.
    typedef typename RemoveConstRef<Argument>::Result ArgumentBase;
    ArgumentBase a = qscriptvalue_cast<ArgumentBase>(context->argument(0));
    return engine->toScriptValue( (wrapped->*member)(a) );
}

// Template that retrieves a wrapped object
// and calls a void member function with 1 parameter that is a wrapper of
// of some interface.
// Typically used for something like 'setCurrentEditor(Editor*)'
// To be called from a script callback.

template <class  ThisWrapper, class ThisWrapped, class ArgumentWrapper, class ArgumentWrapped>
static QScriptValue scriptCallVoidMember_Wrapped1(QScriptContext *context, QScriptEngine *engine,
                                                  ThisWrapped *   (ThisWrapper::*thisWrappedAccessor)  () const,
                                                  ArgumentWrapped *(ArgumentWrapper::*argumentWrappedAccessor)() const,
                                                  void  (ThisWrapped::*member)(ArgumentWrapped *a1),
                                                  bool acceptNullArgument = false)
{
    const QScriptValue voidRC = QScriptValue(engine, QScriptValue::UndefinedValue);
    if (context->argumentCount() < 1)
        return voidRC;

    ThisWrapped *thisWrapped = wrappedThisFromContext(context, thisWrappedAccessor);
    ArgumentWrapped *aw = wrappedFromScriptValue(context->argument(0), argumentWrappedAccessor);
    if (acceptNullArgument || aw)
        (thisWrapped->*member)(aw);
    return voidRC;
}

// Macros that define the static functions to call members

#define SCRIPT_CALL_CONST_MEMBER_0(funcName, accessor, member) \
static QScriptValue funcName(QScriptContext *context, QScriptEngine *engine) \
{   return SharedTools::scriptCallConstMember_0(context, engine, accessor, member); }

#define SCRIPT_CALL_MEMBER_0(funcName, accessor, member) \
static QScriptValue funcName(QScriptContext *context, QScriptEngine *engine) \
{   return SharedTools::scriptCallMember_0(context, engine, accessor, member); }

#define SCRIPT_CALL_CONST_MEMBER_1(funcName, accessor, member) \
static QScriptValue funcName(QScriptContext *context, QScriptEngine *engine) \
{   return SharedTools::scriptCallConstMember_1(context, engine, accessor, member); }

#define SCRIPT_CALL_MEMBER_1(funcName, accessor, member) \
static QScriptValue funcName(QScriptContext *context, QScriptEngine *engine) \
{   return SharedTools::scriptCallMember_1(context, engine, accessor, member); }

// Create a script list of wrapped non-qobjects by wrapping them.
// Wrapper must cast to QScriptValue.

template <class Wrapper, class Iterator>
    static inline QScriptValue wrapObjectList( QScriptEngine *engine, Iterator i1, Iterator i2)
{
    QScriptValue rc = engine->newArray(i2 - i1); // Grrr!
    quint32 i = 0;
    for ( ; i1 != i2 ; ++i1, i++) {
        Wrapper * wrapper =  new Wrapper(*engine, *i1);
        rc.setProperty(i, *wrapper);
    }
    return rc;
}

// Unwrap a list of wrapped objects from a script list.

template <class Wrapper, class Wrapped>
    static inline QList<Wrapped*> unwrapObjectList(const QScriptValue &v,
                                                   Wrapped *(Wrapper::*wrappedAccessor)() const)
{
    QList<Wrapped*> rc;

    if (!v.isArray())
        return rc;

    const  quint32 len = v.property(QLatin1String("length")).toUInt32();
    if (!len)
        return rc;

    for (quint32 i = 0; i < len; i++) {
        const QScriptValue e = v.property(i);
        if (e.isQObject()) {
            QObject *o = e.toQObject();
            if (Wrapper * wrapper = qobject_cast<Wrapper *>(o))
                rc.push_back((wrapper->*wrappedAccessor)());
        }
    }
    return rc;
}

// Traditional registration of a prototype for an interface.
// that can be converted via script value casts via Q_DECLARE_METATYPE.

template <class Interface, class Prototype>
    static void registerInterfaceWithDefaultPrototype(QScriptEngine &engine)
{
    Prototype *protoType = new Prototype(&engine);
    const QScriptValue scriptProtoType = engine.newQObject(protoType);

    engine.setDefaultPrototype(qMetaTypeId<Interface*>(), scriptProtoType);
}

// Convert a class derived from QObject to Scriptvalue via engine->newQObject() to make
// the signals, slots and properties visible.
// To be registered as a magic creation function with qScriptRegisterMetaType().
// (see registerQObject()

template <class SomeQObject>
static QScriptValue qObjectToScriptValue(QScriptEngine *engine, SomeQObject * const &qo)
{
    return engine->newQObject(qo, QScriptEngine::QtOwnership, QScriptEngine::ExcludeChildObjects);
}

// Convert  Scriptvalue back to a class derived from  QObject via QScriptValue::toQObject()
// To be registered as a magic conversion function with  qScriptRegisterMetaType().
// (see registerQObject)

template <class SomeQObject>
static void scriptValueToQObject(const QScriptValue &sv, SomeQObject * &p)
{
    QObject *qObject =  sv.toQObject();
    p = qobject_cast<SomeQObject*>(qObject);
    Q_ASSERT(p);
}

// Register a QObject-derived class which has Q_DECLARE_METATYPE(Ptr*)
// with the engine using qObjectToScriptValue/scriptValueToQObject as
// conversion functions to make it possible to use for example
// Q_PROPERTY(QMainWindow*).

template <class SomeQObject>
static void registerQObject(QScriptEngine *engine)
{
    qScriptRegisterMetaType<SomeQObject*>(engine,
                                          qObjectToScriptValue<SomeQObject>,
                                          scriptValueToQObject<SomeQObject>);
}

} // namespace SharedTools

#endif // WRAP_HELPERS_H
