/**
 ******************************************************************************
 *
 * @file       uavobjectparser.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2015
 *
 * @brief      Parses XML files and extracts object information.
 *
 * @see        The GNU Public License (GPL) Version 3
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

#include <QtDebug>
#include <QTextStream>
#include "uavobjectparser.h"

/**
 * Constructor
 */
UAVObjectParser::UAVObjectParser()
{
    fieldTypeStrXML << "int8" << "int16" << "int32" << "uint8"
        << "uint16" << "uint32" <<"float" << "enum";

    updateModeStrXML << "manual" << "periodic" << "onchange" << "throttled";

    accessModeStr << "ACCESS_READWRITE" << "ACCESS_READONLY";

    fieldTypeNumBytes << int(1) << int(2) << int(4) <<
                        int(1) << int(2) << int(4) <<
                        int(4) << int(1);

    accessModeStrXML << "readwrite" << "readonly";

}

/**
 * Get number of objects
 */
int UAVObjectParser::getNumObjects()
{
    return objInfo.length();
}

/**
 * Get the UAVO hash
 */
quint64 UAVObjectParser::getUavoHash()
{
    return uavoHash;
}

/**
 * Get the detailed object information
 */
QList<ObjectInfo*> UAVObjectParser::getObjectInfo()
{
    return objInfo;
}

FieldInfo* UAVObjectParser::getFieldByName(QString &name, ObjectInfo **objRet) {
    if (objRet) {
        *objRet = NULL;
    }

    // Split name into object and field name
    QStringList splitStr = name.split(".");

    if (splitStr.size() != 2) {
        return NULL;
    }

    QString objName = splitStr[0].trimmed();
    QString fieldName = splitStr[1].trimmed();

    // Pull out the object
    ObjectInfo *obj = getObjectByName(objName);

    if (!obj) {
        return NULL;
    }

    // Retrieve field info
    foreach (FieldInfo *field, obj->fields) {
        if (field->name == fieldName) {
            // Got a match-- return it.
            if (objRet) {
                *objRet = obj;
            }

            return field;
        }
    }

    // Didn't find it? Give up. (return neither obj nor field)
    return NULL;
}

int UAVObjectParser::checkDefaultValues(FieldInfo *field)
{
    // Check that the default values are actually in the options list
    for(int n = 0; n < field->defaultValues.length(); ++n) {
        if (field->type == FIELDTYPE_ENUM && !field->options.contains(field->defaultValues[n])) {
            return -1;
        }
    }

    return 0;
}

int UAVObjectParser::resolveFieldParent(ObjectInfo *item, FieldInfo *field)
{
    if (field->parent) {
        // We could have done this already because of how we recursively
        // do stuff.
        return 0;
    }

    if (!field->parentName.isEmpty()) {
        // There is a parent relationship we have to resolve here.
        field->parent = getFieldByName(field->parentName,
                &field->parentObj);

        if (!field->parent) {
            return -1;
        }

        // Add to object parent set, if not present.
        item->parents.insert(field->parentObj);

        // Make sure any upwards dependencies are resolved before using the
        // field info.  This allows inheritance multiple levels deep.
        resolveFieldParent(field->parentObj, field->parent);

        // If the child had no options specified, take the list from
        // the parent
        if (field->options.isEmpty()) {
            field->options.append(field->parent->options);
        }
    }

    return 0;
}

QString UAVObjectParser::resolveParents()
{
    foreach (ObjectInfo *item, objInfo) {
        foreach (FieldInfo *field, item->fields) {
            // Because resolveFieldParent can recurse, make sure we've not
            // set a parent here already.
            if (resolveFieldParent(item, field) < 0) {
                return QString("Invalid parent for %1.%2")
                    .arg(item->name)
                    .arg(field->name);
            }

            for (int n = 0; n < field->options.length(); n++) {
                if (findOptionIndex(field, n) < 0) {
                    return QString("Parent of %1.%2.%3 missing")
                        .arg(item->name)
                        .arg(field->name)
                        .arg(field->options[n]);
                }
            }

            if(checkDefaultValues(field) < 0) {
                return QString("Invalid default value for %1.%2")
                    .arg(item->name)
                    .arg(field->name);
            }
        }
    }

    return QString();
}

void UAVObjectParser::calculateAllIds()
{
    foreach (ObjectInfo *item, objInfo) {
        calculateID(item);
    }

    /* Sort the object info list by ID, now that they're all defined */
    std::sort(objInfo.begin(), objInfo.end(), [](ObjectInfo *o1, ObjectInfo *o2) {
            return o2->id < o1->id;
            });

    /* This is a 64 bit hash to have a bit more protection against collisions
     * and to eventually supplant the current uavo-sha1 which takes into
     * account whitespace, etc.  It is not a cryptographically secure hash,
     * instead borrowing from the above shift-add-xor hash, but good enough.
     */
    quint64 hash=0;

    foreach (ObjectInfo *item, objInfo) {
        hash ^= (hash<<7) + (hash>>2) + item->id;
    }

    uavoHash = hash;
}

ObjectInfo* UAVObjectParser::getObjectByName(QString& name) {
    foreach (ObjectInfo *item, objInfo) {
        if (item->name == name) {
            return item;
        }
    }
    
    return NULL;
}


ObjectInfo* UAVObjectParser::getObjectByIndex(int objIndex)
{
    ObjectInfo *ret = objInfo[objIndex];

    if (ret != NULL) {
        if (ret->id == 0) {
            // Lazily calculate IDs on first retrieval.  This lets us have
            // dependent objects that are fixed up after parse completes.
            calculateID(ret);
        }
    }

    return ret;
}

/**
 * Get the name of the object
 */
QString UAVObjectParser::getObjectName(int objIndex)
{
    ObjectInfo* info = getObjectByIndex(objIndex);
    if (info == NULL)
        return QString();

    return info->name;
}

/**
 * Get the ID of the object
 */
quint32 UAVObjectParser::getObjectID(int objIndex)
{
    ObjectInfo* info = getObjectByIndex(objIndex);
    if (info == NULL)
        return 0;
    return info->id;
}

/**
 * Get the number of bytes in the data fields of this object
 */
int UAVObjectParser::getNumBytes(int objIndex)
{    
    ObjectInfo* info = getObjectByIndex(objIndex);
    if (info == NULL)
        return 0;
    return info->numBytes;
}

/**
 * Calculate the number of bytes in this object's fields
 */
void UAVObjectParser::calculateSize(ObjectInfo *info) {
    Q_ASSERT(info != NULL);
    info->numBytes = 0;
    for (int n = 0; n < info->fields.length(); ++n)
    {
        info->numBytes += info->fields[n]->numBytes * info->fields[n]->numElements;
    }
}

bool fieldTypeLessThan(const FieldInfo* f1, const FieldInfo* f2)
{
    return f1->numBytes > f2->numBytes;
}

/**
 * Returns a human-meaningful error message
 * @param fileName The filename we found an error in
 * @param errMsg A human readable description of what's wrong
 * @param errorLine The line number where this problem occurred
 * @param errorCol The column number
 * @returns A formatted error string.
 */
QString UAVObjectParser::genErrorMsg(QString& fileName,
        QString errMsg, int errorLine, int errorCol)
{
    QString result;

    QTextStream ts(&result);

    ts << fileName << ": " << errorLine << ":" << errorCol << ": " << errMsg;

    return result;
}

/**
 * Parse supplied XML file
 * @param xml The xml text
 * @param filename The xml filename
 * @returns Null QString() on success, error message on failure
 */
QString UAVObjectParser::parseXML(QString& xml, QString& filename)
{
    QString errorMsg;
    int errorLine;
    int errorCol;

    // Create DOM document and parse it
    QDomDocument doc("UAVObjects");
    if (!doc.setContent(xml, &errorMsg, &errorLine, &errorCol)) {
        return genErrorMsg(filename, errorMsg, errorLine, errorCol);
    }

    // Read all objects contained in the XML file, creating an new ObjectInfo for each
    QDomElement docElement = doc.documentElement();
    QDomNode node = docElement.firstChild();
    while ( !node.isNull() ) {
        // Create new object entry
        ObjectInfo* info = new ObjectInfo();
        info->filename=filename;
        // Process object attributes
        QString status = processObjectAttributes(node, info);
        if (!status.isNull())
            return status;

        // Process child elements (fields and metadata)
        QDomNode childNode = node.firstChild();
        int fieldFound = 0;
        int accessFound = 0;
        int telGCSFound = 0;
        int telFlightFound = 0;
        int logFound = 0;
        int descriptionFound = 0;
        while ( !childNode.isNull() ) {
            // Process element depending on its type
            if ( childNode.nodeName().compare(QString("field")) == 0 ) {
                QString status = processObjectFields(childNode, info);
                if (!status.isNull())
                    return genErrorMsg(filename, status,
                            childNode.lineNumber(), childNode.columnNumber());

                fieldFound++;
            }
            else if ( childNode.nodeName().compare(QString("access")) == 0 ) {
                QString status = processObjectAccess(childNode, info);
                if (!status.isNull())
                    return genErrorMsg(filename, status,
                            childNode.lineNumber(), childNode.columnNumber());

                accessFound++;
            }
            else if ( childNode.nodeName().compare(QString("telemetrygcs")) == 0 ) {
                QString status = processObjectMetadata(childNode, &info->gcsTelemetryUpdateMode,
                                                       &info->gcsTelemetryUpdatePeriod, &info->gcsTelemetryAcked);
                if (!status.isNull())
                    return genErrorMsg(filename, status,
                            childNode.lineNumber(), childNode.columnNumber());

                telGCSFound++;
            }
            else if ( childNode.nodeName().compare(QString("telemetryflight")) == 0 ) {
                QString status = processObjectMetadata(childNode, &info->flightTelemetryUpdateMode,
                                                       &info->flightTelemetryUpdatePeriod, &info->flightTelemetryAcked);
                if (!status.isNull())
                    return genErrorMsg(filename, status,
                            childNode.lineNumber(), childNode.columnNumber());

                telFlightFound++;
            }
            else if ( childNode.nodeName().compare(QString("logging")) == 0 ) {
                QString status = processObjectMetadata(childNode, &info->loggingUpdateMode,
                                                       &info->loggingUpdatePeriod, NULL);
                if (!status.isNull())
                    return genErrorMsg(filename, status,
                            childNode.lineNumber(), childNode.columnNumber());

                logFound++;
            }
            else if ( childNode.nodeName().compare(QString("description")) == 0 ) {
                QString status = processObjectDescription(childNode, &info->description);

                if (!status.isNull())
                    return genErrorMsg(filename, status,
                            childNode.lineNumber(), childNode.columnNumber());

                descriptionFound++;
            }
            else if (!childNode.isComment()) {
                return genErrorMsg(filename, "Unknown object element",
                        childNode.lineNumber(), childNode.columnNumber());
            }

            // Get next element
            childNode = childNode.nextSibling();
        }
        
        // Sort all fields according to size
        qStableSort(info->fields.begin(), info->fields.end(), fieldTypeLessThan);

        // Sort all fields according to size
        qStableSort(info->fields.begin(), info->fields.end(), fieldTypeLessThan);

        // Make sure that required elements were found
        if ( fieldFound == 0)
            return genErrorMsg(filename, "no field elements present",
                    node.lineNumber(), node.columnNumber());

        if ( accessFound != 1 )
            return genErrorMsg(filename, "missing or duplicate access element",
                    node.lineNumber(), node.columnNumber());

        if ( telGCSFound != 1 )
            return genErrorMsg(filename, "missing or duplicate telemetrygcs element",
                    node.lineNumber(), node.columnNumber());

        if ( telFlightFound != 1 )
            return genErrorMsg(filename, "missing or duplicate telemetryflight element",
                    node.lineNumber(), node.columnNumber());

        if ( logFound != 1 )
            return genErrorMsg(filename, "missing or duplicate logging element",
                    node.lineNumber(), node.columnNumber());

        if ( descriptionFound != 1 )
            return genErrorMsg(filename, "missing or duplicate description element",
                    node.lineNumber(), node.columnNumber());

        // Calculate size
        calculateSize(info);

        // Check size against max allowed
        if(info->numBytes > UAVO_MAX_SIZE)
            return genErrorMsg(filename, QString("total object size(%1 bytes) exceeds maximum limit (%2 bytes)")
                    .arg(QString::number(info->numBytes), QString::number(UAVO_MAX_SIZE)), 0, 0);

        // Add object
        objInfo.append(info);

        // Get next object
        node = node.nextSibling();
    }

    all_units.removeDuplicates();
    // Done, return null string
    return QString();
}

int UAVObjectParser::findOptionIndex(FieldInfo *field, quint32 inputIdx) {
    if (!field->parent) {
        return inputIdx;
    }

    // Walk up inheritance tree.
    FieldInfo *root = field->parent;

    while (root->parent) {
        root = root->parent;
    }

    QString& optionName = field->options[inputIdx];

    QStringList options = root->options;
    for (int m = 0; m < options.length(); m++) {
        if (optionName == options[m]) {
            return m;
        }
    }

    return -1;
}

/**
 * Calculate the unique object ID based on the object information.
 * The ID will change if the object definition changes, this is intentional
 * and is used to avoid connecting objects with incompatible configurations.
 * The LSB is set to zero and is reserved for metadata
 */
void UAVObjectParser::calculateID(ObjectInfo* info)
{
    // Hash object name
    quint32 hash = updateHash(info->name, 0);
    // Hash object attributes
    hash = updateHash(info->isSettings, hash);
    hash = updateHash(info->isSingleInst, hash);
    // Hash field information
    for (int n = 0; n < info->fields.length(); ++n) {
        hash = updateHash(info->fields[n]->name, hash);
        hash = updateHash(info->fields[n]->numElements, hash);
        hash = updateHash(info->fields[n]->type, hash);
        if(info->fields[n]->type == FIELDTYPE_ENUM) {
            QStringList options = info->fields[n]->options;
            int nextIdx = 0;

            for (int m = 0; m < options.length(); m++) {
                int idx = findOptionIndex(info->fields[n], m);

                if (idx < 0) abort();

                // Not contiguous options.  Update with next value.
                if (idx != nextIdx) {
                    hash = updateHash((quint32) idx, hash);
                }

                nextIdx = idx+1;

                hash = updateHash(options[m], hash);
            }
        }
    }
    // Done
    info->id = hash & 0xFFFFFFFE;
}

/**
 * Shift-Add-XOR hash implementation. LSB is set to zero, it is reserved
 * for the ID of the metaobject.
 *
 * http://eternallyconfuzzled.com/tuts/algorithms/jsw_tut_hashing.aspx
 */
quint32 UAVObjectParser::updateHash(quint32 value, quint32 hash)
{
    return (hash ^ ((hash<<5) + (hash>>2) + value));
}

/**
 * Update the hash given a string
 */
quint32 UAVObjectParser::updateHash(QString& value, quint32 hash)
{
    QByteArray bytes = value.toLatin1();
    quint32 hashout = hash;
    for (int n = 0; n < bytes.length(); ++n)
        hashout = updateHash(bytes[n], hashout);

    return hashout;
}

/**
 * Process the metadata part of the XML
 */
QString UAVObjectParser::processObjectMetadata(QDomNode& childNode, UpdateMode* mode, int* period, bool* acked)
{
    // Get updatemode attribute
    QDomNamedNodeMap elemAttributes = childNode.attributes();
    QDomNode elemAttr = elemAttributes.namedItem("updatemode");
    if ( elemAttr.isNull() )
        return QString("Object:telemetrygcs:updatemode attribute is missing");

    int index = updateModeStrXML.indexOf( elemAttr.nodeValue() );

    if (index<0)
        return QString("Object:telemetrygcs:updatemode attribute value is invalid");

    *mode = (UpdateMode)index;

    // Get period attribute
    elemAttr = elemAttributes.namedItem("period");
    if ( elemAttr.isNull() )
        return QString("Object:telemetrygcs:period attribute is missing");

    *period = elemAttr.nodeValue().toInt();


    // Get acked attribute (only if acked parameter is not null, not applicable for logging metadata)
    if ( acked != NULL) {
        elemAttr = elemAttributes.namedItem("acked");
        if ( elemAttr.isNull())
            return QString("Object:telemetrygcs:acked attribute is missing");

        if ( elemAttr.nodeValue().compare(QString("true")) == 0 )
            *acked = true;
        else if ( elemAttr.nodeValue().compare(QString("false")) == 0 )
            *acked = false;
        else
            return QString("Object:telemetrygcs:acked attribute value is invalid");
    }
    // Done
    return QString();
}

/**
 * Process the object access tag of the XML
 */
QString UAVObjectParser::processObjectAccess(QDomNode& childNode, ObjectInfo* info)
{
    // Get gcs attribute
    QDomNamedNodeMap elemAttributes = childNode.attributes();
    QDomNode elemAttr = elemAttributes.namedItem("gcs");
    if ( elemAttr.isNull() )
        return QString("Object:access:gcs attribute is missing");

    int index = accessModeStrXML.indexOf( elemAttr.nodeValue() );
    if (index >= 0)
        info->gcsAccess = (AccessMode)index;
    else
        return QString("Object:access:gcs attribute value is invalid");

    // Get flight attribute
    elemAttr = elemAttributes.namedItem("flight");
    if ( elemAttr.isNull() )
        return QString("Object:access:flight attribute is missing");

    index = accessModeStrXML.indexOf( elemAttr.nodeValue() );
    if (index >= 0)
        info->flightAccess = (AccessMode)index;
    else
        return QString("Object:access:flight attribute value is invalid");

    // Done
    return QString();
}

/**
 * Process the object fields of the XML
 */
QString UAVObjectParser::processObjectFields(QDomNode& childNode, ObjectInfo* info)
{
    // Create field
    FieldInfo* field = new FieldInfo();
    field->parent = NULL;
    // Get name attribute
    QDomNamedNodeMap elemAttributes = childNode.attributes();
    QDomNode elemAttr = elemAttributes.namedItem("name");
    if (elemAttr.isNull()) {
        return QString("Object:field:name attribute is missing");
    }
    QString name = elemAttr.nodeValue();

    // Check to see is this field is a clone of another
    // field that has already been declared
    elemAttr = elemAttributes.namedItem("cloneof");
    if (!elemAttr.isNull()) {
        QString parentName = elemAttr.nodeValue(); 
        if (!parentName.isEmpty()) {
           foreach(FieldInfo * parent, info->fields) {
                if (parent->name == parentName) {
                    // clone from this parent
                    *field = *parent;   // safe shallow copy, no ptrs in struct
                    field->name = name; // set our name
                    // Add field to object
                    info->fields.append(field);
                    // Done
                    return QString();
                }
            }
            return QString("Object:field::cloneof parent unknown");
        }
        else {
            return QString("Object:field:cloneof attribute is empty");
        }
    }
    else {
        // this field is not a clone, so remember its name
        field->name = name;
    }

    // Get units attribute
    elemAttr = elemAttributes.namedItem("units");
    if ( elemAttr.isNull() )
        return QString("Object:field:units attribute is missing");

    field->units = elemAttr.nodeValue();
    all_units << field->units;

    // Get type attribute
    elemAttr = elemAttributes.namedItem("type");
    if ( elemAttr.isNull() )
        return QString("Object:field:type attribute is missing");

    int index = fieldTypeStrXML.indexOf(elemAttr.nodeValue());
    if (index >= 0) {
        field->type = (FieldType)index;
        field->numBytes = fieldTypeNumBytes[index];
    }  
    else {
        return QString("Object:field:type attribute value is invalid");
    }

    // Get numelements or elementnames attribute
    field->numElements = 0;
    // Look for element names as an attribute first
    elemAttr = elemAttributes.namedItem("elementnames");
    if ( !elemAttr.isNull() ) {
        // Get element names
        QStringList names = elemAttr.nodeValue().split(",", QString::SkipEmptyParts);
        for (int n = 0; n < names.length(); ++n)
            names[n] = names[n].trimmed();

        field->elementNames = names;
        field->numElements = names.length();
        field->defaultElementNames = false;
    }
    else {
        // Look for a list of child elementname nodes
        QDomNode listNode = childNode.firstChildElement("elementnames");
        if (!listNode.isNull()) {
            for (QDomElement node = listNode.firstChildElement("elementname");
                 !node.isNull(); node = node.nextSiblingElement("elementname")) {
                QDomNode name = node.firstChild();
                if (!name.isNull() && name.isText() && !name.nodeValue().isEmpty()) {
                    field->elementNames.append(name.nodeValue());
                }
            }
            field->numElements = field->elementNames.length();
            field->defaultElementNames = false;
        }
    }
    // If no element names were found, then fall back to looking
    // for the number of elements in the 'elements' attribute
    if (field->numElements == 0) {
        elemAttr = elemAttributes.namedItem("elements");
        if ( elemAttr.isNull() ) {
            return QString("Object:field:elements and Object:field:elementnames attribute/element is missing");
        }
        else {
            field->numElements = elemAttr.nodeValue().toInt();

            // If the number of elements is still 0, return an error.
            if(field->numElements==0){
                return QString("--> " + info->name +"." + field->name + ": elements cannot be 0.");
            }

            for (int n = 0; n < field->numElements; ++n)
                field->elementNames.append(QString("%1").arg(n));

            field->defaultElementNames = true;
        }
    }
    // Get options attribute or child elements (only if an enum type)
    if (field->type == FIELDTYPE_ENUM) {
        elemAttr = elemAttributes.namedItem("parent");
        if (!elemAttr.isNull()) {
            field->parentName = elemAttr.nodeValue().trimmed();
        }

        // Look for options attribute
        elemAttr = elemAttributes.namedItem("options");
        if (!elemAttr.isNull()) {
            QStringList options = elemAttr.nodeValue().split(",", QString::SkipEmptyParts);
            for (int n = 0; n < options.length(); ++n) {
                options[n] = options[n].trimmed();
            }
            field->options = options;
        }
        else {
            // Look for a list of child 'option' nodes
            QDomNode listNode = childNode.firstChildElement("options");
            if (!listNode.isNull()) {
                for (QDomElement node = listNode.firstChildElement("option");
                     !node.isNull(); node = node.nextSiblingElement("option")) {
                    QDomNode name = node.firstChild();
                        if (!name.isNull() && name.isText() && !name.nodeValue().isEmpty()) {
                        field->options.append(name.nodeValue());
                    }
                }
            }
        }
        if ((field->options.isEmpty()) && (field->parentName.isEmpty())) {
            return QString("Object:field:options attribute/element is missing");
        }
    }

    // Get the default value attribute (required for settings objects, optional for the rest)
    elemAttr = elemAttributes.namedItem("defaultvalue");
    if ( elemAttr.isNull() ) {
        if ( info->isSettings )
            return QString("Object:field:defaultvalue attribute is missing (required for settings objects)");
        field->defaultValues = QStringList();
    }
    else  {
        QStringList defaults = elemAttr.nodeValue().split(",", QString::SkipEmptyParts);
        for (int n = 0; n < defaults.length(); ++n)
            defaults[n] = defaults[n].trimmed();

        if(defaults.length() != field->numElements) {
            if(defaults.length() != 1)
                return QString("Object:field:incorrect number of default values");

            /*support legacy single default for multiple elements
            We should really issue a warning*/
            for(int ct=1; ct< field->numElements; ct++)
                defaults.append(defaults[0]);
        }
        field->defaultValues = defaults;
    }

    // Limits attribute
    elemAttr = elemAttributes.namedItem("limits");
    if ( elemAttr.isNull() ) {
        field->limitValues=QString();
    }
    else{
        field->limitValues=elemAttr.nodeValue();
    }

    // Look for description string (for UI usage)
    QDomNode node = childNode.firstChildElement("description");
    if (!node.isNull()) {
        QDomNode description = node.firstChild();
        if (!description.isNull() && description.isText() && !description.nodeValue().isEmpty()) {
            field->description = description.nodeValue().trimmed();
        }
    }

    // Add field to object
    info->fields.append(field);
    // Done
    return QString();
}

/**
 * Process the object attributes from the XML
 */
QString UAVObjectParser::processObjectAttributes(QDomNode& node, ObjectInfo* info)
{
    // Get name attribute
    QDomNamedNodeMap attributes = node.attributes();
    QDomNode attr = attributes.namedItem("name");
    if ( attr.isNull() )
        return QString("Object:name attribute is missing");

    info->name = attr.nodeValue();
    info->namelc = attr.nodeValue().toLower();

    // Get category attribute if present
    attr = attributes.namedItem("category");
    if ( !attr.isNull() )
    {
        info->category = attr.nodeValue();
    }

    // Get singleinstance attribute
    attr = attributes.namedItem("singleinstance");
    if ( attr.isNull() )
        return QString("Object:singleinstance attribute is missing");

    if ( attr.nodeValue().compare(QString("true")) == 0 )
        info->isSingleInst = true;
    else if ( attr.nodeValue().compare(QString("false")) == 0 )
        info->isSingleInst = false;
    else
        return QString("Object:singleinstance attribute value is invalid");

    // Get settings attribute
    attr = attributes.namedItem("settings");
    if ( attr.isNull() )
        return QString("Object:settings attribute is missing");

    if ( attr.nodeValue().compare(QString("true")) == 0 )
            info->isSettings = true;
    else if ( attr.nodeValue().compare(QString("false")) == 0 )
        info->isSettings = false;
    else
        return QString("Object:settings attribute value is invalid");


    // Settings objects can only have a single instance
    if ( info->isSettings && !info->isSingleInst )
        return QString("Object: Settings objects can not have multiple instances");

    // Done
    return QString();
}

/**
 * Process the description field from the XML file
 */
QString UAVObjectParser::processObjectDescription(QDomNode& childNode, QString * description)
{
    description->append(childNode.firstChild().nodeValue());
    return QString();
}
