/**
 ******************************************************************************
 *
 * @file       uavobjectparser.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
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

#include "uavobjectparser.h"
#include <iostream>



using namespace std;


/**
 * Constructor
 */
UAVObjectParser::UAVObjectParser()
{
    fieldTypeStrXML << "struct" << "int8" << "int16" << "int32" << "uint8"
                    << "uint16" << "uint32" <<"float" << "enum";

    updateModeStrXML << "manual" << "periodic" << "onchange" << "throttled";

    accessModeStr << "ACCESS_READWRITE" << "ACCESS_READONLY";

    //Numbytes = -1 for structs, because it is computed dynamicly depending on the structs
    fieldTypeNumBytes << int(-1) << int(1) << int(2) << int(4) <<
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
 * Get the detailed object information
 */
QList<ObjectInfo*> UAVObjectParser::getObjectInfo()
{
    return objInfo;
}

ObjectInfo* UAVObjectParser::getObjectByIndex(int objIndex)
{
    return objInfo[objIndex];
}

/**
 * Get the name of the object
 */
QString UAVObjectParser::getObjectName(int objIndex)
{
    ObjectInfo* info = objInfo[objIndex];
    if (info == NULL)
        return QString();

    return info->name;
}

/**
 * Get the ID of the object
 */
quint32 UAVObjectParser::getObjectID(int objIndex)
{
    ObjectInfo* info = objInfo[objIndex];
    if (info == NULL)
        return 0;
    return info->id;
}

/**
 * Get the number of bytes in the data fields of this object
 */
int UAVObjectParser::getNumBytes(int objIndex)
{
    ObjectInfo* info = objInfo[objIndex];
    if (info == NULL)
    {
        return 0;
    }
    else
    {
        return info->field->numBytes;
    }
}

bool fieldTypeLessThan(const FieldInfo* f1, const FieldInfo* f2)
{
    return f1->numBytes > f2->numBytes;
}

/**
 * Parse supplied XML file
 * @param xml The xml text
 * @param filename The xml filename
 * @returns Null QString() on success, error message on failure
 */
QString UAVObjectParser::parseXML(QString& xml, QString& filename)
{
    // Create DOM document and parse it
    QDomDocument doc("UAVObjects");
    bool parsed = doc.setContent(xml);
    if (!parsed) return QString("Improperly formated XML file");

    // Read all objects contained in the XML file, creating an new ObjectInfo for each
    QDomElement docElement = doc.documentElement();
    QDomNode node = docElement.firstChild();
    while ( !node.isNull() ) {
        // Create new object entry
        ObjectInfo* info = new ObjectInfo;

        info->filename=filename;
        info->field=NULL;
        // Process object attributes
        QString status = processObjectAttributes(node, info);
        if (!status.isNull())
            return status;

        // Process child elements (fields and metadata)
        QDomNode childNode = node.firstChild();
        bool fieldFound = false;
        bool accessFound = false;
        bool telGCSFound = false;
        bool telFlightFound = false;
        bool logFound = false;
        bool descriptionFound = false;
        while ( !childNode.isNull() ) {


            if ( childNode.nodeName().compare(QString("data")) == 0 ) {
                // new UAVO data
                QString status = processField(childNode,NULL,info);
                if (!status.isNull())
                    return status;
                fieldFound = true;
            }
            else if ( childNode.nodeName().compare(QString("meta")) == 0 ) {
                // new UAVO metadata
                //TODO put this in a function
                QDomNode childchildNode = childNode.firstChild();
                while ( !childchildNode.isNull() ) {
                    if ( childchildNode.nodeName().compare(QString("access")) == 0 ) {
                        QString status = processObjectAccess(childchildNode, info);
                        if (!status.isNull())
                            return status;

                        accessFound = true;
                    }
                    else if ( childchildNode.nodeName().compare(QString("telemetrygcs")) == 0 ) {
                        QString status = processObjectMetadata(childchildNode, &info->gcsTelemetryUpdateMode,
                                                               &info->gcsTelemetryUpdatePeriod, &info->gcsTelemetryAcked);
                        if (!status.isNull())
                            return status;

                        telGCSFound = true;
                    }
                    else if ( childchildNode.nodeName().compare(QString("telemetryflight")) == 0 ) {
                        QString status = processObjectMetadata(childchildNode, &info->flightTelemetryUpdateMode,
                                                               &info->flightTelemetryUpdatePeriod, &info->flightTelemetryAcked);
                        if (!status.isNull())
                            return status;

                        telFlightFound = true;
                    }
                    else if ( childchildNode.nodeName().compare(QString("logging")) == 0 ) {
                        QString status = processObjectMetadata(childchildNode, &info->loggingUpdateMode,
                                                               &info->loggingUpdatePeriod, NULL);
                        if (!status.isNull())
                            return status;

                        logFound = true;
                    }
                    else if ( childchildNode.nodeName().compare(QString("description")) == 0 ) {
                        QString status = processObjectDescription(childchildNode, &info->description);

                        if (!status.isNull())
                            return status;

                        descriptionFound = true;
                    }
                    else if (!childchildNode.isComment()) {
                        return QString("Unknown object element in the meta section");
                    }
                    // Get next element
                    childchildNode = childchildNode.nextSibling();
                }
            }
            else if ( childNode.nodeName().compare(QString("field")) == 0 ) {
                // old schoold fields
                QString status = processObjectFields(childNode, info);
                if (!status.isNull())
                    return status;
                fieldFound = true;
            }
            else if ( childNode.nodeName().compare(QString("access")) == 0 ) {
                QString status = processObjectAccess(childNode, info);
                if (!status.isNull())
                    return status;

                accessFound = true;
            }
            else if ( childNode.nodeName().compare(QString("telemetrygcs")) == 0 ) {
                QString status = processObjectMetadata(childNode, &info->gcsTelemetryUpdateMode,
                                                       &info->gcsTelemetryUpdatePeriod, &info->gcsTelemetryAcked);
                if (!status.isNull())
                    return status;

                telGCSFound = true;
            }
            else if ( childNode.nodeName().compare(QString("telemetryflight")) == 0 ) {
                QString status = processObjectMetadata(childNode, &info->flightTelemetryUpdateMode,
                                                       &info->flightTelemetryUpdatePeriod, &info->flightTelemetryAcked);
                if (!status.isNull())
                    return status;

                telFlightFound = true;
            }
            else if ( childNode.nodeName().compare(QString("logging")) == 0 ) {
                QString status = processObjectMetadata(childNode, &info->loggingUpdateMode,
                                                       &info->loggingUpdatePeriod, NULL);
                if (!status.isNull())
                    return status;

                logFound = true;
            }
            else if ( childNode.nodeName().compare(QString("description")) == 0 ) {
                QString status = processObjectDescription(childNode, &info->description);

                if (!status.isNull())
                    return status;

                descriptionFound = true;
            }
            else if (!childNode.isComment()) {
                return QString("Unknown object element");
            }

            // Get next element
            childNode = childNode.nextSibling();
        }

        // Sort all fields according to size
        //        qStableSort(info->field->childrenFields.begin(), info->field->childrenFields.end(), fieldTypeLessThan);

        // Sort all fields according to size
        //        qStableSort(info->field->childrenFields.begin(), info->field->childrenFields.end(), fieldTypeLessThan);

        // Make sure that required elements were found
        if ( !fieldFound )
            return QString("Object::field element is missing");

        if ( !accessFound )
            return QString("Object::access element is missing");

        if ( !telGCSFound )
            return QString("Object::telemetrygcs element is missing");

        if ( !telFlightFound )
            return QString("Object::telemetryflight element is missing");

        if ( !logFound )
            return QString("Object::logging element is missing");

        // TODO: Make into error once all objects updated
        if ( !descriptionFound )
            return QString("Object::description element is missing");

        // Calculate ID
        calculateID(info);

        // Add object
        objInfo.append(info);

        // Get next object
        node = node.nextSibling();
    }

    all_units.removeDuplicates();
    // Done, return null string
    return QString();
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
    hash = updateHash(info->field,hash);
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
    QByteArray bytes = value.toAscii();
    quint32 hashout = hash;
    for (int n = 0; n < bytes.length(); ++n)
        hashout = updateHash(bytes[n], hashout);

    return hashout;
}

/**
 * Recursive update of hash with subfields hashes
 */
quint32 UAVObjectParser::updateHash(FieldInfo* field, quint32 hash)
{
    quint32 res = hash;
    res = updateHash(field->name, res);
    res = updateHash(field->numElements, res);
    res = updateHash(field->type, res);
    if(field->type == FIELDTYPE_ENUM) {
        QStringList options = field->options;
        for (int n = 0; n < options.length(); n++)
            res = updateHash(options[n], res);
    }
    if(field->type == FIELDTYPE_STRUCT) {
        QList<FieldInfo*> children = field->childrenFields;
        for (int n = 0; n < children.length(); n++)
            res = updateHash(children[n], res);
    }
    return res;
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
 * @brief Gets the size of a field, recursively if it is a struct field
 * @param field
 * @return size in bytes of the field, including its subfields
 */
int UAVObjectParser::fieldNumBytes(FieldInfo* field)
{
    int numBytes = 0;
    if(field->type==FIELDTYPE_STRUCT) {
        foreach(FieldInfo* child, field->childrenFields) {
            // Compute total number of bytes
            numBytes = numBytes + child->numBytes * child->numElements;
        }

    }
    else {
        numBytes = fieldTypeNumBytes[field->type];
    }

    return numBytes;
}

QStringList UAVObjectParser::fieldPath(FieldInfo* field) {
    QStringList res = QStringList();
    FieldInfo* current=field;
    while(current!=NULL) {
        res.prepend(current->name);
        current = current->parentField;
    }
    return res;
}

/**
 * @brief Process a field of the UAVO, recursively going into its subfields
 * @param node the XML Node
 * @param parent the parent field (should always be non null, except for the root field of the object)
 * @param info the UAVO info
 * @return an error string
 */
QString UAVObjectParser::processField(QDomNode& node, FieldInfo* parent, ObjectInfo* info)
{

    // Create field
    FieldInfo* field = new FieldInfo;
    field->parentField = parent;
    field->childrenFields = QList<FieldInfo*>();

    //If there is no parent, then we set the field as the root field of the UAVO
    if(field->parentField==NULL)  {
        info->field = field;
    }


    // Get name attribute
    QDomNamedNodeMap elemAttributes = node.attributes();
    QDomNode fieldNameAttr = elemAttributes.namedItem("name");
    QString name;
    if (!fieldNameAttr.isNull()) {
        name = fieldNameAttr.nodeValue();
    }
    else if(field->parentField == NULL) {
        // Default name for the root field is the UAVObject name
        name = QString(info->name) ;
    }
    else {
        return QString("Object:field:name attribute is missing");
    }
    // Check to see is this field is a clone of another
    // field that has already been declared
    fieldNameAttr = elemAttributes.namedItem("cloneof");
    if (!fieldNameAttr.isNull()) {
        QString cloneSourceName = fieldNameAttr.nodeValue();
        if (!cloneSourceName.isEmpty()) {
            foreach(FieldInfo * potentialCloneSource, parent->childrenFields) {
                if (potentialCloneSource->name == cloneSourceName) {
                    // clone from this parent
                    //TODO DEEP copy instead of shallow maybe ?
                    *field = *potentialCloneSource;   // safe shallow copy, no ptrs in struct
                    field->name = name; // set our name
                    field->parentField = parent; //set the parent
                    field->parentField->childrenFields.append(field);  // Add field to parent
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
    fieldNameAttr = elemAttributes.namedItem("units");
    if ( !fieldNameAttr.isNull() )
    {
        field->units = fieldNameAttr.nodeValue();
    }
    else
    {
        // If we don not have a units attribute, we inherit it from the parent
        if(field->parentField!=NULL)  {
            //Maybe ?
            //          field->units = QString(parent->units);
            field->units = field->parentField->units;
        }
        else {
            field->units = QString("");

            //            return QString("Object:field:units attribute is missing, and parent does not have units either");
        }
    }
    all_units << field->units;


    // Get type attribute
    fieldNameAttr = elemAttributes.namedItem("type");
    if ( !fieldNameAttr.isNull() ) {
        int index = fieldTypeStrXML.indexOf(fieldNameAttr.nodeValue());
        if (index >= 0) {
            field->type = (FieldType)index;
        }
        else {
            return QString("Object:field:type attribute value is invalid");
        }
    }
    else {
        if(field->parentField!=NULL) {
            return QString("Object:field:type attribute is missing");
        }
        else {
            field->type = FIELDTYPE_STRUCT;
        }
    }

    // Get nested structs
    // Important : This block is a pivot point :
    // Properties which flow from the root of the tree to the leafs (e.g. units, because units of a field = units of its parent by default ) must be computed before this point
    // Properties which flow from the leafs of the tree to the root (e.g. size, because size of field = sum of the size of children) must be computed after this point
    if(field->type==FIELDTYPE_STRUCT) {
        // Process sub fields recursively
        for (QDomElement itemNode = node.firstChildElement("data"); !itemNode.isNull(); itemNode = itemNode.nextSiblingElement("data")) {

            QString status = processField(itemNode,field,info);
            if(!status.isNull())
                return status;
        }
    }

    field->numBytes = fieldNumBytes(field);

    // Get numelements or elementnames attribute
    field->numElements = 0;
    // Look for element names as an attribute first
    fieldNameAttr = elemAttributes.namedItem("elementnames");
    if ( !fieldNameAttr.isNull() ) {
        // Get element names
        QStringList names = fieldNameAttr.nodeValue().split(",", QString::SkipEmptyParts);
        for (int n = 0; n < names.length(); ++n)
            names[n] = names[n].trimmed();

        field->elementNames = names;
        field->numElements = names.length();
        field->defaultElementNames = false;
    }
    else {
        // Look for a list of child elementname nodes
        QDomNode listNode = node.firstChildElement("elementnames");
        // To simplify syntax, new school UAVOs <elementnames> for array fields do not have to be inside of a <elementnames> node, they can just be free inside their parent node
        if(listNode.isNull())
            listNode = node;
        if (!listNode.isNull()) {
            for (QDomElement itemNode = listNode.firstChildElement("elementname");
                 !itemNode.isNull(); itemNode = itemNode.nextSiblingElement("elementname")) {
                QDomNode name = itemNode.firstChild();
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
        fieldNameAttr = elemAttributes.namedItem("elements");
        if ( fieldNameAttr.isNull() ) {
            if(field->parentField!=NULL) {
                field->numElements = 1;
                //                return QString("Object:field:elements and Object:field:elementnames attribute/element is missing");
            }
            else {
                field->numElements=1;
                field->defaultElementNames = true;
            }
        }
        else {
            field->numElements = fieldNameAttr.nodeValue().toInt();

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

        // Look for options attribute
        fieldNameAttr = elemAttributes.namedItem("options");
        if (!fieldNameAttr.isNull()) {
            QStringList options = fieldNameAttr.nodeValue().split(",", QString::SkipEmptyParts);
            for (int n = 0; n < options.length(); ++n) {
                options[n] = options[n].trimmed();
            }
            field->options = options;
        }
        else {
            // Look for a list of child 'option' nodes
            QDomNode listNode = node.firstChildElement("options");
            // To simplify syntax, new school UAVOs <option> for enum fields do not have to be inside of a <options> node, they can just be free inside their parent node
            if(listNode.isNull())
                listNode = node;
            if (!listNode.isNull()) {
                for (QDomElement itemNode = listNode.firstChildElement("option");
                     !itemNode.isNull(); itemNode = itemNode.nextSiblingElement("option")) {
                    QDomNode name = itemNode.firstChild();
                    if (!name.isNull() && name.isText() && !name.nodeValue().isEmpty()) {
                        field->options.append(name.nodeValue());
                    }
                }
            }
        }
        if (field->options.length() == 0) {
            return QString("Object:field:options attribute/element is missing");
        }
    }


    // Get the default value attribute (required for settings objects, optional for the rest)
    fieldNameAttr = elemAttributes.namedItem("defaultvalue");
    if(field->type != FIELDTYPE_STRUCT) {
        //This is not a struct, we can have default values
        fieldNameAttr = elemAttributes.namedItem("defaultvalue");
        if ( !fieldNameAttr.isNull() )  {
            QStringList defaults = fieldNameAttr.nodeValue().split(",", QString::SkipEmptyParts);
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
        else {


            QStringList defaults = QStringList();
            if(field->type != FIELDTYPE_ENUM) {
                //Default to zero for numerical fields
                for (int n = 0; n < field->numElements; ++n)
                    defaults.append("0");
            }
            else {
                //Default to the first enum option
                for (int n = 0; n < field->numElements; ++n)
                    defaults.append(field->options.first());
            }

            field->defaultValues = defaults;


            //            if ( info->isSettings ) {
            //                return QString("Object:field:defaultvalue attribute is missing (required for settings objects)");
            //            }
            //            field->defaultValues = QStringList();
        }
    }
    else {
        //This is a struct, throw an error if a default value is specified
        if ( !fieldNameAttr.isNull() ) {
            return QString("Object:field:defaultvalue is defined (forbidden for struct fields)");
        }
    }

    // Limits attribute
    fieldNameAttr = elemAttributes.namedItem("limits");
    if(field->type != FIELDTYPE_STRUCT) {
        if ( !fieldNameAttr.isNull() ) {
            field->limitValues=fieldNameAttr.nodeValue();
        }
        else {
            field->limitValues=QString();
        }
    }
    else {
        if ( !fieldNameAttr.isNull() ) {
            return QString("Object:field:limits is defined (forbidden for struct fields)");
        }
    }

    // Add field to parent field
    if(field->parentField!=NULL) {

        field->parentField->childrenFields.append(field);
    }

    // Done
    return QString();
}



/** This function exists only for retrocompatibility purposes, should be deprecated soon
 * @brief Process the object fields of a flat, old school UAVO
 * @param childNode xml node to process
 * @param info object to which this field belong
 * @return an error string
 */
QString UAVObjectParser::processObjectFields(QDomNode& childNode, ObjectInfo* info)
{


    // Create field
    FieldInfo* field = new FieldInfo;

    //Create root field if it does not exist yet (typically in old school UAVO files)
    if(info->field==NULL) {

        FieldInfo* rootField = new FieldInfo;
        rootField->name = QString(info->name);
        rootField->type=FIELDTYPE_STRUCT;
        rootField->numElements=1;
        rootField->defaultElementNames=true;
        rootField->parentField=NULL;
        rootField->units=QString("");
        rootField->childrenFields = QList<FieldInfo*>();

        info->field=rootField;
    }




    // Get name attribute
    QDomNamedNodeMap elemAttributes = childNode.attributes();
    QDomNode elemAttr = elemAttributes.namedItem("name");
    if (elemAttr.isNull()) {
        return QString("Object:field:name attribute is missing");
    }
    QString name = elemAttr.nodeValue().trimmed();



    // Check to see is this field is a clone of another
    // field that has already been declared
    elemAttr = elemAttributes.namedItem("cloneof");
    if (!elemAttr.isNull()) {
        QString parentName = elemAttr.nodeValue();
        if (!parentName.isEmpty()) {
            foreach(FieldInfo * parent, info->field->childrenFields) {

                if (parent->name == parentName) {

                    // clone from this parent
                    *field = *parent;   // safe shallow copy, no ptrs in struct
                    field->name = name; // set our name
                    // Add field to object
                    info->field->childrenFields.append(field);
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
        if (field->options.length() == 0) {
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
    // Add field to object

    info->field->childrenFields.append(field);
    field->parentField = info->field;
    //    info->field->childrenFields.append(new FieldInfo);

    info->field->numBytes = fieldNumBytes(info->field);

    //Done
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
