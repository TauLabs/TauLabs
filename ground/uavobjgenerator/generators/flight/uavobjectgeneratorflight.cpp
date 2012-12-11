/**
 ******************************************************************************
 *
 * @file       uavobjectgeneratorflight.cpp
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      produce flight code for uavobjects
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

#include "uavobjectgeneratorflight.h"

#include <iostream>

using namespace std;

/**
 * @brief Generate UAVOs for flight code
 * @param XML parser that contains parsed data
 * @param templatepath path for code templates
 * @param outputpath output path
 * @return true if everything went fine
 */
bool UAVObjectGeneratorFlight::generate(UAVObjectParser* parser,QString templatepath,QString outputpath) {

    fieldTypeStrC << "struct" << "int8_t" << "int16_t" << "int32_t" <<"uint8_t"
                  <<"uint16_t" << "uint32_t" << "float" << "uint8_t";

    QString flightObjInit,objInc,objFileNames,objNames;
    qint32 sizeCalc;
    flightCodePath = QDir( templatepath + QString("flight/targets/UAVObjects"));
    flightOutputPath = QDir( outputpath + QString("flight") );
    flightOutputPath.mkpath(flightOutputPath.absolutePath());

    flightCodeTemplate = readFile( flightCodePath.absoluteFilePath("uavobjecttemplate.c") );
    flightIncludeTemplate = readFile( flightCodePath.absoluteFilePath("inc/uavobjecttemplate.h") );
    flightInitTemplate = readFile( flightCodePath.absoluteFilePath("uavobjectsinittemplate.c") );
    flightInitIncludeTemplate = readFile( flightCodePath.absoluteFilePath("inc/uavobjectsinittemplate.h") );
    flightMakeTemplate = readFile( flightCodePath.absoluteFilePath("Makefiletemplate.inc") );

    if ( flightCodeTemplate.isNull() || flightIncludeTemplate.isNull() || flightInitTemplate.isNull()) {
        cout << "Error: Could not open flight template files." << endl;
        return false;
    }

    sizeCalc = 0;
    for (int objidx = 0; objidx < parser->getNumObjects(); ++objidx) {
        ObjectInfo* info=parser->getObjectByIndex(objidx);
        processObject(info);
        flightObjInit.append("#ifdef UAVOBJ_INIT_" + info->namelc +"\r\n");
        flightObjInit.append("    " + info->name + "Initialize();\r\n");
        flightObjInit.append("#endif\r\n");
        objInc.append("#include \"" + info->namelc + ".h\"\r\n");
        objFileNames.append(" " + info->namelc);
        objNames.append(" " + info->name);
        if (parser->getNumBytes(objidx)>sizeCalc) {
            sizeCalc = parser->getNumBytes(objidx);
        }
    }

    // Write the flight object inialization files
    flightInitTemplate.replace( QString("$(OBJINC)"), objInc);
    flightInitTemplate.replace( QString("$(OBJINIT)"), flightObjInit);
    bool res = writeFileIfDiffrent( flightOutputPath.absolutePath() + "/uavobjectsinit.c",
                                    flightInitTemplate );
    if (!res) {
        cout << "Error: Could not write flight object init file" << endl;
        return false;
    }

    // Write the flight object initialization header
    flightInitIncludeTemplate.replace( QString("$(SIZECALCULATION)"), QString().setNum(sizeCalc));
    res = writeFileIfDiffrent( flightOutputPath.absolutePath() + "/uavobjectsinit.h",
                               flightInitIncludeTemplate );
    if (!res) {
        cout << "Error: Could not write flight object init header file" << endl;
        return false;
    }

    // Write the flight object Makefile
    flightMakeTemplate.replace( QString("$(UAVOBJFILENAMES)"), objFileNames);
    flightMakeTemplate.replace( QString("$(UAVOBJNAMES)"), objNames);
    res = writeFileIfDiffrent( flightOutputPath.absolutePath() + "/Makefile.inc",
                               flightMakeTemplate );
    if (!res) {
        cout << "Error: Could not write flight Makefile" << endl;
        return false;
    }

    return true; // if we come here everything should be fine
}

/**
 * @brief Generate the Flight code for one UAVO
 * @param info the object to process
 * @return true if everything went fine
 */
bool UAVObjectGeneratorFlight::processObject(ObjectInfo* info)
{
    if (info == NULL)
        return false;

    bool res = true;

    // Prepare output strings
    QString outInclude = flightIncludeTemplate;
    QString outCode = flightCodeTemplate;

    // Replace common tags
    res = res && replaceCommonTags(outInclude, info);
    res = res && replaceCommonTags(outCode, info);

    // Replace the $(DATAFIELDS) tag
    QString datafields = QString("");
    res = res && generateStructDefinitions(info->field,datafields);
    outInclude.replace(QString("$(DATAFIELDS)"), datafields);

    // Replace the $(DATAFIELDINFO) tag
    QString enums = QString("");
    res = res && generateEnumDefinitions(info->field,enums);
    outInclude.replace(QString("$(DATAFIELDINFO)"), enums);

    // Replace the $(INITFIELDS) tag
    QString initfields = QString("");
    res = res && generateInitFields(info->field, initfields,QString("data"));
    outCode.replace(QString("$(INITFIELDS)"), initfields);

    // Replace the $(SETGETFIELDS) tag
    QString setgetfields = QString("");
    res = res && generateSetGetFields(info->field, setgetfields);
    outCode.replace(QString("$(SETGETFIELDS)"), setgetfields);

    // Replace the $(SETGETFIELDSEXTERN) tag
    QString setgetfieldsextern = QString("");
    res = res && generateExternSetGetFields(info->field, setgetfieldsextern);
    outInclude.replace(QString("$(SETGETFIELDSEXTERN)"), setgetfieldsextern);

    if (!res) {
        cout << "Error: Could not substitute tags" << endl;
        return false;
    }

    // Write the flight code
    res = res &&  writeFileIfDiffrent( flightOutputPath.absolutePath() + "/" + info->namelc + ".c", outCode );
    if (!res) {
        cout << "Error: Could not write flight code files" << endl;
        return false;
    }

    res = res &&  writeFileIfDiffrent( flightOutputPath.absolutePath() + "/" + info->namelc + ".h", outInclude );
    if (!res) {
        cout << "Error: Could not write flight include files" << endl;
        return false;
    }

    return res;
}


/**
 * @brief Retrieves the path to a specific field in the field arborescence
 * @param field
 * @return a list representing the path to the field. The root is the first element of the list, the field is the last element
 */
QStringList UAVObjectGeneratorFlight::fieldPath(FieldInfo* field) {
    QStringList result = QStringList();
    FieldInfo* current=field;
    while(current!=NULL) {
        result.prepend(current->name);
        current = current->parentField;
    }
    return result;
}

/**
 * @brief Generate the field type
 * @param field
 * @return the C type for the field, even if it is a struct field
 */
QString UAVObjectGeneratorFlight::fieldType(FieldInfo* field) {

    return (field->type == FIELDTYPE_STRUCT)?
                fieldPath(field).join("").replace(QRegExp(ENUM_SPECIAL_CHARS), "").append("Data"):
                fieldTypeStrC[field->type] ;
}

/**
 * @brief Generate the field descriptor
 * @param field
 * @return a string like "int8 blabla[10]"
 */
QString UAVObjectGeneratorFlight::fieldDescriptor(FieldInfo* field) {
    return (field->numElements > 1)?
                QString("%1 %2[%3]").arg(fieldType(field)).arg(field->name).arg(field->numElements):
                QString("%1 %2").arg(fieldType(field)).arg(field->name);
}


/**
 * @brief Generate the structs definitions string, recursively
 * @param field field to process
 * @param datafields string to prepend with the definition of the field
 * @return true if everything went fine
 */
bool UAVObjectGeneratorFlight::generateStructDefinitions(FieldInfo* field, QString& datafields)
{

    bool res = true;

    QString buffer = QString("");
    //for each subfield add its declaration in the struct declaration for the field
    foreach(FieldInfo* childField, field->childrenFields) {
            buffer.append( QString("    %1;\r\n").arg(fieldDescriptor(childField) ));
    }

    QString cStructName = fieldPath(field).join("").replace(QRegExp(ENUM_SPECIAL_CHARS), "") + QString("Data");
    datafields.prepend(QString("typedef struct {\r\n") + buffer +  QString("} __attribute__((packed)) ") + cStructName +QString(";\r\n\r\n"));


    // for each subfield of struct type, prepend datafields with the corresponding struct type definition
    foreach(FieldInfo* childField, field->childrenFields) {
        if(childField->type == FIELDTYPE_STRUCT) {
            res = res && generateStructDefinitions(childField, datafields);
        }
    }

    return res;
}


/**
 * @brief Generate the structs definitions string, recursively
 * @param field field to process
 * @param datafields string to prepend with the definition of the field
 * @return true if everything went fine
 */
bool UAVObjectGeneratorFlight::generateEnumDefinitions(FieldInfo* field, QString& enums)
{

    bool res = true;



    QString buffer = QString("");
    foreach(FieldInfo* childField, field->childrenFields) {

        // For enums
        if (childField->type == FIELDTYPE_ENUM)
        {

            buffer.append(QString("// Enumeration options for field %1 \r\n").arg(fieldPath(childField).join(QString("."))));
            buffer.append("typedef enum { ");
            // Go through each option
            QStringList options = QStringList();
            int i = 0;
            foreach(QString option,childField->options) {
                options.append(QString("%1_%2=%3").arg(fieldPath(childField).join(QString("_")).toUpper()).arg(option.toUpper().replace(QRegExp(ENUM_SPECIAL_CHARS), "")).arg(i));
                i++;
            }
            buffer.append(options.join(QString(", ")));
            buffer.append( QString(" } %1Options;\r\n").arg(fieldPath(childField).join(QString(""))));
        }


        // Generate element names (only if field has more than one element)
        if (childField->numElements > 1 && !childField->defaultElementNames)
        {

            buffer.append(QString("// Array element names for field %1 \r\n").arg(fieldPath(childField).join(QString("."))));
            buffer.append("typedef enum { ");
            // Go through the element names
            QStringList elementNames = QStringList();
            int i = 0;
            foreach(QString elementName,childField->elementNames) {
                elementNames.append(QString("%1_%2=%3").arg(fieldPath(childField).join(QString("_")).toUpper()).arg(elementName.toUpper().replace(QRegExp(ENUM_SPECIAL_CHARS), "")).arg(i));
                i++;
            }
            buffer.append(elementNames.join(QString(", ")));
            buffer.append( QString(" } %1Elem;\r\n").arg(fieldPath(childField).join(QString(""))));
        }
        // Generate array information
        if (childField->numElements > 1)
        {


            buffer.append(QString("// Number of elements for field %1 \r\n").arg(fieldPath(childField).join(QString("."))));

            buffer.append( QString("#define %1_NUMELEM %2\r\n")
                           .arg(fieldPath(childField).join(QString("_")).toUpper() )
                           .arg( childField->numElements ));
        }
    }
    enums.prepend(buffer);


    // process subfields
    foreach(FieldInfo* childField, field->childrenFields) {
        if(childField->type == FIELDTYPE_STRUCT)
            res = res && generateEnumDefinitions(childField, enums);
    }

    return res;
}


/**
 * @brief Generate instructions for the initialization of fields, recursively
 * @param field to initialize
 * @param initfields string to append with initialization code
 * @return true if everything went fine
 */
bool UAVObjectGeneratorFlight::generateInitFields(FieldInfo* field, QString& initfields,QString instanceName)
{
    bool res = true;

    foreach(FieldInfo* childField, field->childrenFields) {
        if (!childField->defaultValues.isEmpty() )
        {
            // For non-array fields
            if ( childField->numElements == 1)
            {
                if ( childField->type == FIELDTYPE_STRUCT )
                {
                    res = res && generateInitFields(childField,initfields,QString("%1.%2").arg(instanceName).arg(childField->name));
                }
                else if ( childField->type == FIELDTYPE_ENUM )
                {
                    initfields.append( QString("\t%1.%2 = %3;\r\n")
                                       .arg(instanceName)
                                       .arg( childField->name)
                                       .arg( childField->options.indexOf( childField->defaultValues[0] ) ) );
                }
                else if ( childField->type == FIELDTYPE_FLOAT32 )
                {
                    initfields.append( QString("\t%1.%2 = %3;\r\n")
                                       .arg(instanceName)
                                       .arg( childField->name)
                                       .arg( childField->defaultValues[0].toFloat() ) );
                }
                else
                {
                    initfields.append( QString("\t%1.%2 = %3;\r\n")
                                       .arg(instanceName)
                                       .arg( childField->name)
                                       .arg( childField->defaultValues[0].toInt() ) );
                }
            }
            else
            {
                // Initialize all fields in the array
                for (int idx = 0; idx < childField->numElements; ++idx)
                {
                    if ( childField->type == FIELDTYPE_STRUCT )
                    {
                        res = res && generateInitFields(childField,initfields,QString("%1.%2[%3]").arg(instanceName).arg(childField->name).arg( idx ));
                    }
                    else if ( childField->type == FIELDTYPE_ENUM )
                    {
                        initfields.append( QString("\t%1.%2[%3] = %4;\r\n")
                                           .arg(instanceName)
                                           .arg( childField->name)
                                           .arg( idx )
                                           .arg( childField->options.indexOf( childField->defaultValues[idx] ) ) );
                    }
                    else if ( childField->type == FIELDTYPE_FLOAT32 )
                    {
                        initfields.append( QString("\t%1.%2[%3] = %4;\r\n")
                                           .arg(instanceName)
                                           .arg( childField->name)
                                           .arg( idx )
                                           .arg( childField->defaultValues[idx].toFloat() ) );
                    }
                    else
                    {
                        initfields.append( QString("\t%1.%2[%3] = %4;\r\n")
                                           .arg(instanceName)
                                           .arg( childField->name)
                                           .arg( idx )
                                           .arg( childField->defaultValues[idx].toInt() ) );
                    }
                }
            }
        }
    }
    return res;
}

/**
 * @brief Generate getter and setter only for the first level, not recursively (because I do not see the use for this)
 * @param field
 * @param setgetfields the string to append with getters and setters
 * @return true if everything went fine
 */
bool UAVObjectGeneratorFlight::generateSetGetFields(FieldInfo* field, QString& setgetfields)
{
    foreach(FieldInfo* childField, field->childrenFields) {
        if ( childField->numElements == 1)
        {
            QStringList fieldpath = fieldPath(childField);
            QString fieldtype = fieldType(childField);
            /* Set */
            setgetfields.append( QString("void %1Set( %2 *New%3 )\r\n")
                                 .arg( fieldpath.join(QString("")) )
                                 .arg( fieldtype )
                                 .arg( fieldpath.last() ));
            setgetfields.append( QString("{\r\n") );
            setgetfields.append( QString("\tUAVObjSetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), sizeof(%3));\r\n")
                                 .arg( fieldpath.first() )
                                 .arg( fieldpath.last() )
                                 .arg( fieldtype ));
            setgetfields.append( QString("}\r\n") );
            /* Get */
            setgetfields.append( QString("void %1Get( %2 *New%3 )\r\n")
                                 .arg( fieldpath.join(QString("")) )
                                 .arg( fieldtype )
                                 .arg( fieldpath.last() ));
            setgetfields.append( QString("{\r\n") );
            setgetfields.append( QString("\tUAVObjGetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), sizeof(%3));\r\n")
                                 .arg( fieldpath.first() )
                                 .arg( fieldpath.last() )
                                 .arg( fieldtype ));
            setgetfields.append( QString("}\r\n") );
        }
        else
        {
            QStringList fieldpath = fieldPath(childField);
            QString fieldtype = fieldType(childField);
            int fieldnumelements = childField->numElements;
            /* Set */
            setgetfields.append( QString("void %1Set( %2 *New%3 )\r\n")
                                 .arg( fieldpath.join(QString("")) )
                                 .arg( fieldtype )
                                 .arg( fieldpath.last() ));
            setgetfields.append( QString("{\r\n") );
            setgetfields.append( QString("\tUAVObjSetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), %4*sizeof(%3));\r\n")
                                 .arg( fieldpath.first() )
                                 .arg( fieldpath.last() )
                                 .arg( fieldtype )
                                 .arg( fieldnumelements ));
            setgetfields.append( QString("}\r\n") );
            /* Get */
            setgetfields.append( QString("void %1Get( %2 *New%3 )\r\n")
                                 .arg( fieldpath.join(QString("")) )
                                 .arg( fieldtype )
                                 .arg( fieldpath.last() ));
            setgetfields.append( QString("{\r\n") );
            setgetfields.append( QString("\tUAVObjGetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), %4*sizeof(%3));\r\n")
                                 .arg( fieldpath.first() )
                                 .arg( fieldpath.last() )
                                 .arg( fieldtype )
                                 .arg( fieldnumelements ));
            setgetfields.append( QString("}\r\n") );
        }
    }
    return true;
}

/**
 * @brief Generate declaration of getters/setters in the header file
 * @param field
 * @param setgetfields the string to append with the declaration
 * @return true if everything went fine
 */
bool UAVObjectGeneratorFlight::generateExternSetGetFields(FieldInfo* field, QString& setgetfields)
{
    foreach(FieldInfo* childField, field->childrenFields) {
        if ( childField->numElements == 1)
        {
            QStringList fieldpath = fieldPath(childField);
            QString fieldtype = fieldType(childField);
            /* Set */
            setgetfields.append( QString("extern void %1Set( %2 *New%3 );\r\n")
                                 .arg( fieldpath.join(QString("")) )
                                 .arg( fieldtype )
                                 .arg( fieldpath.last() ));
            /* Get */
            setgetfields.append( QString("extern void %1Get( %2 *New%3 );\r\n")
                                 .arg( fieldpath.join(QString("")) )
                                 .arg( fieldtype )
                                 .arg( fieldpath.last() ));
        }
        else
        {
            QStringList fieldpath = fieldPath(childField);
            QString fieldtype = fieldType(childField);
            /* Set */
            setgetfields.append( QString("void %1Set( %2 *New%3 );\r\n")
                                 .arg( fieldpath.join(QString("")) )
                                 .arg( fieldtype )
                                 .arg( fieldpath.last() ));
            /* Get */
            setgetfields.append( QString("void %1Get( %2 *New%3 );\r\n")
                                 .arg( fieldpath.join(QString("")) )
                                 .arg( fieldtype )
                                 .arg( fieldpath.last() ));
        }
    }
    return true;
}

