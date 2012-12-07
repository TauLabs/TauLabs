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
        if ( childField->numElements > 1 )
            buffer.append( QString("    %1 %2[%3];\r\n").arg(fieldTypeStrC[childField->type]).arg(childField->name).arg(childField->numElements) );
        else
            buffer.append( QString("    %1 %2;\r\n").arg(fieldTypeStrC[childField->type]).arg(childField->name) );
    }
    // this line is for retrocompatibility reasons, would better disapear but...
    QString structName = (field->parentField==NULL)?(field->name + QString("Data")):(field->name);
    datafields.prepend(QString("typedef struct {\r\n") + buffer +  QString("} __attribute__((packed))") + structName +QString(";\r\n\r\n"));


    // for each subfield of struct type, prepend datafields with the corresponding struct type definition
    foreach(FieldInfo* childField, field->childrenFields) {
        if(childField->type == FIELDTYPE_STRUCT)
            res = res && generateStructDefinitions(childField, datafields);
    }

    return res;

}


/**
 * @brief Retrieves the path to a specific field in the field arborescence
 * @param field
 * @return a list representing the path to the field. The root is the first element of the list, the field is the last element
 */
QStringList UAVObjectGeneratorFlight::fieldPath(FieldInfo* field) {
    QStringList res = QStringList();
    FieldInfo* current=field;

    while(current!=NULL) {

        res.prepend(current->name);

        current = current->parentField;

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
 * @brief Generate the Flight object files
 * @param info the object to process
 * @return true if everything went fine
 */
bool UAVObjectGeneratorFlight::processObject(ObjectInfo* info)
{
    if (info == NULL)
        return false;


    // Prepare output strings
    QString outInclude = flightIncludeTemplate;
    QString outCode = flightCodeTemplate;

    // Replace common tags
    replaceCommonTags(outInclude, info);
    replaceCommonTags(outCode, info);


    // Replace the $(DATAFIELDS) tag
    QString datafields = QString("");
    generateStructDefinitions(info->field,datafields);
    outInclude.replace(QString("$(DATAFIELDS)"), datafields);



    // Replace the $(DATAFIELDINFO) tag
    QString enums = QString("");
    generateEnumDefinitions(info->field,enums);
    outInclude.replace(QString("$(DATAFIELDINFO)"), enums);


//TODO from here replace to recursive
    // Replace the $(INITFIELDS) tag
    QString initfields;
    for (int n = 0; n < info->field->childrenFields.length(); ++n)
    {
        if (!info->field->childrenFields[n]->defaultValues.isEmpty() )
        {
            // For non-array fields
            if ( info->field->childrenFields[n]->numElements == 1)
            {
                if ( info->field->childrenFields[n]->type == FIELDTYPE_ENUM )
                {
                    initfields.append( QString("\tdata.%1 = %2;\r\n")
                                       .arg( info->field->childrenFields[n]->name )
                                       .arg( info->field->childrenFields[n]->options.indexOf( info->field->childrenFields[n]->defaultValues[0] ) ) );
                }
                else if ( info->field->childrenFields[n]->type == FIELDTYPE_FLOAT32 )
                {
                    initfields.append( QString("\tdata.%1 = %2;\r\n")
                                       .arg( info->field->childrenFields[n]->name )
                                       .arg( info->field->childrenFields[n]->defaultValues[0].toFloat() ) );
                }
                else
                {
                    initfields.append( QString("\tdata.%1 = %2;\r\n")
                                       .arg( info->field->childrenFields[n]->name )
                                       .arg( info->field->childrenFields[n]->defaultValues[0].toInt() ) );
                }
            }
            else
            {
                // Initialize all fields in the array
                for (int idx = 0; idx < info->field->childrenFields[n]->numElements; ++idx)
                {
                    if ( info->field->childrenFields[n]->type == FIELDTYPE_ENUM )
                    {
                        initfields.append( QString("\tdata.%1[%2] = %3;\r\n")
                                           .arg( info->field->childrenFields[n]->name )
                                           .arg( idx )
                                           .arg( info->field->childrenFields[n]->options.indexOf( info->field->childrenFields[n]->defaultValues[idx] ) ) );
                    }
                    else if ( info->field->childrenFields[n]->type == FIELDTYPE_FLOAT32 )
                    {
                        initfields.append( QString("\tdata.%1[%2] = %3;\r\n")
                                           .arg( info->field->childrenFields[n]->name )
                                           .arg( idx )
                                           .arg( info->field->childrenFields[n]->defaultValues[idx].toFloat() ) );
                    }
                    else
                    {
                        initfields.append( QString("\tdata.%1[%2] = %3;\r\n")
                                           .arg( info->field->childrenFields[n]->name )
                                           .arg( idx )
                                           .arg( info->field->childrenFields[n]->defaultValues[idx].toInt() ) );
                    }
                }
            }
        }
    }
    outCode.replace(QString("$(INITFIELDS)"), initfields);

    // Replace the $(SETGETFIELDS) tag
    QString setgetfields;
    for (int n = 0; n < info->field->childrenFields.length(); ++n)
    {
        //if (!info->field->childrenFields[n]->defaultValues.isEmpty() )
        {
            // For non-array fields
            if ( info->field->childrenFields[n]->numElements == 1)
            {

                /* Set */
                setgetfields.append( QString("void %2%3Set( %1 *New%3 )\r\n")
                                     .arg( fieldTypeStrC[info->field->childrenFields[n]->type] )
                                     .arg( info->name )
                                     .arg( info->field->childrenFields[n]->name ) );
                setgetfields.append( QString("{\r\n") );
                setgetfields.append( QString("\tUAVObjSetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), sizeof(%3));\r\n")
                                     .arg( info->name )
                                     .arg( info->field->childrenFields[n]->name )
                                     .arg( fieldTypeStrC[info->field->childrenFields[n]->type] ) );
                setgetfields.append( QString("}\r\n") );

                /* GET */
                setgetfields.append( QString("void %2%3Get( %1 *New%3 )\r\n")
                                     .arg( fieldTypeStrC[info->field->childrenFields[n]->type] )
                                     .arg( info->name )
                                     .arg( info->field->childrenFields[n]->name ));
                setgetfields.append( QString("{\r\n") );
                setgetfields.append( QString("\tUAVObjGetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), sizeof(%3));\r\n")
                                     .arg( info->name )
                                     .arg( info->field->childrenFields[n]->name )
                                     .arg( fieldTypeStrC[info->field->childrenFields[n]->type] ) );
                setgetfields.append( QString("}\r\n") );

            }
            else
            {

                /* SET */
                setgetfields.append( QString("void %2%3Set( %1 *New%3 )\r\n")
                                     .arg( fieldTypeStrC[info->field->childrenFields[n]->type] )
                                     .arg( info->name )
                                     .arg( info->field->childrenFields[n]->name ) );
                setgetfields.append( QString("{\r\n") );
                setgetfields.append( QString("\tUAVObjSetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), %3*sizeof(%4));\r\n")
                                     .arg( info->name )
                                     .arg( info->field->childrenFields[n]->name )
                                     .arg( info->field->childrenFields[n]->numElements )
                                     .arg( fieldTypeStrC[info->field->childrenFields[n]->type] ) );
                setgetfields.append( QString("}\r\n") );

                /* GET */
                setgetfields.append( QString("void %2%3Get( %1 *New%3 )\r\n")
                                     .arg( fieldTypeStrC[info->field->childrenFields[n]->type] )
                                     .arg( info->name )
                                     .arg( info->field->childrenFields[n]->name ) );
                setgetfields.append( QString("{\r\n") );
                setgetfields.append( QString("\tUAVObjGetDataField(%1Handle(), (void*)New%2, offsetof( %1Data, %2), %3*sizeof(%4));\r\n")
                                     .arg( info->name )
                                     .arg( info->field->childrenFields[n]->name )
                                     .arg( info->field->childrenFields[n]->numElements )
                                     .arg( fieldTypeStrC[info->field->childrenFields[n]->type] ) );
                setgetfields.append( QString("}\r\n") );
            }
        }
    }
    outCode.replace(QString("$(SETGETFIELDS)"), setgetfields);

    // Replace the $(SETGETFIELDSEXTERN) tag
    QString setgetfieldsextern;
    for (int n = 0; n < info->field->childrenFields.length(); ++n)
    {
        //if (!info->field->childrenFields[n]->defaultValues.isEmpty() )
        {

            /* SET */
            setgetfieldsextern.append( QString("extern void %2%3Set( %1 *New%3 );\r\n")
                                       .arg( fieldTypeStrC[info->field->childrenFields[n]->type] )
                                       .arg( info->name )
                                       .arg( info->field->childrenFields[n]->name ) );

            /* GET */
            setgetfieldsextern.append( QString("extern void %2%3Get( %1 *New%3 );\r\n")
                                       .arg( fieldTypeStrC[info->field->childrenFields[n]->type] )
                                       .arg( info->name )
                                       .arg( info->field->childrenFields[n]->name ) );
        }
    }
    outInclude.replace(QString("$(SETGETFIELDSEXTERN)"), setgetfieldsextern);

    // Write the flight code
    bool res = writeFileIfDiffrent( flightOutputPath.absolutePath() + "/" + info->namelc + ".c", outCode );
    if (!res) {
        cout << "Error: Could not write flight code files" << endl;
        return false;
    }

    res = writeFileIfDiffrent( flightOutputPath.absolutePath() + "/" + info->namelc + ".h", outInclude );
    if (!res) {
        cout << "Error: Could not write flight include files" << endl;
        return false;
    }

    return true;
}


