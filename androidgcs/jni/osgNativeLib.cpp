/**
 ******************************************************************************
 * @file       osgNativeLib.cpp
 * @author     Android NDK examples
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2013
 * @brief      The core native library to interface to OSG
 * @see        The GNU Public License (GPL) Version 3
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
#include <string.h>
#include <jni.h>
#include <android/log.h>

#include <iostream>

#include "OsgMainApp.hpp"

OsgMainApp mainApp;

extern "C" {
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_init(JNIEnv * env, jobject obj, jint width, jint height);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_step(JNIEnv * env, jobject obj);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_clearContents(JNIEnv * env, jobject obj);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_mouseButtonPressEvent(JNIEnv * env, jobject obj, jfloat x, jfloat y, jint button);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_mouseButtonReleaseEvent(JNIEnv * env, jobject obj, jfloat x, jfloat y, jint button);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_mouseMoveEvent(JNIEnv * env, jobject obj, jfloat x, jfloat y);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_keyboardDown(JNIEnv * env, jobject obj, jint key);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_keyboardUp(JNIEnv * env, jobject obj, jint key);
    JNIEXPORT jintArray JNICALL Java_org_taulabs_osg_osgNativeLib_getClearColor(JNIEnv * env, jobject obj);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_setClearColor(JNIEnv * env, jobject obj, jint red, jint green, jint blue);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_loadObject(JNIEnv * env, jobject obj, jstring address);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_unLoadObject(JNIEnv * env, jobject obj, jint number);
    JNIEXPORT jobjectArray JNICALL Java_org_taulabs_osg_osgNativeLib_getObjectNames(JNIEnv * env, jobject obj);
    JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_setQuat(JNIEnv * env, jobject obj, jfloat q1, jfloat q2, jfloat q3, jfloat q4);
};

JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_init(JNIEnv * env, jobject obj, jint width, jint height){
    mainApp.initOsgWindow(0,0,width,height);
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_step(JNIEnv * env, jobject obj){
    mainApp.draw();
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_clearContents(JNIEnv * env, jobject obj){
    mainApp.clearScene();
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_mouseButtonPressEvent(JNIEnv * env, jobject obj, jfloat x, jfloat y, jint button){
    mainApp.mouseButtonPressEvent(x,y,button);
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_mouseButtonReleaseEvent(JNIEnv * env, jobject obj, jfloat x, jfloat y, jint button){
    mainApp.mouseButtonReleaseEvent(x,y,button);
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_mouseMoveEvent(JNIEnv * env, jobject obj, jfloat x, jfloat y){
    mainApp.mouseMoveEvent(x,y);
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_keyboardDown(JNIEnv * env, jobject obj, jint key){
    mainApp.keyboardDown(key);
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_keyboardUp(JNIEnv * env, jobject obj, jint key){
    mainApp.keyboardUp(key);
}
JNIEXPORT jintArray JNICALL Java_org_taulabs_osg_osgNativeLib_getClearColor(JNIEnv * env, jobject obj){

    jintArray color;
    color = env->NewIntArray(3);
    if (color == NULL) {
        return NULL;
    }
    osg::Vec4 vTemp1 = mainApp.getClearColor();

    jint vTemp2[3];

    vTemp2[0] = (int) (vTemp1.r() * 255);
    vTemp2[1] = (int) (vTemp1.g() * 255);
    vTemp2[2] = (int) (vTemp1.b() * 255);

    std::cout<<vTemp2[0]<<" "<<vTemp2[1]<<" "<<vTemp2[2]<<" "<<std::endl;

    env->SetIntArrayRegion(color, 0, 3, vTemp2);

    return color;
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_setClearColor(JNIEnv * env, jobject obj, jint red, jint green, jint blue){
    osg::Vec4 tVec((float) red / 255.0f, (float) green / 255.0f, (float) blue / 255.0f, 0.0f);
    mainApp.setClearColor(tVec);
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_loadObject(JNIEnv * env, jobject obj, jstring address){
    //Import Strings from JNI
    const char *nativeAddress = env->GetStringUTFChars(address, JNI_FALSE);

    mainApp.loadObject(std::string(nativeAddress));

    //Release Strings to JNI
    env->ReleaseStringUTFChars(address, nativeAddress);
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_loadObject(JNIEnv * env, jobject obj, jstring address, jstring name){
    //Import Strings from JNI
    const char *nativeAddress = env->GetStringUTFChars(address, JNI_FALSE);
    const char *nativeName = env->GetStringUTFChars(name, JNI_FALSE);

    mainApp.loadObject(std::string(nativeName),std::string(nativeAddress));

    //Release Strings to JNI
    env->ReleaseStringUTFChars(address, nativeAddress);
    env->ReleaseStringUTFChars(address, nativeName);
}
JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_unLoadObject(JNIEnv * env, jobject obj, jint number){

    mainApp.unLoadObject(number);

}
JNIEXPORT jobjectArray JNICALL Java_org_taulabs_osg_osgNativeLib_getObjectNames(JNIEnv * env, jobject obj){

    jobjectArray fileNames;
    unsigned int numModels = mainApp.getNumberObjects();
    fileNames = (jobjectArray)env->NewObjectArray(numModels,env->FindClass("java/lang/String"),env->NewStringUTF(""));

    for(unsigned int i=0;i < numModels;i++){
        std::string name = mainApp.getObjectName(i);
        env->SetObjectArrayElement(fileNames,i,env->NewStringUTF(name.c_str()));
    }

    return fileNames;
}

JNIEXPORT void JNICALL Java_org_taulabs_osg_osgNativeLib_setQuat(JNIEnv * env, jobject obj, jfloat q1, jfloat q2, jfloat q3, jfloat q4){
	float q[4] = {q1, q2, q3, q4};
	mainApp.setQuat(q);
}
