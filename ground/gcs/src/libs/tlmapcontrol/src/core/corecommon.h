#ifndef CORECOMMON
#define CORECOMMON

#ifdef TLMAPWIDGET_LIBRARY
    #define TLMAPWIDGET_EXPORT __declspec(dllexport)
#else
    #define TLMAPWIDGET_EXPORT __declspec(dllimport)
#endif

#endif // CORECOMMON

