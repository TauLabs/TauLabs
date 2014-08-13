include(../gcs.pri)

TEMPLATE = subdirs
SUBDIRS = taulabs/translations

DATACOLLECTIONS = dials models pfd sounds diagrams mapicons stylesheets default_configurations

equals(copydata, 1) {
   for(dir, DATACOLLECTIONS) {
       exists($$GCS_SOURCE_TREE/share/taulabs/$$dir) {
           macx:data_copy.commands += $(COPY_DIR) $$targetPath(\"$$GCS_SOURCE_TREE/share/taulabs/$$dir\") $$targetPath(\"$$GCS_DATA_PATH/\") $$addNewline()
           win32:data_copy.commands += $(COPY_DIR) $$targetPath(\"$$GCS_SOURCE_TREE/share/taulabs/$$dir\") $$targetPath(\"$$GCS_DATA_PATH/$$dir\") $$addNewline()
           unix:data_copy.commands += $(MKDIR) $$targetPath(\"$$GCS_DATA_PATH/$$dir\") $$addNewline()
           unix:data_copy.commands += $(COPY_DIR) $$targetPath(\"$$GCS_SOURCE_TREE/share/taulabs/$$dir\") $$targetPath(\"$$GCS_DATA_PATH/\") $$addNewline()
       }
   }

    data_copy.target = FORCE
    QMAKE_EXTRA_TARGETS += data_copy
}
