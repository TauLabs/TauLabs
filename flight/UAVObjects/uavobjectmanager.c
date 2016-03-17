/**
******************************************************************************
 * @addtogroup TauLabsCore Tau Labs Core components
 * @{
 * @addtogroup UAVObjectHandling UAVObject handling code
 * @{
 *
 * @file       uavobjectmanager.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2012-2014
 * @author     dRonin, http://dronin.org Copyright (C) 2015
 * @brief      Object manager library. This library holds a collection of all objects.
 *             It can be used by all modules/libraries to find an object reference.
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

#include "openpilot.h"
#include "pios_struct_helper.h"
#include "pios_heap.h"		/* PIOS_malloc_no_dma */
#include "pios_mutex.h"
#include "pios_queue.h"
#include "misc_math.h"

extern uintptr_t pios_uavo_settings_fs_id;

// Constants

// Private types

// Macros
#define SET_BITS(var, shift, value, mask) var = (var & ~(mask << shift)) | (value << shift);

/**
 * List of event queues and the eventmask associated with the queue.
 */

/** opaque type for instances **/
typedef void* InstanceHandle; 

struct ObjectEventEntry {
	union {
		struct pios_queue         *queue;
		void                      *cbCtx;
	} cbInfo;

	UAVObjEventCallback       cb;
	uint8_t                   hasThrottle : 1;
	uint8_t                   eventMask : 7;
	struct ObjectEventEntry * next;
};

struct ObjectEventEntryThrottled {
	struct ObjectEventEntry   entry; // MUST be first! So throttled entry can be interpreted as ObjectEventEntry

	uint32_t                  due;
	uint16_t                  interval;
};

/*
  MetaInstance   == [UAVOBase [UAVObjMetadata]]
  SingleInstance == [UAVOBase [UAVOData [InstanceData]]]
  MultiInstance  == [UAVOBase [UAVOData [NumInstances [InstanceData0 [next]]]]
                                                  ____________________/
                                                  \-->[InstanceData1 [next]]
                                                  _________...________/
                                                  \-->[InstanceDataN [next]]
 */

/*
 * UAVO Base Type
 *   - All Types of UAVObjects are of this base type
 *   - The flags determine what type(s) this object
 */
struct UAVOBase {
	/* Let these objects be added to an event queue */
	struct ObjectEventEntry * next_event;

	/* Describe the type of object that follows this header */
	struct UAVOInfo {
		bool isMeta        : 1;
		bool isSingle      : 1;
		bool isSettings    : 1;
	} flags;

} __attribute__((packed));

/* Augmented type for Meta UAVO */
struct UAVOMeta {
	struct UAVOBase   base;
	UAVObjMetadata    instance0;
} __attribute__((packed));

/* Shared data structure for all data-carrying UAVObjects (UAVOSingle and UAVOMulti) */
struct UAVOData {
	struct UAVOBase   base;
	uint32_t          id;
	/*
	 * Embed the Meta object as another complete UAVO
	 * inside the payload for this UAVO.
	 */
	struct UAVOMeta   metaObj;
	struct UAVOData * next;
	uint16_t          instance_size;
} __attribute__((packed));

/* Augmented type for Single Instance Data UAVO */
struct UAVOSingle {
	struct UAVOData   uavo;

	uint8_t           instance0[];
	/* 
	 * Additional space will be malloc'd here to hold the
	 * the data for this instance.
	 */
} __attribute__((packed));

/* Part of a linked list of instances chained off of a multi instance UAVO. */
struct UAVOMultiInst {
	struct UAVOMultiInst * next;
	uint8_t                instance[];
	/* 
	 * Additional space will be malloc'd here to hold the
	 * the data for this instance.
	 */
} __attribute__((packed));

/* Augmented type for Multi Instance Data UAVO */
struct UAVOMulti {
	struct UAVOData        uavo;

	uint16_t               num_instances;
	struct UAVOMultiInst   instance0;
	/*
	 * Additional space will be malloc'd here to hold the
	 * the data for instance 0.
	 */
} __attribute__((packed));

/** all information about a metaobject are hardcoded constants **/
#define MetaNumBytes sizeof(UAVObjMetadata)

/* XXX TODO: All this reckless casting needs to die a horrific death! */
#define MetaBaseObjectPtr(obj) ((struct UAVOData *)((obj)-offsetof(struct UAVOData, metaObj)))
//#define MetaObjectPtr(obj) ((struct UAVOMeta*) &((obj)->metaObj))
#define MetaObjectPtr(obj) (&((obj)->metaObj.base))
#define MetaDataPtr(obj) ((UAVObjMetadata*)&((obj)->instance0))
#define LinkedMetaDataPtr(obj) ((UAVObjMetadata*)&((obj)->metaObj.instance0))
#define MetaObjectId(id) ((id)+1)

/** all information about instances are dependant on object type **/
#define ObjSingleInstanceDataOffset(obj) ((void*)(&(( (struct UAVOSingle*)obj )->instance0)))
#define InstanceDataOffset(inst) ((void*)&(( (struct UAVOMultiInst*)inst )->instance))
#define InstanceData(instance) (void*)instance

// Private functions
static int32_t sendEvent(struct UAVOBase * obj, uint16_t instId,
			UAVObjEventType event, void *obj_data, int len);
static InstanceHandle createInstance(struct UAVOData * obj, uint16_t instId);
static InstanceHandle getInstance(struct UAVOData * obj, uint16_t instId);
static int32_t connectObj(UAVObjHandle obj_handle, struct pios_queue *queue,
			UAVObjEventCallback cb, void *cbCtx, uint8_t eventMask,
			uint16_t interval);
static int32_t disconnectObj(UAVObjHandle obj_handle, struct pios_queue *queue,
			UAVObjEventCallback cb, void *cbCtx);

// Private variables
static struct UAVOData * uavo_list;
static struct ObjectEventEntry * events_unused;
static struct ObjectEventEntry * events_unused_throttled;
static struct pios_recursive_mutex *mutex;
static const UAVObjMetadata defMetadata = {
	.flags = (ACCESS_READWRITE << UAVOBJ_ACCESS_SHIFT |
		ACCESS_READWRITE << UAVOBJ_GCS_ACCESS_SHIFT |
		1 << UAVOBJ_TELEMETRY_ACKED_SHIFT |
		1 << UAVOBJ_GCS_TELEMETRY_ACKED_SHIFT |
		UPDATEMODE_ONCHANGE << UAVOBJ_TELEMETRY_UPDATE_MODE_SHIFT |
		UPDATEMODE_ONCHANGE << UAVOBJ_GCS_TELEMETRY_UPDATE_MODE_SHIFT),
	.telemetryUpdatePeriod    = 0,
	.gcsTelemetryUpdatePeriod = 0,
	.loggingUpdatePeriod      = 0,
};

static UAVObjStats stats;
static new_uavo_instance_cb_t newUavObjInstanceCB;

#define UAVO_CB_STACK_SIZE 512

static void *cb_stack;

/**
 * Initialize the object manager
 * \return 0 Success
 * \return -1 Failure
 */
int32_t UAVObjInitialize()
{
	// Initialize variables
	uavo_list = NULL;
	events_unused = NULL;
	events_unused_throttled = NULL;

	// Allocate the stack used for callbacks.
	cb_stack = PIOS_malloc_no_dma(UAVO_CB_STACK_SIZE);

	PIOS_Assert(cb_stack);

	// ARM stack grows down, so we should point to the "top valid" location
	cb_stack += UAVO_CB_STACK_SIZE - 4;

	memset(&stats, 0, sizeof(UAVObjStats));

	// Create mutex
	mutex = PIOS_Recursive_Mutex_Create();
	if (mutex == NULL)
		return -1;
	// Done
	return 0;
}

/*****************
 * Statistics
 ****************/

/**
 * Get the statistics counters
 * @param[out] statsOut The statistics counters will be copied there
 */
void UAVObjGetStats(UAVObjStats * statsOut)
{
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	memcpy(statsOut, &stats, sizeof(UAVObjStats));
	PIOS_Recursive_Mutex_Unlock(mutex);
}

/**
 * Clear the statistics counters
 */
void UAVObjClearStats()
{
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	memset(&stats, 0, sizeof(UAVObjStats));
	PIOS_Recursive_Mutex_Unlock(mutex);
}

/************************
 * Object Initialization
 ***********************/

static void UAVObjInitMetaData (struct UAVOMeta * obj_meta)
{
	/* Fill in the common part of the UAVO */
	struct UAVOBase * uavo_base = &(obj_meta->base);
	memset(uavo_base, 0, sizeof(*uavo_base));
	uavo_base->flags.isMeta   = true;
	uavo_base->flags.isSingle = true;
	uavo_base->next_event     = NULL;

	/* Clear the instance data carried in the UAVO */
	memset(&(obj_meta->instance0), 0, sizeof(obj_meta->instance0));
}

static struct UAVOData * UAVObjAllocSingle(uint32_t num_bytes)
{
	/* Compute the complete size of the object, including the data for a single embedded instance */
	uint32_t object_size = sizeof(struct UAVOSingle) + num_bytes;

	/* Allocate the object from the heap */
	struct UAVOSingle * uavo_single = (struct UAVOSingle *) PIOS_malloc_no_dma(object_size);
	if (!uavo_single)
		return (NULL);

	/* Fill in the common part of the UAVO */
	struct UAVOBase * uavo_base = &(uavo_single->uavo.base);
	memset(uavo_base, 0, sizeof(*uavo_base));
	uavo_base->flags.isSingle = true;
	uavo_base->next_event     = NULL;

	/* Clear the instance data carried in the UAVO */
	memset(&(uavo_single->instance0), 0, num_bytes);

	/* Give back the generic UAVO part */
	return (&(uavo_single->uavo));
}

static struct UAVOData * UAVObjAllocMulti(uint32_t num_bytes)
{
	/* Compute the complete size of the object, including the data for a single embedded instance */
	uint32_t object_size = sizeof(struct UAVOMulti) + num_bytes;

	/* Allocate the object from the heap */
	struct UAVOMulti * uavo_multi = (struct UAVOMulti *) PIOS_malloc_no_dma(object_size);
	if (!uavo_multi)
		return (NULL);

	/* Fill in the common part of the UAVO */
	struct UAVOBase * uavo_base = &(uavo_multi->uavo.base);
	memset(uavo_base, 0, sizeof(*uavo_base));
	uavo_base->flags.isSingle = false;
	uavo_base->next_event     = NULL;

	/* Set up the type-specific part of the UAVO */
	uavo_multi->num_instances = 1;

	/* Clear the instance data carried in the UAVO */
	uavo_multi->instance0.next = NULL;
	memset (&(uavo_multi->instance0.instance), 0, num_bytes);

	/* Give back the generic UAVO part */
	return (&(uavo_multi->uavo));
}

/**************************
 * UAVObject Database APIs
 *************************/

/**
 * Register and new object in the object manager.
 * \param[in] id Unique object ID
 * \param[in] isSingleInstance Is this a single instance or multi-instance object
 * \param[in] isSettings Is this a settings object
 * \param[in] numBytes Number of bytes of object data (for one instance)
 * \param[in] initCb Default field and metadata initialization function
 * \return Object handle, or NULL if failure.
 * \return
 */
UAVObjHandle UAVObjRegister(uint32_t id, 
			int32_t isSingleInstance, int32_t isSettings,
			uint32_t num_bytes,
			UAVObjInitializeCallback initCb)
{
	struct UAVOData * uavo_data = NULL;

	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	/* Don't allow duplicate registrations */
	if (UAVObjGetByID(id))
		goto unlock_exit;

	/* Map the various flags to one of the UAVO types we understand */
	if (isSingleInstance) {
		uavo_data = UAVObjAllocSingle (num_bytes);
	} else {
		uavo_data = UAVObjAllocMulti (num_bytes);
	}

	if (!uavo_data)
		goto unlock_exit;

	/* Fill in the details about this UAVO */
	uavo_data->id            = id;
	uavo_data->instance_size = num_bytes;
	if (isSettings) {
		uavo_data->base.flags.isSettings = true;
	}

	/* Initialize the embedded meta UAVO */
	UAVObjInitMetaData (&uavo_data->metaObj);

	/* Add the newly created object to the global list of objects */
	LL_APPEND(uavo_list, uavo_data);

	/* Initialize object fields and metadata to default values */
	if (initCb)
		initCb((UAVObjHandle) uavo_data, 0);

	/* Always try to load the meta object from flash */
	UAVObjLoad((UAVObjHandle) &(uavo_data->metaObj), 0);

	/* Attempt to load settings object from flash */
	if (uavo_data->base.flags.isSettings)
		UAVObjLoad((UAVObjHandle) uavo_data, 0);

	// fire events for outer object and its embedded meta object
	UAVObjInstanceUpdated((UAVObjHandle) uavo_data, 0);
	UAVObjInstanceUpdated((UAVObjHandle) &(uavo_data->metaObj), 0);

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return (UAVObjHandle) uavo_data;
}

/**
 * Retrieve an object from the list given its id
 * \param[in] The object ID
 * \return The object or NULL if not found.
 */
UAVObjHandle UAVObjGetByID(uint32_t id)
{
	UAVObjHandle found_obj = NULL;

	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	// Look for object
	struct UAVOData * tmp_obj;
	LL_FOREACH(uavo_list, tmp_obj) {
		if (tmp_obj->id == id) {
			found_obj = &tmp_obj->base;
			goto unlock_exit;
		}
		if (MetaObjectId(tmp_obj->id) == id) {
			found_obj = &(tmp_obj->metaObj.base);
			goto unlock_exit;
		}
	}

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return found_obj;
}

/**
 * Get the object's ID
 * \param[in] obj The object handle
 * \return The object ID
 */
uint32_t UAVObjGetID(UAVObjHandle obj_handle)
{
	PIOS_Assert(obj_handle);

	/* Recover the common object header */
	struct UAVOBase * uavo_base = (struct UAVOBase *) obj_handle;

	if (UAVObjIsMetaobject(obj_handle)) {
		/* We have a meta object, find our containing UAVO */
		struct UAVOData * uavo_data = container_of ((struct UAVOMeta *)uavo_base, struct UAVOData, metaObj);

		return MetaObjectId (uavo_data->id);
	} else {
		/* We have a data object, augment our pointer */
		struct UAVOData * uavo_data = (struct UAVOData *) uavo_base;

		return (uavo_data->id);
	}
}

/**
 * Get the number of bytes of the object's data (for one instance)
 * \param[in] obj The object handle
 * \return The number of bytes
 */
uint32_t UAVObjGetNumBytes(UAVObjHandle obj)
{
	PIOS_Assert(obj);

	uint32_t instance_size;

	/* Recover the common object header */
	struct UAVOBase * uavo_base = (struct UAVOBase *) obj;

	if (uavo_base->flags.isMeta) {
		instance_size = MetaNumBytes;
	} else {
		/* We have a data object, augment our pointer */
		struct UAVOData * uavo = (struct UAVOData *) uavo_base;

		instance_size = uavo->instance_size;
	}

	return (instance_size);
}

/**
 * Get the object this object is linked to. For regular objects, the linked object
 * is the metaobject. For metaobjects the linked object is the parent object.
 * This function is normally only needed by the telemetry module.
 * \param[in] obj The object handle
 * \return The object linked object handle
 */
UAVObjHandle UAVObjGetLinkedObj(UAVObjHandle obj_handle)
{
	PIOS_Assert(obj_handle);

	/* Recover the common object header */
	struct UAVOBase * uavo_base = (struct UAVOBase *) obj_handle;

	if (UAVObjIsMetaobject(obj_handle)) {
		/* We have a meta object, find our containing UAVO. */
		struct UAVOData * uavo_data = container_of ((struct UAVOMeta *)uavo_base, struct UAVOData, metaObj);

		return (UAVObjHandle) uavo_data;
	} else {
		/* We have a data object, augment our pointer */
		struct UAVOData * uavo_data = (struct UAVOData *) uavo_base;

		return (UAVObjHandle) &(uavo_data->metaObj);
	}
}

/**
 * Get the number of instances contained in the object.
 * \param[in] obj The object handle
 * \return The number of instances
 */
uint16_t UAVObjGetNumInstances(UAVObjHandle obj_handle)
{
	PIOS_Assert(obj_handle);

	if (UAVObjIsSingleInstance(obj_handle)) {
		/* Only one instance is allowed */
		return 1;
	} else {
		/* Multi-instance object.  Inspect the object */
		/* Augment our pointer to reflect the proper type */
		struct UAVOMulti * uavo_multi = (struct UAVOMulti *) obj_handle;
		return uavo_multi->num_instances;
	}
}

/**
 * Create a new instance in the object.
 * \param[in] obj The object handle
 * \return The instance ID or 0 if an error
 */
uint16_t UAVObjCreateInstance(UAVObjHandle obj_handle, UAVObjInitializeCallback initCb)
{
	PIOS_Assert(obj_handle);
	if (UAVObjIsMetaobject(obj_handle)) {
		return 0;
	}

	// Lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	InstanceHandle instEntry;
	uint16_t instId = 0;

	// Create new instance
	instId = UAVObjGetNumInstances(obj_handle);
	instEntry = createInstance((struct UAVOData *) obj_handle, instId);
	if (instEntry == NULL) {
		goto unlock_exit;
	}

	// Initialize instance data
	if (initCb) {
		initCb(obj_handle, instId);
	}

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);

	return instId;
}

/**
 * Does this object contains a single instance or multiple instances?
 * \param[in] obj The object handle
 * \return True (1) if this is a single instance object
 */
bool UAVObjIsSingleInstance(UAVObjHandle obj_handle)
{
	PIOS_Assert(obj_handle);

	return obj_handle->flags.isSingle;
}

/**
 * Is this a metaobject?
 * \param[in] obj The object handle
 * \return True (1) if this is metaobject
 */
bool UAVObjIsMetaobject(UAVObjHandle obj_handle)
{
	PIOS_Assert(obj_handle);

	/* Recover the common object header */
	struct UAVOBase * uavo_base = (struct UAVOBase *) obj_handle;

	return uavo_base->flags.isMeta;
}

/**
 * Is this a settings object?
 * \param[in] obj The object handle
 * \return True (1) if this is a settings object
 */
bool UAVObjIsSettings(UAVObjHandle obj_handle)
{
	PIOS_Assert(obj_handle);

	/* Recover the common object header */
	struct UAVOBase * uavo_base = (struct UAVOBase *) obj_handle;

	return uavo_base->flags.isSettings;
}

/**
 * Unpack an object from a byte array
 * \param[in] obj The object handle
 * \param[in] instId The instance ID
 * \param[in] dataIn The byte array
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjUnpack(UAVObjHandle obj_handle, uint16_t instId,
		const uint8_t * dataIn)
{
	PIOS_Assert(obj_handle);

	// Lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	void *target;
	int len;

	if (UAVObjIsMetaobject(obj_handle)) {
		if (instId != 0) {
			goto unlock_exit;
		}

		target = MetaDataPtr((struct UAVOMeta *)obj_handle);
		len = MetaNumBytes;
		memcpy(MetaDataPtr((struct UAVOMeta *)obj_handle), dataIn, MetaNumBytes);
	} else {
		struct UAVOData *obj;
		InstanceHandle instEntry;

		// Cast handle to object
		obj = (struct UAVOData *) obj_handle;

		// Get the instance
		instEntry = getInstance(obj, instId);

		// If the instance does not exist create it and any other instances before it
		if (instEntry == NULL) {
			instEntry = createInstance(obj, instId);
			if (instEntry == NULL) {
				goto unlock_exit;
			}
		}
		// Set the data

		target = InstanceData(instEntry);
		len = obj->instance_size;
	}

	memcpy(target, dataIn, len);

	// Fire event
	sendEvent((struct UAVOBase*)obj_handle, instId, EV_UNPACKED,
		target, len);

	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

/**
 * Pack an object to a byte array
 * \param[in] obj The object handle
 * \param[in] instId The instance ID
 * \param[out] dataOut The byte array
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjPack(UAVObjHandle obj_handle, uint16_t instId, uint8_t * dataOut)
{
	PIOS_Assert(obj_handle);

	// Lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	if (UAVObjIsMetaobject(obj_handle)) {
		if (instId != 0) {
			goto unlock_exit;
		}
		memcpy(dataOut, MetaDataPtr((struct UAVOMeta *)obj_handle), MetaNumBytes);
	} else {
		struct UAVOData *obj;
		InstanceHandle instEntry;

		// Cast handle to object
		obj = (struct UAVOData *) obj_handle;

		// Get the instance
		instEntry = getInstance(obj, instId);
		if (instEntry == NULL) {
			goto unlock_exit;
		}
		// Pack data
		memcpy(dataOut, InstanceData(instEntry), obj->instance_size);
	}

	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

#if defined(PIOS_INCLUDE_FASTHEAP)
/**
 * Trampoline buffer used for loads from the underlying filesystem.
 * This is required on platforms that store the UAVO data in non-DMA
 * RAM regions since the underlying flash driver may use DMA to transfer
 * the data into the buffer that we give it.
 */
static uint8_t uavobj_save_trampoline[256] __attribute__((aligned(4)));
#endif	/* PIOS_INCLUDE_FASTHEAP */

/**
 * Save the data of the specified object to the file system (SD card).
 * If the object contains multiple instances, all of them will be saved.
 * A new file with the name of the object will be created.
 * The object data can be restored using the UAVObjLoad function.
 * @param[in] obj The object handle.
 * @param[in] instId The instance ID
 * @param[in] file File to append to
 * @return 0 if success or -1 if failure
 */
int32_t UAVObjSave(UAVObjHandle obj_handle, uint16_t instId)
{
	PIOS_Assert(obj_handle);

	if (UAVObjIsMetaobject(obj_handle)) {
		if (instId != 0)
			return -1;

		// Save the object to the filesystem
		int32_t rc;
#if defined(PIOS_INCLUDE_FASTHEAP)
		memcpy(uavobj_save_trampoline,
			MetaDataPtr((struct UAVOMeta *)obj_handle),
			UAVObjGetNumBytes(obj_handle));

		rc = PIOS_FLASHFS_ObjSave(pios_uavo_settings_fs_id,
					UAVObjGetID(obj_handle),
					instId,
					uavobj_save_trampoline,
					UAVObjGetNumBytes(obj_handle));
#else /* PIOS_INCLUDE_FASTHEAP */
		rc = PIOS_FLASHFS_ObjSave(pios_uavo_settings_fs_id,
					UAVObjGetID(obj_handle),
					instId,
					(uint8_t*) MetaDataPtr((struct UAVOMeta *)obj_handle),
					UAVObjGetNumBytes(obj_handle));
#endif  /* PIOS_INCLUDE_FASTHEAP */

		if (rc != 0)
			return -1;
	} else {
		InstanceHandle instEntry = getInstance( (struct UAVOData *)obj_handle, instId);

		if (instEntry == NULL)
			return -1;

		if (InstanceData(instEntry) == NULL)
			return -1;

		// Save the object to the filesystem
		int32_t rc;
#if defined(PIOS_INCLUDE_FASTHEAP)
		memcpy(uavobj_save_trampoline,
			InstanceData(instEntry),
			UAVObjGetNumBytes(obj_handle));

		rc = PIOS_FLASHFS_ObjSave(pios_uavo_settings_fs_id,
					UAVObjGetID(obj_handle),
					instId,
					uavobj_save_trampoline,
					UAVObjGetNumBytes(obj_handle));
#else /* PIOS_INCLUDE_FASTHEAP */
		rc = PIOS_FLASHFS_ObjSave(pios_uavo_settings_fs_id,
					UAVObjGetID(obj_handle),
					instId,
					InstanceData(instEntry),
					UAVObjGetNumBytes(obj_handle));
#endif  /* PIOS_INCLUDE_FASTHEAP */

		if (rc != 0)
			return -1;
	}

	return 0;
}

#if defined(PIOS_INCLUDE_FASTHEAP)
/**
 * Trampoline buffer used for loads from the underlying filesystem.
 * This is required on platforms that store the UAVO data in non-DMA
 * RAM regions since the underlying flash driver may use DMA to transfer
 * the data into the buffer that we give it.
 */
static uint8_t uavobj_load_trampoline[256] __attribute__((aligned(4)));
#endif	/* PIOS_INCLUDE_FASTHEAP */

/**
 * Load an object from the file system (SD card).
 * A file with the name of the object will be opened.
 * The object data can be saved using the UAVObjSave function.
 * @param[in] obj The object handle.
 * @param[in] instId The object instance
 * @return 0 if success or -1 if failure
 */
int32_t UAVObjLoad(UAVObjHandle obj_handle, uint16_t instId)
{
	PIOS_Assert(obj_handle);

	void *target;
	int len;

	if (UAVObjIsMetaobject(obj_handle)) {
		if (instId != 0)
			return -1;

		target = MetaDataPtr((struct UAVOMeta *)obj_handle);
		len = UAVObjGetNumBytes(obj_handle);
	} else {

		InstanceHandle instEntry = getInstance( (struct UAVOData *)obj_handle, instId);

		if (instEntry == NULL)
			return -1;

		target = InstanceData(instEntry);
		len = UAVObjGetNumBytes(obj_handle);
	}

	// Load the object from the filesystem
	int32_t rc;
#if defined(PIOS_INCLUDE_FASTHEAP)
	rc = PIOS_FLASHFS_ObjLoad(pios_uavo_settings_fs_id,
			UAVObjGetID(obj_handle),
			instId,
			uavobj_load_trampoline,
			len);
#else  /* PIOS_INCLUDE_FASTHEAP */
	rc = PIOS_FLASHFS_ObjLoad(pios_uavo_settings_fs_id,
			UAVObjGetID(obj_handle),
			instId,
			target,
			len);
#endif  /* PIOS_INCLUDE_FASTHEAP */

	if (rc != 0)
		return -1;

#if defined(PIOS_INCLUDE_FASTHEAP)
	memcpy(target, uavobj_load_trampoline, len);
#endif  /* PIOS_INCLUDE_FASTHEAP */

	sendEvent((struct UAVOBase*)obj_handle, instId, EV_UNPACKED, target, len);
	return 0;
}

/**
 * Delete an object from the file system (SD card).
 * @param[in] obj_id The object id
 * @param[in] inst_id The object instance
 * @return 0 if success or -1 if failure
 */
int32_t UAVObjDeleteById(uint32_t obj_id, uint16_t inst_id)
{
	PIOS_FLASHFS_ObjDelete(pios_uavo_settings_fs_id, obj_id, inst_id);

	return 0;
}

/**
 * Save all settings objects to the SD card.
 * @return 0 if success or -1 if failure
 */
int32_t UAVObjSaveSettings()
{
	struct UAVOData *obj;

	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	// Save all settings objects
	LL_FOREACH(uavo_list, obj) {
		// Check if this is a settings object
		if (UAVObjIsSettings(&obj->base)) {
			// Save object
			if (UAVObjSave(&obj->base, 0) ==
				-1) {
				goto unlock_exit;
			}
		}
	}

	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

/**
 * Load all settings objects from the SD card.
 * @return 0 if success or -1 if failure
 */
int32_t UAVObjLoadSettings()
{
	struct UAVOData *obj;

	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	// Load all settings objects
	LL_FOREACH(uavo_list, obj) {
		// Check if this is a settings object
		if (UAVObjIsSettings(&obj->base)) {
			// Load object
			if (UAVObjLoad((UAVObjHandle) obj, 0) ==
				-1) {
				goto unlock_exit;
			}
		}
	}

	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

/**
 * Delete all settings objects from the SD card.
 * @return 0 if success or -1 if failure
 */
int32_t UAVObjDeleteSettings()
{
	struct UAVOData *obj;

	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	// Save all settings objects
	LL_FOREACH(uavo_list, obj) {
		// Check if this is a settings object
		if (UAVObjIsSettings(&obj->base)) {
			// Save object
			if (UAVObjDeleteById(UAVObjGetID(&obj->base), 0)
				== -1) {
				goto unlock_exit;
			}
		}
	}

	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

/**
 * Save all metaobjects to the SD card.
 * @return 0 if success or -1 if failure
 */
int32_t UAVObjSaveMetaobjects()
{
	struct UAVOData *obj;

	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	// Save all settings objects
	LL_FOREACH(uavo_list, obj) {
		// Save object
		if (UAVObjSave(MetaObjectPtr(obj), 0) ==
			-1) {
			goto unlock_exit;
		}
	}

	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

/**
 * Load all metaobjects from the SD card.
 * @return 0 if success or -1 if failure
 */
int32_t UAVObjLoadMetaobjects()
{
	struct UAVOData *obj;

	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	// Load all settings objects
	LL_FOREACH(uavo_list, obj) {
		// Load object
		if (UAVObjLoad((UAVObjHandle) MetaObjectPtr(obj), 0) ==
			-1) {
			goto unlock_exit;
		}
	}

	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

/**
 * Delete all metaobjects from the SD card.
 * @return 0 if success or -1 if failure
 */
int32_t UAVObjDeleteMetaobjects()
{
	struct UAVOData *obj;

	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	// Load all settings objects
	LL_FOREACH(uavo_list, obj) {
		// Load object
		if (UAVObjDeleteById(UAVObjGetID(MetaObjectPtr(obj)), 0)
			== -1) {
			goto unlock_exit;
		}
	}

	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

/**
 * Set the object data
 * \param[in] obj The object handle
 * \param[in] dataIn The object's data structure
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjSetData(UAVObjHandle obj_handle, const void *dataIn)
{
	return UAVObjSetInstanceData(obj_handle, 0, dataIn);
}

/**
 * Set the object data
 * \param[in] obj The object handle
 * \param[in] dataIn The object's data structure
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjSetDataField(UAVObjHandle obj_handle, const void* dataIn, uint32_t offset, uint32_t size)
{
	return UAVObjSetInstanceDataField(obj_handle, 0, dataIn, offset, size);
}

/**
 * Get the object data
 * \param[in] obj The object handle
 * \param[out] dataOut The object's data structure
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjGetData(UAVObjHandle obj_handle, void *dataOut)
{
	return UAVObjGetInstanceData(obj_handle, 0, dataOut);
}

/**
 * Get the object data
 * \param[in] obj The object handle
 * \param[out] dataOut The object's data structure
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjGetDataField(UAVObjHandle obj_handle, void* dataOut, uint32_t offset, uint32_t size)
{
	return UAVObjGetInstanceDataField(obj_handle, 0, dataOut, offset, size);
}

#define INSTANCE_COPY_ALL 0xffffffff

/**
 * Set the data of a specific object instance
 * \param[in] obj The object handle
 * \param[in] instId The object instance ID
 * \param[in] dataIn The object's data structure
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjSetInstanceData(UAVObjHandle obj_handle, uint16_t instId,
			const void *dataIn)
{
	return UAVObjSetInstanceDataField(obj_handle, instId, dataIn,
		0, INSTANCE_COPY_ALL);
}

/**
 * Set the data of a specific object instance
 * \param[in] obj The object handle
 * \param[in] instId The object instance ID
 * \param[in] dataIn The object's data structure
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjSetInstanceDataField(UAVObjHandle obj_handle, uint16_t instId, const void* dataIn, uint32_t offset, uint32_t size)
{
	PIOS_Assert(obj_handle);

	// Lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	void *target;
	int obj_len;

	if (UAVObjIsMetaobject(obj_handle)) {
		// Get instance information
		if (instId != 0) {
			goto unlock_exit;
		}

		obj_len = MetaNumBytes;

		target = MetaDataPtr((struct UAVOMeta *)obj_handle);
	} else {
		struct UAVOData * obj;
		InstanceHandle instEntry;

		// Cast to object info
		obj = (struct UAVOData *)obj_handle;

		// Check access level
		if (UAVObjReadOnly(obj_handle)) {
			goto unlock_exit;
		}

		// Get instance information
		instEntry = getInstance(obj, instId);
		if (instEntry == NULL) {
			goto unlock_exit;
		}

		obj_len = obj->instance_size;

		target = InstanceData(instEntry);
	}

	if (size == INSTANCE_COPY_ALL) {
		size = obj_len;
	}

	// Check for overrun
	if ((size + offset) > obj_len) {
		// XXX Should consider asserting!!!
		goto unlock_exit;
	}

	// Set data
	memcpy(target + offset, dataIn, size);

	// Fire event
	sendEvent((struct UAVOBase *)obj_handle, instId, EV_UPDATED,
		target, obj_len);
	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

/**
 * Get the data of a specific object instance
 * \param[in] obj The object handle
 * \param[in] instId The object instance ID
 * \param[out] dataOut The object's data structure
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjGetInstanceData(UAVObjHandle obj_handle, uint16_t instId,
			void *dataOut)
{
	PIOS_Assert(obj_handle);

	// Lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	if (UAVObjIsMetaobject(obj_handle)) {
		// Get instance information
		if (instId != 0) {
			goto unlock_exit;
		}
		// Set data
		memcpy(dataOut, MetaDataPtr((struct UAVOMeta *)obj_handle), MetaNumBytes);
	} else {
		struct UAVOData *obj;
		InstanceHandle instEntry;

		// Cast to object info
		obj = (struct UAVOData *) obj_handle;

		// Get instance information
		instEntry = getInstance(obj, instId);
		if (instEntry == NULL) {
			goto unlock_exit;
		}
		// Set data
		memcpy(dataOut, InstanceData(instEntry), obj->instance_size);
	}

	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

/**
 * Get the data of a specific object instance
 * \param[in] obj The object handle
 * \param[in] instId The object instance ID
 * \param[out] dataOut The object's data structure
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjGetInstanceDataField(UAVObjHandle obj_handle, uint16_t instId, void* dataOut, uint32_t offset, uint32_t size)
{
	PIOS_Assert(obj_handle);

	// Lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	int32_t rc = -1;

	if (UAVObjIsMetaobject(obj_handle)) {
		// Get instance information
		if (instId != 0) {
			goto unlock_exit;
		}

		// Check for overrun
		if ((size + offset) > MetaNumBytes) {
			goto unlock_exit;
		}

		// Set data
		memcpy(dataOut, MetaDataPtr((struct UAVOMeta *)obj_handle) + offset, size);
	} else {
		struct UAVOData * obj;
		InstanceHandle instEntry;

		// Cast to object info
		obj = (struct UAVOData *)obj_handle;

		// Get instance information
		instEntry = getInstance(obj, instId);
		if (instEntry == NULL) {
			goto unlock_exit;
		}

		// Check for overrun
		if ((size + offset) > obj->instance_size) {
			goto unlock_exit;
		}
		
		// Set data
		memcpy(dataOut, InstanceData(instEntry) + offset, size);
	}

	rc = 0;

unlock_exit:
	PIOS_Recursive_Mutex_Unlock(mutex);
	return rc;
}

/**
 * Set the object metadata
 * \param[in] obj The object handle
 * \param[in] dataIn The object's metadata structure
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjSetMetadata(UAVObjHandle obj_handle, const UAVObjMetadata * dataIn)
{
	PIOS_Assert(obj_handle);

	// Set metadata (metadata of metaobjects can not be modified)
	if (UAVObjIsMetaobject(obj_handle)) {
		return -1;
	}

	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	UAVObjSetData((UAVObjHandle) MetaObjectPtr((struct UAVOData *)obj_handle), dataIn);

	PIOS_Recursive_Mutex_Unlock(mutex);
	return 0;
}

/**
 * Get the object metadata
 * \param[in] obj The object handle
 * \param[out] dataOut The object's metadata structure
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjGetMetadata(UAVObjHandle obj_handle, UAVObjMetadata * dataOut)
{
	PIOS_Assert(obj_handle);

	// Lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	// Get metadata
	if (UAVObjIsMetaobject(obj_handle)) {
		memcpy(dataOut, &defMetadata, sizeof(UAVObjMetadata));
	} else {
		UAVObjGetData((UAVObjHandle) MetaObjectPtr( (struct UAVOData *)obj_handle ),
			dataOut);
	}

	// Unlock
	PIOS_Recursive_Mutex_Unlock(mutex);
	return 0;
}

/*******************************
 * Object Metadata Manipulation
 ******************************/

/**
 * Get the UAVObject metadata access member
 * \param[in] metadata The metadata object
 * \return the access type
 */
UAVObjAccessType UAVObjGetAccess(const UAVObjMetadata* metadata)
{
	PIOS_Assert(metadata);
	return (metadata->flags >> UAVOBJ_ACCESS_SHIFT) & 1;
}

/**
 * Set the UAVObject metadata access member
 * \param[in] metadata The metadata object
 * \param[in] mode The access mode
 */
void UAVObjSetAccess(UAVObjMetadata* metadata, UAVObjAccessType mode)
{
	PIOS_Assert(metadata);
	SET_BITS(metadata->flags, UAVOBJ_ACCESS_SHIFT, mode, 1);
}

/**
 * Get the UAVObject metadata GCS access member
 * \param[in] metadata The metadata object
 * \return the GCS access type
 */
UAVObjAccessType UAVObjGetGcsAccess(const UAVObjMetadata* metadata)
{
	PIOS_Assert(metadata);
	return (metadata->flags >> UAVOBJ_GCS_ACCESS_SHIFT) & 1;
}

/**
 * Set the UAVObject metadata GCS access member
 * \param[in] metadata The metadata object
 * \param[in] mode The access mode
 */
void UAVObjSetGcsAccess(UAVObjMetadata* metadata, UAVObjAccessType mode) {
	PIOS_Assert(metadata);
	SET_BITS(metadata->flags, UAVOBJ_GCS_ACCESS_SHIFT, mode, 1);
}

/**
 * Get the UAVObject metadata telemetry acked member
 * \param[in] metadata The metadata object
 * \return the telemetry acked boolean
 */
uint8_t UAVObjGetTelemetryAcked(const UAVObjMetadata* metadata) {
	PIOS_Assert(metadata);
	return (metadata->flags >> UAVOBJ_TELEMETRY_ACKED_SHIFT) & 1;
}

/**
 * Set the UAVObject metadata telemetry acked member
 * \param[in] metadata The metadata object
 * \param[in] val The telemetry acked boolean
 */
void UAVObjSetTelemetryAcked(UAVObjMetadata* metadata, uint8_t val) {
	PIOS_Assert(metadata);
	SET_BITS(metadata->flags, UAVOBJ_TELEMETRY_ACKED_SHIFT, val, 1);
}

/**
 * Get the UAVObject metadata GCS telemetry acked member
 * \param[in] metadata The metadata object
 * \return the telemetry acked boolean
 */
uint8_t UAVObjGetGcsTelemetryAcked(const UAVObjMetadata* metadata) {
	PIOS_Assert(metadata);
	return (metadata->flags >> UAVOBJ_GCS_TELEMETRY_ACKED_SHIFT) & 1;
}

/**
 * Set the UAVObject metadata GCS telemetry acked member
 * \param[in] metadata The metadata object
 * \param[in] val The GCS telemetry acked boolean
 */
void UAVObjSetGcsTelemetryAcked(UAVObjMetadata* metadata, uint8_t val) {
	PIOS_Assert(metadata);
	SET_BITS(metadata->flags, UAVOBJ_GCS_TELEMETRY_ACKED_SHIFT, val, 1);
}

/**
 * Get the UAVObject metadata telemetry update mode
 * \param[in] metadata The metadata object
 * \return the telemetry update mode
 */
UAVObjUpdateMode UAVObjGetTelemetryUpdateMode(const UAVObjMetadata* metadata) {
	PIOS_Assert(metadata);
	return (metadata->flags >> UAVOBJ_TELEMETRY_UPDATE_MODE_SHIFT) & UAVOBJ_UPDATE_MODE_MASK;
}

/**
 * Set the UAVObject metadata telemetry update mode member
 * \param[in] metadata The metadata object
 * \param[in] val The telemetry update mode
 */
void UAVObjSetTelemetryUpdateMode(UAVObjMetadata* metadata, UAVObjUpdateMode val) {
	PIOS_Assert(metadata);
	SET_BITS(metadata->flags, UAVOBJ_TELEMETRY_UPDATE_MODE_SHIFT, val, UAVOBJ_UPDATE_MODE_MASK);
}

/**
 * Get the UAVObject metadata GCS telemetry update mode
 * \param[in] metadata The metadata object
 * \return the GCS telemetry update mode
 */
UAVObjUpdateMode UAVObjGetGcsTelemetryUpdateMode(const UAVObjMetadata* metadata) {
	PIOS_Assert(metadata);
	return (metadata->flags >> UAVOBJ_GCS_TELEMETRY_UPDATE_MODE_SHIFT) & UAVOBJ_UPDATE_MODE_MASK;
}

/**
 * Set the UAVObject metadata GCS telemetry update mode member
 * \param[in] metadata The metadata object
 * \param[in] val The GCS telemetry update mode
 */
void UAVObjSetGcsTelemetryUpdateMode(UAVObjMetadata* metadata, UAVObjUpdateMode val) {
	PIOS_Assert(metadata);
	SET_BITS(metadata->flags, UAVOBJ_GCS_TELEMETRY_UPDATE_MODE_SHIFT, val, UAVOBJ_UPDATE_MODE_MASK);
}


/**
 * Check if an object is read only
 * \param[in] obj The object handle
 * \return 
 *   \arg 0 if not read only 
 *   \arg 1 if read only
 *   \arg -1 if unable to get meta data
 */
int8_t UAVObjReadOnly(UAVObjHandle obj_handle)
{
	PIOS_Assert(obj_handle);
	if (!UAVObjIsMetaobject(obj_handle)) {
		return UAVObjGetAccess(LinkedMetaDataPtr( (struct UAVOData *)obj_handle)) == ACCESS_READONLY;
	}
	return -1;
}

/**
 * Connect an event queue to the object, if the queue is already connected then the event mask is only updated.
 * All events matching the event mask will be pushed to the event queue.
 * \param[in] obj The object handle
 * \param[in] queue The event queue
 * \param[in] eventMask The event mask, if EV_MASK_ALL_UPDATES then all events are enabled (e.g. EV_UPDATED | EV_UPDATED_MANUAL)
 * \param[in] interval The interval at which to throttle updates; 0 is unthrottled
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjConnectQueueThrottled(UAVObjHandle obj_handle,
		struct pios_queue *queue, uint8_t eventMask, uint16_t interval)
{
	PIOS_Assert(obj_handle);
	PIOS_Assert(queue);
	int32_t res;
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	res = connectObj(obj_handle, queue, NULL, NULL, eventMask, interval);
	PIOS_Recursive_Mutex_Unlock(mutex);
	return res;
}

int32_t UAVObjConnectQueue(UAVObjHandle obj_handle, struct pios_queue *queue,
		uint8_t eventMask) {
	return UAVObjConnectQueueThrottled(obj_handle, queue, eventMask, 0);
}


/**
 * Disconnect an event queue from the object.
 * \param[in] obj The object handle
 * \param[in] queue The event queue
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjDisconnectQueue(UAVObjHandle obj_handle, struct pios_queue *queue)
{
	PIOS_Assert(obj_handle);
	PIOS_Assert(queue);
	int32_t res;
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	res = disconnectObj(obj_handle, queue, NULL, NULL);
	PIOS_Recursive_Mutex_Unlock(mutex);
	return res;
}

/**
 * Sets a flag passed in the ctx parameter to true.
 * Conforms to the UAVObjConnectCallback* signature.
 *
 * Using this is a best practice to listen for configuration changes.  This
 * sets a volatile flag that you can check at the top of the task main function,
 * and update configuration as appropriate.
 *
 * Note that the flag is considered a uint8_t, but the width doesn't really
 * matter-- it will be "set" as long as it is at least 8 bits wide.
 *
 * \param[in] ctx The event callback context
 */
void UAVObjCbSetFlag(UAVObjEvent *objEv, void *ctx, void *obj, int len) {
	volatile uint8_t *flag = ctx;

	*flag = 1;
}

/**
 * Copies the passed in object to the ctx pointer.
 * Conforms to the UAVObjConnectCallback* signature.
 *
 * Using UAVObjCbSetFlag is preferred.  This should only be used via the
 * wrapper in the individual UAV objects to ensure that objects are not
 * mismatched (wrong registration type -> ctx mapping).
 *
 * \param[in] ctx The event callback context.
 * \param[in] obj The pointer to the raw object data.
 * \param[in] len The length of data to copy.
 */
void UAVObjCbCopyData(UAVObjEvent *objEv, void *ctx, void *obj, int len) {
	memcpy(ctx, obj, len);
}

/**
 * Connect an event callback to the object, if the callback is already connected then the event mask is only updated.
 * The supplied callback will be invoked on all events matching the event mask.
 * \param[in] obj The object handle
 * \param[in] cb The event callback
 * \param[in] eventMask The event mask, if EV_MASK_ALL_UPDATES then all events are enabled (e.g. EV_UPDATED | EV_UPDATED_MANUAL)
 * \param[in] interval The interval at which to throttle updates; 0 is unthrottled
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjConnectCallbackThrottled(UAVObjHandle obj_handle, UAVObjEventCallback cb,
			void *cbCtx, uint8_t eventMask, uint16_t interval)
{
	PIOS_Assert(obj_handle);
	int32_t res;
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	res = connectObj(obj_handle, 0, cb, cbCtx, eventMask, interval);
	PIOS_Recursive_Mutex_Unlock(mutex);
	return res;
}

int32_t UAVObjConnectCallback(UAVObjHandle obj_handle, UAVObjEventCallback cb,
			void *cbCtx, uint8_t eventMask)
{
	return UAVObjConnectCallbackThrottled(obj_handle, cb, cbCtx, eventMask, 0);
}

/**
 * Disconnect an event callback from the object.
 * \param[in] obj The object handle
 * \param[in] cb The event callback
 * \return 0 if success or -1 if failure
 */
int32_t UAVObjDisconnectCallback(UAVObjHandle obj_handle, UAVObjEventCallback cb,
		void *cbCtx)
{
	PIOS_Assert(obj_handle);
	int32_t res;
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	res = disconnectObj(obj_handle, 0, cb, cbCtx);
	PIOS_Recursive_Mutex_Unlock(mutex);
	return res;
}

/**
 * Request an update of the object's data from the GCS. The call will not wait for the response, a EV_UPDATED event
 * will be generated as soon as the object is updated.
 * \param[in] obj The object handle
 */
void UAVObjRequestUpdate(UAVObjHandle obj_handle)
{
	UAVObjRequestInstanceUpdate(obj_handle, UAVOBJ_ALL_INSTANCES);
}

/**
 * Request an update of the object's data from the GCS. The call will not wait for the response, a EV_UPDATED event
 * will be generated as soon as the object is updated.
 * \param[in] obj The object handle
 * \param[in] instId Object instance ID to update
 */
void UAVObjRequestInstanceUpdate(UAVObjHandle obj_handle, uint16_t instId)
{
	PIOS_Assert(obj_handle);
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	sendEvent((struct UAVOBase *) obj_handle, instId, EV_UPDATE_REQ,
		NULL, 0);
	PIOS_Recursive_Mutex_Unlock(mutex);
}

/**
 * Send the object's data to the GCS (triggers a EV_UPDATED_MANUAL event on this object).
 * \param[in] obj The object handle
 */
void UAVObjUpdated(UAVObjHandle obj_handle)
{
	UAVObjInstanceUpdated(obj_handle, UAVOBJ_ALL_INSTANCES);
}

/**
 * Send the object's data to the GCS (triggers a EV_UPDATED_MANUAL event on this object).
 * \param[in] obj The object handle
 * \param[in] instId The object instance ID
 */
void UAVObjInstanceUpdated(UAVObjHandle obj_handle, uint16_t instId)
{
	PIOS_Assert(obj_handle);
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	sendEvent((struct UAVOBase *) obj_handle, instId, EV_UPDATED_MANUAL,
		NULL, 0);
	PIOS_Recursive_Mutex_Unlock(mutex);
}

/**
 * Iterate through all objects in the list.
 * \param iterator This function will be called once for each object,
 * the object will be passed as a parameter
 */
void UAVObjIterate(void (*iterator) (UAVObjHandle obj))
{
	PIOS_Assert(iterator);

	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	// Iterate through the list and invoke iterator for each object
	struct UAVOData *obj;
	LL_FOREACH(uavo_list, obj) {
		(*iterator) ((UAVObjHandle) obj);
		(*iterator) ((UAVObjHandle) &obj->metaObj);
	}

	// Release lock
	PIOS_Recursive_Mutex_Unlock(mutex);
}

/* type signature must match invokeCallback below, with 4 or fewer args */
static void __attribute__((used)) realInvokeCallback(struct ObjectEventEntry *event,
		UAVObjEvent *msg, void *obj_data, int len);

static void realInvokeCallback(struct ObjectEventEntry *event,
		UAVObjEvent *msg, void *obj_data, int len) {
	event->cb(msg, event->cbInfo.cbCtx, obj_data, len);
}

#if (!defined(SIM_POSIX)) && defined(__arm__)
static void invokeCallback(struct ObjectEventEntry *event, UAVObjEvent *msg,
		void *obj_data, int len) {
	/* If we're inlined, we need to force these to the right parameter
	 * slots.  If we show up in a call they're already there.  This
	 * convinces gcc to do the right thing.
	 */
	register struct ObjectEventEntry *my_event asm("r0") = event;
	register UAVObjEvent *my_msg asm("r1") = msg;
	register void *my_obj_data asm("r2") = obj_data;
	register int my_len asm("r3") = len;

	asm volatile (
		"mov	r4, sp\n\t"		// r4 = old stack pointer
		"mov	sp, %0\n\t"		// set up the new stack
		"bl	realInvokeCallback\n\t"	// run realInvokeCallback--
						// with same args
		"mov	sp, r4\n\t"		// Put back the stack frame

		:
		"+r" (cb_stack),
		"+r" (my_event), "+r" (my_msg), "+r" (my_obj_data),
		"+r" (my_len)		// mentioned as input and output
					// to guarantee that they don't
					// move under us and that the regs
					// are not used after this call
					// which might clobber r0-r3
		: // no pure read-only registers
		: "memory",		// callback may clobber memory,
		"r4", "ip", "lr"	// we clobber r4, ip, and lr
		// And call-clobbered floating point registers
		, "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
		"s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15"
	);
}
#else
#define invokeCallback realInvokeCallback
#endif

/* First argument is deliberately not a pointer to get a copy of msg */
static int32_t pumpOneEvent(UAVObjEvent msg, void *obj_data, int len) {
	// Go through each object and push the event message in the queue (if event is activated for the queue)
	struct ObjectEventEntry *event;
	LL_FOREACH(msg.obj->next_event, event) {
		if (event->eventMask == 0
			|| (event->eventMask & msg.event) != 0) {
			if (event->hasThrottle) {
				// This is a throttled event (triggered with a spacing of at least "interval" ms)
				struct ObjectEventEntryThrottled *throtInfo =
					(struct ObjectEventEntryThrottled *) event;

				uint32_t now = PIOS_Thread_Systime();
				if (throtInfo->due > now){
					continue;
				}

				// Set time for next callback
				throtInfo->due += ((now - throtInfo->due) / throtInfo->interval + 1) * throtInfo->interval;
			}

			// Invoke callback (from event task) if a valid one is registered
			if (event->cb) {
				// invoke callback directly; callbacks must be well behaved
				invokeCallback(event, &msg, obj_data, len);
			} else if (event->cbInfo.queue) {
				// Send to queue if a valid queue is registered
				// will not block
				if (PIOS_Queue_Send(event->cbInfo.queue, &msg, 0) != true) {
					stats.lastQueueErrorID = UAVObjGetID(msg.obj);
					++stats.eventQueueErrors;
				}
			}

		}
	}

	return 0;
}

/**
 * Send a triggered event to all event queues registered on the object.
 */
static int32_t sendEvent(struct UAVOBase * obj, uint16_t instId,
			UAVObjEventType triggered_event,
			void *obj_data, int len)
{
	static uint8_t num_pending = 0;

	static struct PendEvent {
		UAVObjEvent msg;
		void *obj_data;
		int len;
	} pending_events[3];

	/* The logic to spool up callbacks here may be a little confusing.
	 * basically, this relies on the fact that we are in a re-entrant
	 * locked section.  If we get in here and the static variable
	 * in_progress is set, we are entering from a task that itself is
	 * performing a parent callback.
	 *
	 * In other words, while executing a callback it did a uav object
	 * update that will trigger in turn more callbacks.
	 *
	 * To handle this, we have a small buffer to store the pending
	 * callbacks.
	 *
	 * We also make the point of disallowing a callback from generating
	 * the exact same callback.  This is relevant to things like
	 * the session managing object in telemetry.  While it is possible
	 * to do this safely (by "stopping" the quasi-recursion) it seems
	 * better to disallow it.
	 *
	 * However, infinite loops are still possible; callback A can
	 * trigger callback B which triggers callback A.  Don't do that.
	 */

	if (num_pending >= 3) {
		/* Unable to pump event; backlog too long */
		stats.eventCallbackErrors++;
		stats.lastCallbackErrorID = UAVObjGetID(obj);

		return -1;
	}

	static struct UAVOBase *in_progress = NULL;

	if (num_pending) {
		if (in_progress == obj) {
			return -1;	/* We don't fire events
					 * of the same type generated by
					 * an event callback. */
		}
	}

	pending_events[num_pending].msg = (UAVObjEvent) {
		.obj    = obj,
		.event  = triggered_event,
		.instId = instId
	};

	pending_events[num_pending].obj_data = obj_data;
	pending_events[num_pending].len = len;

	num_pending++;

	/* Only enter the section of pumping events if we are the "first event" */
	if (!in_progress) {
		/* While there are events to pump.. */
		while (num_pending) {
			/* Deallocate the top one.. */
			num_pending--;

			/* Mask off events of the same type resulting from
			 * the callback... */
			in_progress = pending_events[num_pending].msg.obj;

			/* And pump the event. */
			pumpOneEvent(pending_events[num_pending].msg,
				pending_events[num_pending].obj_data,
				pending_events[num_pending].len);
		}
	}

	in_progress = NULL;

	return 0;
}

/**
 * Create a new object instance, return the instance info or NULL if failure.
 */
static InstanceHandle createInstance(struct UAVOData * obj, uint16_t instId)
{
	struct UAVOMultiInst *instEntry;

	/* Don't allow more than one instance for single instance objects */
	if (UAVObjIsSingleInstance(&(obj->base))) {
		PIOS_Assert(0);
		return NULL;
	}

	/* Don't create more than the allowed number of instances */
	if (instId >= UAVOBJ_MAX_INSTANCES) {
		return NULL;
	}

	/* Don't allow duplicate instances */
	if (instId < UAVObjGetNumInstances(&(obj->base))) {
		return NULL;
	}

	// Create any missing instances (all instance IDs must be sequential)
	for (uint16_t n = UAVObjGetNumInstances(&(obj->base)); n < instId; ++n) {
		if (createInstance(obj, n) == NULL) {
			return NULL;
		}
	}

	/* Create the actual instance */
	instEntry = (struct UAVOMultiInst *) PIOS_malloc_no_dma(sizeof(struct UAVOMultiInst)+obj->instance_size);
	if (!instEntry)
		return NULL;
	memset(InstanceDataOffset(instEntry), 0, obj->instance_size);
	LL_APPEND(( (struct UAVOMulti*)obj )->instance0.next, instEntry);

	( (struct UAVOMulti*)obj )->num_instances++;

	// Fire event
	UAVObjInstanceUpdated((UAVObjHandle) obj, instId);

	// Done
	if (newUavObjInstanceCB) {
		newUavObjInstanceCB(obj->id, UAVObjGetNumInstances(&obj->base));
	}
	return InstanceDataOffset(instEntry);
}

/**
 * Get the instance information or NULL if the instance does not exist
 */
static InstanceHandle getInstance(struct UAVOData * obj, uint16_t instId)
{
	if (UAVObjIsMetaobject(&obj->base)) {
		/* Metadata Instance */

		if (instId != 0)
			return NULL;

		/* Augment our pointer to reflect the proper type */
		struct UAVOMeta * uavo_meta = (struct UAVOMeta *) obj;
		return (&(uavo_meta->instance0));

	} else if (UAVObjIsSingleInstance(&(obj->base))) {
		/* Single Instance */

		if (instId != 0)
			return NULL;

		/* Augment our pointer to reflect the proper type */
		struct UAVOSingle * uavo_single = (struct UAVOSingle *) obj;
		return (&(uavo_single->instance0));
	} else {
		/* Multi Instance */
		/* Augment our pointer to reflect the proper type */
		struct UAVOMulti * uavo_multi = (struct UAVOMulti *) obj;
		if (instId >= uavo_multi->num_instances)
			return NULL;

		// Look for specified instance ID
		uint16_t instance = 0;
		struct UAVOMultiInst *instEntry;
		LL_FOREACH(&(uavo_multi->instance0), instEntry) {
			if (instance++ == instId) {
				/* Found it */
				return &(instEntry->instance);
			}
		}
		/* Instance was not found */
		return NULL;
	}
}

/**
 * Connect an event queue to the object, if the queue is already connected then the event mask is only updated.
 * \param[in] obj The object handle
 * \param[in] queue The event queue
 * \param[in] cb The event callback
 * \param[in] eventMask The event mask, if EV_MASK_ALL_UPDATES then all events are enabled (e.g. EV_UPDATED | EV_UPDATED_MANUAL)
 * \param[in] interval The interval at which to throttle updates; 0 is unthrottled
 * \return 0 if success or -1 if failure
 */
static int32_t connectObj(UAVObjHandle obj_handle, struct pios_queue *queue,
			UAVObjEventCallback cb, void *cbCtx, uint8_t eventMask,
			uint16_t interval)
{
	if (queue && cb) {
		return -1;
	}

	struct ObjectEventEntry *event;
	struct ObjectEventEntryThrottled *throttled;
	struct UAVOBase *obj;

	// Check that the queue is not already connected, if it is simply update event mask
	obj = (struct UAVOBase *) obj_handle;
	LL_FOREACH(obj->next_event, event) {
		if ((event->cb == cb && event->cbInfo.cbCtx == cbCtx) ||
				((!event->cb) && event->cbInfo.queue == queue)) {
			// Already connected, update event mask and throttling (if possible)
			event->eventMask = eventMask;
			if (event->hasThrottle) {
				if (interval == 0) {
					event->hasThrottle = 0;
				}
				else {
					throttled = (struct ObjectEventEntryThrottled *) event;
					throttled->interval = interval;
				}
				return 0;
			}
			else {
				if (interval == 0) {
					// We don't need to do anything
					return 0;
				}
				else {
					// We are changing the callback from unthrottled to throttled,
					// need to allocate a new event (not ideal, as it leaks memory)
					LL_DELETE(obj->next_event, event);
					break;
				}
			}
		}
	}

	int mallocSize = sizeof(*event);
	struct ObjectEventEntry ** unused = &events_unused;

	if (interval) {
		mallocSize = sizeof(*throttled);
		unused = &events_unused_throttled;
	}

	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
	if (*unused != NULL) {
		// We can re-use the memory of a previously disconnected event
		event = *unused;
		LL_DELETE(*unused, event);
	}
	else {
		event =	(struct ObjectEventEntry *) PIOS_malloc_no_dma(mallocSize);
		if (event == NULL) {
			PIOS_Recursive_Mutex_Unlock(mutex);
			return -1;
		}
	}
	PIOS_Recursive_Mutex_Unlock(mutex);

	memset(event, 0, mallocSize);
	event->cb = cb;

	if (!cb) {
		event->cbInfo.queue = queue;
	} else {
		event->cbInfo.cbCtx = cbCtx;
	}

	event->eventMask = eventMask;
	event->hasThrottle = 0;

	if (interval) {
		event->hasThrottle = 1;
		throttled = (struct ObjectEventEntryThrottled *) event;

		throttled->interval = interval;
		throttled->due = PIOS_Thread_Systime() + randomize_int(throttled->interval);
	}

	LL_APPEND(obj->next_event, event);

	// Done
	return 0;
}

/**
 * Disconnect an event queue from the object
 * \param[in] obj The object handle
 * \param[in] queue The event queue
 * \param[in] cb The event callback
 * \return 0 if success or -1 if failure
 */
static int32_t disconnectObj(UAVObjHandle obj_handle, struct pios_queue *queue,
			UAVObjEventCallback cb, void *cbCtx)
{
	struct ObjectEventEntry *event;
	struct UAVOBase *obj;

	// Find queue and remove it
	obj = (struct UAVOBase *) obj_handle;
	LL_FOREACH(obj->next_event, event) {
		if ((event->cb == cb && event->cbInfo.cbCtx == cbCtx) ||
				((!event->cb) && event->cbInfo.queue == queue)) {
			LL_DELETE(obj->next_event, event);
			// store the unused memory for future reuse
			PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);
			if (event->hasThrottle) {
				LL_APPEND(events_unused_throttled, event);
			}
			else {
				LL_APPEND(events_unused, event);
			}
			PIOS_Recursive_Mutex_Unlock(mutex);
			return 0;
		}
	}

	// If this point is reached the queue was not found
	return -1;
}


/**
 * getEventMask Iterates through the connections and returns the event mask
 * \param[in] obj The object handle
 * \param[in] queue The event queue
 * \return eventMask The event mask, if EV_MASK_ALL then all events are disabled
 */
int32_t getEventMask(UAVObjHandle obj_handle, struct pios_queue *queue)
{
	struct ObjectEventEntry *event;
	struct UAVOBase *obj;

	int32_t eventMask = EV_MASK_ALL;

	// Iterate over the event listeners, looking for the event matching the queue
	obj = (struct UAVOBase *) obj_handle;
	LL_FOREACH(obj->next_event, event) {
		if (event->cbInfo.queue == queue && event->cb == 0) {
			// Already connected, update event mask and return
			eventMask = event->eventMask;
			break;
		}
	}

	// Done
	return eventMask;
}
/**
 * UAVObjCount returns the registered uav objects count
 * \return number of registered uav objects
 */
uint8_t UAVObjCount()
{
	uint8_t count = 0;
	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	// Look for object
	struct UAVOData * tmp_obj;
	LL_FOREACH(uavo_list, tmp_obj) {
		++count;
	}

	// Release lock
	PIOS_Recursive_Mutex_Unlock(mutex);
	return count;
}

/**
 * UAVObjIDByIndex returns the ID of the object with index index
 * \return the ID of the object
 */
uint32_t UAVObjIDByIndex(uint8_t index)
{
	uint8_t count = 0;
	// Get lock
	PIOS_Recursive_Mutex_Lock(mutex, PIOS_MUTEX_TIMEOUT_MAX);

	// Look for object
	struct UAVOData * tmp_obj;
	LL_FOREACH(uavo_list, tmp_obj) {
		if (count == index)
		{
			// Release lock
			PIOS_Recursive_Mutex_Unlock(mutex);
			return tmp_obj->id;
		}
		++count;
	}

	// Release lock
	PIOS_Recursive_Mutex_Unlock(mutex);
	return 0;
}

/**
 * Registers a new UAVO instance created callback
 */
void UAVObjRegisterNewInstanceCB(new_uavo_instance_cb_t callback)
{
	newUavObjInstanceCB = callback;
}
/**
 * @}
 * @}
 */

