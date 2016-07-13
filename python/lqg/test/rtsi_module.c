#include <Python.h>
#include "math.h"

#define NPY_NO_DEPRECATED_API 7
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

#include <pios_heap.h>
#include <rate_torque_si.h>

int not_doublevector(PyArrayObject *vec)
{
	if (PyArray_TYPE(vec) != NPY_DOUBLE) {
		PyErr_SetString(PyExc_ValueError,
              "Vector is not a float or double vector.");
		return 1;
	}
	if (PyArray_NDIM(vec) != 1)  {
		PyErr_Format(PyExc_ValueError,
              "Vector is not a 1 dimensional vector (%d).", PyArray_NDIM(vec));
		return 1;  
	}
	return 0;
}

/**
 * parseFloatVec3(vec_in, vec_out)
 *
 * @param[in] vec_in the python array to extract elements from
 * @param[out] vec_out float array of the numbers
 * @return true if successful, false if not
 *
 * Convert a python array type to a 3 element float
 * vector.
 */
static bool parseFloatVecN(PyArrayObject *vec_in, float *vec_out, int N)
{
	/* verify it is a valid vector */
	if (not_doublevector(vec_in))
		return false;

	if (PyArray_DIM(vec_in,0) != N) {
		PyErr_Format(PyExc_ValueError, "Vector length incorrect. Received %ld and expected %d", PyArray_DIM(vec_in,0), N);
		return false;
	}

	NpyIter *iter;
	NpyIter_IterNextFunc *iternext;

	/*  create the iterators */
	iter = NpyIter_New(vec_in, NPY_ITER_READONLY, NPY_KEEPORDER,
							 NPY_NO_CASTING, NULL);
	if (iter == NULL)
		goto fail;

	iternext = NpyIter_GetIterNext(iter, NULL);
	if (iternext == NULL) {
		NpyIter_Deallocate(iter);
		goto fail;
	}

	double ** dataptr = (double **) NpyIter_GetDataPtrArray(iter);

	/*  iterate over the arrays */
	int i = 0;
	do {
		vec_out[i++] = **dataptr;
	} while(iternext(iter) && (i < N));

	NpyIter_Deallocate(iter);

	return true;

fail:
	fprintf(stderr, "Parse fail\r\n");
	return false;
}

/**
 * parseFloatVec3(vec_in, vec_out)
 *
 * @param[in] vec_in the python array to extract elements from
 * @param[out] vec_out float array of the numbers
 * @return true if successful, false if not
 *
 * Convert a python array type to a 3 element float
 * vector.
 */
static bool parseFloatVec3(PyArrayObject *vec_in, float *vec_out)
{
  return parseFloatVecN(vec_in, vec_out, 3);
}

uintptr_t rtsi_handle;


/**
 * pack_state put the state information into an array
 */
static PyObject*
pack_state(PyObject* self)
{
	const int N = 14;
	int nd = 1;
	int dims[1];
	dims[0] = N;

	PyArrayObject *state;
	state = (PyArrayObject*) PyArray_FromDims(nd, dims, NPY_DOUBLE);
	double *s = (double *) PyArray_DATA(state);

	float f[4];
	rtsi_get_rates(rtsi_handle, f);    // three elements
	s[0] = f[0];
	s[3] = f[1];
	s[6] = f[2];
	rtsi_get_torque(rtsi_handle, f);  // three elements
	s[1] = f[0];
	s[4] = f[1];
	s[7] = f[2];
	rtsi_get_bias(rtsi_handle, f);    // three elements
	s[2] = f[0];
	s[5] = f[1];
	s[8] = f[2];
	rtsi_get_gains(rtsi_handle, f);    // four elements
	s[9] = f[0];
	s[10] = f[1];
	s[11] = f[2];
	s[12] = f[3];
	rtsi_get_tau(rtsi_handle, f);     // one element
	s[13] = f[0];

	return Py_BuildValue("O", state);
}

static PyObject*
init(PyObject* self, PyObject* args)
{
	if (rtsi_handle == 0)
		rtsi_alloc(&rtsi_handle);

	rtsi_init(rtsi_handle);
	return pack_state(self);
}

static PyObject*
predict(PyObject* self, PyObject* args)
{
	PyArrayObject *vec_gyro, *vec_control;
	float gyro_data[3], control_data[3];
	float dT;

	if (!PyArg_ParseTuple(args, "O!O!f", &PyArray_Type, &vec_gyro,
				   &PyArray_Type, &vec_control, &dT))  return NULL;
	if (NULL == vec_gyro)  return NULL;
	if (NULL == vec_control)  return NULL;

	if (!parseFloatVec3(vec_gyro, gyro_data))
		return NULL;
	if (!parseFloatVec3(vec_control, control_data))
		return NULL;

	rtsi_predict(rtsi_handle,control_data,gyro_data,dT);

	return pack_state(self);
}

static PyMethodDef RtsiMethods[] =
{
	{"init", init, METH_VARARGS, "Reset KF state."},
	{"predict", predict, METH_VARARGS, "Advance state 1 time step."},
	{NULL, NULL, 0, NULL}
};
 
PyMODINIT_FUNC
initrtsi(void)
{
	(void) Py_InitModule("rtsi", RtsiMethods);
	import_array();
	init(NULL, NULL);
}
