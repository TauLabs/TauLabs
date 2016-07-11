#include <Python.h>
#include "math.h"

#define NPY_NO_DEPRECATED_API 7
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

#include <pios_heap.h>
#include <rate_torque_kf.h>
#include <rate_torque_kf_optimize.h>

//! State variable
uintptr_t rtkf_handle;

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

/**
 * pack_state put the state information into an array
 */
static PyObject*
pack_state(PyObject* self)
{
	const int N = 9;
	int nd = 1;
	int dims[1];
	dims[0] = N;

	float rate[3];
	float torque[3];
	float bias[3];

	PyArrayObject *state;
	state = (PyArrayObject*) PyArray_FromDims(nd, dims, NPY_DOUBLE);
	double *s = (double *) PyArray_DATA(state);

	rtkf_get_rate(rtkf_handle, rate);
	rtkf_get_torque(rtkf_handle, torque);
	rtkf_get_bias(rtkf_handle, bias);

	for (int i = 0; i < 3; i++) {
		s[i] = rate[i];
		s[i+3] = torque[i];
		s[i+6] = bias[i];
	}

	return Py_BuildValue("O", state);
}

static PyObject*
init(PyObject* self, PyObject* args)
{
	if (rtkf_handle == 0)
		rtkf_alloc(&rtkf_handle);

	rtkf_init(rtkf_handle);
	return pack_state(self);
}

static PyObject*
advance(PyObject* self, PyObject* args)
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

	rtkf_predict(rtkf_handle, 0.5f, control_data, gyro_data,dT);

	return pack_state(self);
}

static PyObject*
configure(PyObject* self, PyObject* args, PyObject *kwarg)
{
	static char *kwlist[] = {"gain", "tau", NULL};

	PyArrayObject *gain_var = NULL;
	float tau_var = NAN;

	if (!PyArg_ParseTupleAndKeywords(args, kwarg, "|Of", kwlist,
		 &gain_var, &tau_var)) {
		return NULL;
	}

	if (gain_var) {
		float gain_new[4];
		if (!parseFloatVecN(gain_var, gain_new, 4))
			return NULL;
		rtkf_set_gains(rtkf_handle, gain_new);
		//printf("Setting gains\r\n");
	}

	if (!isnan(tau_var)) {
		rtkf_set_tau(rtkf_handle, tau_var);
		//printf("Setting tau\r\n");
	}

	return Py_None;
}

static PyObject*
optimize(PyObject* self, PyObject* args, PyObject *kwarg)
{
	float Ts_var = NAN;
	float tau_var = NAN;
	float b1_var = NAN;
	float b2_var = NAN;

	if (!PyArg_ParseTuple(args, "ffff", &Ts_var, &tau_var, &b1_var, &b2_var)) {
		return NULL;
	}

	fprintf(stdout, "%f, %f, %f, %f\r\n", Ts_var, tau_var, b1_var, b2_var);

	rtkfo_init(Ts_var);
	rtkfo_set_tau(tau_var);

	const float gain[4] = {b1_var, b1_var, b1_var, b2_var};
	rtkfo_set_gains(gain);
	
	float process[3] = {1e0, 1e-5f, 1e-7f};   // Pretty robust defaults
	float gyro_noise = 1000.0f;                // Could take greatest value from SI
	rtkfo_set_noise(process, gyro_noise);

	rtkfo_solver();

	const int nd = 2;
	int dims[nd] = {3,3};
	PyArrayObject *calculated_x;
	calculated_x = (PyArrayObject*) PyArray_FromDims(nd, dims, NPY_DOUBLE);
	double *s = (double *) PyArray_DATA(calculated_x);

	float f[3];

	rtkfo_get_roll_gain(f);
	s[0] = f[0];
	s[1] = f[1];
	s[2] = f[2];

	rtkfo_get_roll_gain(f);
	s[3] = f[0];
	s[4] = f[1];
	s[5] = f[2];

	rtkfo_get_yaw_gain(f);
	s[6] = f[0];
	s[7] = f[1];
	s[8] = f[1];

	//fprintf(stdout, "Gains: %f, %f, %f\r\n", f[0], f[1], f[2]);

	return Py_BuildValue("O", calculated_x);
}

static PyMethodDef RtkfMethods[] =
{
	{"init", init, METH_VARARGS, "Reset KF state."},
	{"advance", advance, METH_VARARGS, "Advance state 1 time step."},
	{"configure", (PyCFunction)configure, METH_VARARGS|METH_KEYWORDS, "Configure EKF parameters."},
	{"optimize", (PyCFunction)optimize, METH_VARARGS|METH_KEYWORDS, "Compute Kalman gains."},
	{NULL, NULL, 0, NULL}
};
 
PyMODINIT_FUNC
initrtkf(void)
{
	(void) Py_InitModule("rtkf", RtkfMethods);
	import_array();
	init(NULL, NULL);
}
