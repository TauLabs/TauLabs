#include <Python.h>
#include "math.h"

#define NPY_NO_DEPRECATED_API 7
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

#include <pios_heap.h>
#include <rate_torque_lqr_optimize.h>

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
 * pack_state put the state information into an array
 */
static PyObject*
pack_state(PyObject* self)
{
	const int N1 = 6;
	const int N2 = 3;
	const int nd = 2;
	int dims[nd];
	dims[0] = N1;
	dims[1] = N2;

	PyArrayObject *state;
	state = (PyArrayObject*) PyArray_FromDims(nd, dims, NPY_DOUBLE);
	double *s = (double *) PyArray_DATA(state);

	float f[N1*N2];

	rtlqro_get_roll_rate_gain(f);
	rtlqro_get_pitch_rate_gain(f+3);
	rtlqro_get_yaw_rate_gain(f+6);
	f[2] = 0;
	f[5] = 0;
	f[8] = 0;
	rtlqro_get_roll_attitude_gain(f+9);
	rtlqro_get_pitch_attitude_gain(f+12);
	rtlqro_get_yaw_attitude_gain(f+15);

	for (int i = 0; i < (N1*N2); i++)
		s[i] = f[i];

	return Py_BuildValue("O", state);
}

float angle_cost = 1e-2f;
float angle_rate_cost = 1e-4f;
float angle_torque_cost = 1e-4f;
float rate_cost = 1e-2f;
float rate_torque_cost = 1e-4f;
float yaw_rate_cost = 1e-2f;
float yaw_rate_torque_cost = 1e-4;

static PyObject*
init(PyObject* self, PyObject* args)
{
	rtlqro_init(1.0f/400.0f);

	rtlqro_set_costs(angle_cost,
		angle_rate_cost,
		angle_torque_cost,
		rate_cost,
		rate_torque_cost,
		yaw_rate_cost,
		yaw_rate_torque_cost);

	return Py_None;
}

static PyObject*
solve(PyObject* self, PyObject* args)
{
	rtlqro_set_costs(angle_cost,
		angle_rate_cost,
		angle_torque_cost,
		rate_cost,
		rate_torque_cost,
		yaw_rate_cost,
		yaw_rate_torque_cost);

	rtlqro_solver();

	return pack_state(self);
}

static PyObject*
configure(PyObject* self, PyObject* args, PyObject *kwarg)
{
	static char *kwlist[] = {"gains", "tau", "angle_cost", "angle_rate_cost", "angle_torque_cost", "rate_cost", "rate_torque_cost", NULL};

	PyArrayObject *gain_var = NULL;
	float tau_var = NAN;
	float angle_cost_var = NAN;
	float angle_rate_cost_var = NAN;
	float angle_torque_cost_var = NAN;
	float rate_cost_var = NAN;
	float rate_torque_cost_var = NAN;

	if (!PyArg_ParseTupleAndKeywords(args, kwarg, "|Offffff", kwlist,
		 &gain_var, &tau_var, &angle_cost_var, &angle_rate_cost_var, &angle_torque_cost_var, &rate_cost_var, &rate_torque_cost_var)) {
		return NULL;
	}

	if (gain_var) {
		float gain_new[4];
		if (!parseFloatVecN(gain_var, gain_new, 4))
			return NULL;
		rtlqro_set_gains(gain_new);
	}

	if (!isnan(tau_var)) {
		rtlqro_set_tau(tau_var);
	}

	if (!isnan(angle_cost_var)) {
		angle_cost = angle_cost_var;
	}

	if (!isnan(angle_rate_cost_var)) {
		angle_rate_cost = angle_rate_cost_var;
	}

	if (!isnan(angle_torque_cost_var)) {
		angle_torque_cost = angle_torque_cost_var;
	}

	if (!isnan(rate_cost_var)) {
		rate_cost = rate_cost_var;
	}

	if (!isnan(rate_torque_cost_var)) {
		rate_torque_cost = rate_torque_cost_var;
	}

	return Py_None;
}

static PyMethodDef RtlqrMethods[] =
{
	{"init", init, METH_VARARGS, "Initialize LQR solver."},
	{"solve", solve, METH_VARARGS, "Advance state 1 time step."},
	{"configure", (PyCFunction)configure, METH_VARARGS|METH_KEYWORDS, "Configure EKF parameters."},
	{NULL, NULL, 0, NULL}
};
 
PyMODINIT_FUNC
initrtlqr(void)
{
	(void) Py_InitModule("rtlqr", RtlqrMethods);
	import_array();
	init(NULL, NULL);
}
