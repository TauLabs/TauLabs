#include <Python.h>
#include "math.h"
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

#include <insgps.h>

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
    } while(iternext(iter) && (i < 3));

	NpyIter_Deallocate(iter);

	return true;

fail:
	return false;
}

/**
 * pack_state put the state information into an array
 */
static PyObject*
pack_state(PyObject* self)
{
	float pos[3], vel[3], q[4], gyro_bias[3], accel_bias[3];        
	INSGetState(pos, vel, q, gyro_bias, accel_bias);

	const int N = 16;
	int nd = 1;
  	int dims[1];
  	dims[0] = N;

  	PyArrayObject *state;
  	state = (PyArrayObject*) PyArray_FromDims(nd, dims, NPY_DOUBLE);
  	double *s = (double *) state->data;

  	s[0] = pos[0];
  	s[1] = pos[1];
  	s[2] = pos[2];
  	s[3] = vel[0];
  	s[4] = vel[1];
  	s[5] = vel[2];
  	s[6] = q[0];
  	s[7] = q[1];
  	s[8] = q[2];
  	s[9] = q[3];
  	s[10] = gyro_bias[0];
  	s[11] = gyro_bias[1];
  	s[12] = gyro_bias[2];
  	s[13] = accel_bias[0];
  	s[14] = accel_bias[1];
  	s[15] = accel_bias[2];

	return Py_BuildValue("O", state);
}

/**
 * prediction - perform a state prediction step
 * @params[in] self
 * @params[in] args
 *  - gyro
 *  - accel
 *  - dT
 * @return state
 */
static PyObject*
prediction(PyObject* self, PyObject* args)
{
	PyArrayObject *vec_gyro, *vec_accel;
	float gyro_data[3], accel_data[3];
	float dT;

    if (!PyArg_ParseTuple(args, "O!O!f", &PyArray_Type, &vec_gyro,
    	           &PyArray_Type, &vec_accel, &dT))  return NULL;
	if (NULL == vec_gyro)  return NULL;
	if (NULL == vec_accel)  return NULL;

	parseFloatVec3(vec_gyro, gyro_data);
	parseFloatVec3(vec_accel, accel_data);

	printf("dT: %f [%f %f %f] [%f %f %f]\n", dT, gyro_data[0], gyro_data[1], gyro_data[2], accel_data[0], accel_data[1], accel_data[2]);

	INSStatePrediction(gyro_data, accel_data, dT);
	INSCovariancePrediction(dT);

	return pack_state(self);
	
fail:
	return NULL;
}
 
static PyObject*
full_correction(PyObject* self, PyObject* args)
{
	PyArrayObject *vec_pos, *vec_vel, *vec_mag;
	float pos_data[3], vel_data[3], mag_data[3];
	float baro_data;

    if (!PyArg_ParseTuple(args, "O!O!O!f", &PyArray_Type, &vec_pos,
    	           &PyArray_Type, &vec_vel, &PyArray_Type, &vec_mag,
    	           &baro_data))  return NULL;
	if (NULL == vec_pos)  return NULL;
	if (NULL == vec_vel)  return NULL;
	if (NULL == vec_mag)  return NULL;	

	parseFloatVec3(vec_pos, pos_data);
	parseFloatVec3(vec_vel, vel_data);
	parseFloatVec3(vec_mag, mag_data);

    INSCorrection(mag_data, pos_data, vel_data, baro_data, FULL_SENSORS);

	return pack_state(self);
}

static PyMethodDef InsMethods[] =
{
     {"prediction", prediction, METH_VARARGS, "Advance state 1 time step."},
     {NULL, NULL, 0, NULL}
};
 
PyMODINIT_FUNC
initins(void)
{
     (void) Py_InitModule("ins", InsMethods);
     import_array();
     
     INSGPSInit();
}
