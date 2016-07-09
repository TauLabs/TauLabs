#include <Python.h>
#include "math.h"

#define NPY_NO_DEPRECATED_API 7
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

#include <pios_heap.h>

#include <iostream>
#include "dare.h"

using namespace std;

int not_doublevector(PyArrayObject *vec)
{
	if (PyArray_TYPE(vec) != NPY_DOUBLE) {
		PyErr_SetString(PyExc_ValueError,
              "Vector is not a float or double vector.");
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

	if (PyArray_NDIM(vec_in) != 1)  {
		PyErr_Format(PyExc_ValueError, "Vector is not a 1 dimensional vector (%d).", PyArray_NDIM(vec_in));
		return 1;  
	}

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
		// TODO: throw exception
		return false;

	iternext = NpyIter_GetIterNext(iter, NULL);
	if (iternext == NULL) {
		NpyIter_Deallocate(iter);
		// TODO: throw exception
		return false;
	}

	double ** dataptr = (double **) NpyIter_GetDataPtrArray(iter);

	/*  iterate over the arrays */
	int i = 0;
	do {
		vec_out[i++] = **dataptr;
	} while(iternext(iter) && (i < N));

	NpyIter_Deallocate(iter);

	return true;
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
 * parseFloatArray3x3(PyArrayObject *arr_in, float *vec_out)
 *
 * @param[in] arr_in the python array to extract elements from
 * @param[out] vec_out float array of the numbers
 * @return true if successful, false if not
 *
 * Convert a python 3x3 array type to a 9 element float
 * vector.
 */
static bool parseFloatArray3x3(PyArrayObject *arr_in, float *vec_out)
{
	/* verify it is a valid vector */
	if (not_doublevector(arr_in))
		return false;

	const int N = 3;

	if (PyArray_NDIM(arr_in) != 2) {
		PyErr_Format(PyExc_ValueError, "Array is not 2 dimensional (%d).", PyArray_NDIM(arr_in));
		return false;
	}

	if (PyArray_DIM(arr_in,0) != N || PyArray_DIM(arr_in,1) != N) {
		PyErr_Format(PyExc_ValueError, "Array should be 3x3");
		return false;
	}

	NpyIter *iter;
	NpyIter_IterNextFunc *iternext;

	/*  create the iterators */
	iter = NpyIter_New(arr_in, NPY_ITER_READONLY, NPY_KEEPORDER,
							 NPY_NO_CASTING, NULL);
	if (iter == NULL)
		// TODO: throw exception
		return false;

	iternext = NpyIter_GetIterNext(iter, NULL);
	if (iternext == NULL) {
		NpyIter_Deallocate(iter);
		// TODO: throw exception
		return false;
	}

	double ** dataptr = (double **) NpyIter_GetDataPtrArray(iter);

	/*  iterate over the arrays */
	int i = 0;
	do {
		vec_out[i++] = **dataptr;
	} while(iternext(iter) && (i < (N * N)));

	NpyIter_Deallocate(iter);

	return true;
}

/**
 * @brief solve_dare
 * @param[in] A - 3x3 numpy array
 * @param[in] B - 3x1 numpy vector
 * @param[in] Q - 3x3 numpy array
 * @param[in] R - scalar
 * @returns solution to DARE equation
 */
static PyObject*
solve_dare(PyObject* self, PyObject* args)
{
	PyArrayObject *arr_A, *arr_B, *arr_Q;
	float vec_A[9], vec_B[3], vec_Q[9], scal_R;

	if (!PyArg_ParseTuple(args, "O!O!O!f", &PyArray_Type, &arr_A,
				   &PyArray_Type, &arr_B, &PyArray_Type, &arr_Q, &scal_R))  return NULL;
	if (NULL == arr_A)  return NULL;
	if (NULL == arr_B)  return NULL;
	if (NULL == arr_Q)  return NULL;

	// Parse data from arrays
	if (!parseFloatArray3x3(arr_A, vec_A))
		return NULL;
	if (!parseFloatVec3(arr_B, vec_B))
		return NULL;
	if (!parseFloatArray3x3(arr_Q, vec_Q))
		return NULL;

	MXX A = MXX::Constant(0.0f);
	MXU B = MXU::Constant(0.0f);
	MXX Q = MXX::Identity();
	MUU R = MUU::Identity();

	A(0,0) = vec_A[0];
	A(0,1) = vec_A[1];
	A(0,2) = vec_A[2];
	A(1,0) = vec_A[3];
	A(1,1) = vec_A[4];
	A(1,2) = vec_A[5];
	A(2,0) = vec_A[6];
	A(2,1) = vec_A[7];
	A(2,2) = vec_A[8];

	B(0,0) = vec_B[0];
	B(1,0) = vec_B[1];
	B(2,0) = vec_B[2];

	Q(0,0) = vec_Q[0];
	Q(0,1) = vec_Q[1];
	Q(0,2) = vec_Q[2];
	Q(1,0) = vec_Q[3];
	Q(1,1) = vec_Q[4];
	Q(1,2) = vec_Q[5];
	Q(2,0) = vec_Q[6];
	Q(2,1) = vec_Q[7];
	Q(2,2) = vec_Q[8];

	R(0,0) = scal_R;

	const int nd = 2;
	int dims[nd] = {3,3};
	PyArrayObject *calculated_x;
	calculated_x = (PyArrayObject*) PyArray_FromDims(nd, dims, NPY_DOUBLE);
	double *s = (double *) PyArray_DATA(calculated_x);

	MXX X = dare_solve(A, B, Q, R);

	for (int i = 0; i < dims[0]; i++)
		for (int j = 0; j < dims[1]; j++)
		s[j+3*i] = X(i,j);

	return Py_BuildValue("O", calculated_x);
}

/**
 * @brief solve_dare
 * @param[in] A - 3x3 numpy array
 * @param[in] B - 3x1 numpy vector
 * @param[in] Q - 3x3 numpy array
 * @param[in] R - scalar
 * @returns solution to DARE equation
 */
static PyObject*
solve_kalman(PyObject* self, PyObject* args)
{
	PyArrayObject *arr_A, *arr_B, *arr_Q;
	float vec_A[9], vec_B[3], vec_Q[9], scal_R;

	if (!PyArg_ParseTuple(args, "O!O!O!f", &PyArray_Type, &arr_A,
				   &PyArray_Type, &arr_B, &PyArray_Type, &arr_Q, &scal_R))  return NULL;
	if (NULL == arr_A)  return NULL;
	if (NULL == arr_B)  return NULL;
	if (NULL == arr_Q)  return NULL;

	// Parse data from arrays
	if (!parseFloatArray3x3(arr_A, vec_A))
		return NULL;
	if (!parseFloatVec3(arr_B, vec_B))
		return NULL;
	if (!parseFloatArray3x3(arr_Q, vec_Q))
		return NULL;

	MXX A = MXX::Constant(0.0f);
	MXU B = MXU::Constant(0.0f);
	MXX Q = MXX::Identity();
	MUU R = MUU::Identity();

	A(0,0) = vec_A[0];
	A(0,1) = vec_A[1];
	A(0,2) = vec_A[2];
	A(1,0) = vec_A[3];
	A(1,1) = vec_A[4];
	A(1,2) = vec_A[5];
	A(2,0) = vec_A[6];
	A(2,1) = vec_A[7];
	A(2,2) = vec_A[8];

	B(0,0) = vec_B[0];
	B(1,0) = vec_B[1];
	B(2,0) = vec_B[2];

	Q(0,0) = vec_Q[0];
	Q(0,1) = vec_Q[1];
	Q(0,2) = vec_Q[2];
	Q(1,0) = vec_Q[3];
	Q(1,1) = vec_Q[4];
	Q(1,2) = vec_Q[5];
	Q(2,0) = vec_Q[6];
	Q(2,1) = vec_Q[7];
	Q(2,2) = vec_Q[8];

	R(0,0) = scal_R;

	const int nd = 1;
	int dims[nd] = {3};
	PyArrayObject *calculated_x;
	calculated_x = (PyArrayObject*) PyArray_FromDims(nd, dims, NPY_DOUBLE);
	double *s = (double *) PyArray_DATA(calculated_x);

	MXU X = kalman_gain_solve(A, B, Q, R);

	for (int i = 0; i < dims[0]; i++)
		s[i] = X(i,0);

	return Py_BuildValue("O", calculated_x);
}

/**
 * @brief solve_dare
 * @param[in] A - 3x3 numpy array
 * @param[in] B - 3x1 numpy vector
 * @param[in] Q - 3x3 numpy array
 * @param[in] R - scalar
 * @returns solution to DARE equation
 */
static PyObject*
solve_lqr(PyObject* self, PyObject* args)
{
	PyArrayObject *arr_A, *arr_B, *arr_Q;
	float vec_A[9], vec_B[3], vec_Q[9], scal_R;

	if (!PyArg_ParseTuple(args, "O!O!O!f", &PyArray_Type, &arr_A,
				   &PyArray_Type, &arr_B, &PyArray_Type, &arr_Q, &scal_R))  return NULL;
	if (NULL == arr_A)  return NULL;
	if (NULL == arr_B)  return NULL;
	if (NULL == arr_Q)  return NULL;

	// Parse data from arrays
	if (!parseFloatArray3x3(arr_A, vec_A))
		return NULL;
	if (!parseFloatVec3(arr_B, vec_B))
		return NULL;
	if (!parseFloatArray3x3(arr_Q, vec_Q))
		return NULL;

	MXX A = MXX::Constant(0.0f);
	MXU B = MXU::Constant(0.0f);
	MXX Q = MXX::Identity();
	MUU R = MUU::Identity();

	A(0,0) = vec_A[0];
	A(0,1) = vec_A[1];
	A(0,2) = vec_A[2];
	A(1,0) = vec_A[3];
	A(1,1) = vec_A[4];
	A(1,2) = vec_A[5];
	A(2,0) = vec_A[6];
	A(2,1) = vec_A[7];
	A(2,2) = vec_A[8];

	B(0,0) = vec_B[0];
	B(1,0) = vec_B[1];
	B(2,0) = vec_B[2];

	Q(0,0) = vec_Q[0];
	Q(0,1) = vec_Q[1];
	Q(0,2) = vec_Q[2];
	Q(1,0) = vec_Q[3];
	Q(1,1) = vec_Q[4];
	Q(1,2) = vec_Q[5];
	Q(2,0) = vec_Q[6];
	Q(2,1) = vec_Q[7];
	Q(2,2) = vec_Q[8];

	R(0,0) = scal_R;

	const int nd = 2;
	int dims[nd] = {1,3};
	PyArrayObject *calculated_x;
	calculated_x = (PyArrayObject*) PyArray_FromDims(nd, dims, NPY_DOUBLE);
	double *s = (double *) PyArray_DATA(calculated_x);

	MUX X = lqr_gain_solve(A, B, Q, R);

	for (int i = 0; i < dims[1]; i++)
		s[i] = X(0,i);

	/*cout << "A: " << A << endl;
	cout << "B: " << B << endl;
	cout << "Q: " << Q << endl;
	cout << "R: " << R << endl;
	cout << "X: " << X << endl;*/

	return Py_BuildValue("O", calculated_x);
}

static PyMethodDef DareMethods[] =
{
	{"dare", solve_dare, METH_VARARGS, "Solve ricatti equation."},
	{"kalman", solve_kalman, METH_VARARGS, "Calculate Kalman gains."},
	{"lqr", solve_lqr, METH_VARARGS, "Calculate LQR gains."},
	{NULL, NULL, 0, NULL}
};
 
PyMODINIT_FUNC
initdare(void)
{
	(void) Py_InitModule("dare", DareMethods);
	import_array();
}
