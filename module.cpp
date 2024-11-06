#pragma once
#include <assert.h>
#include <vector>
#include <SDKDDKVer.h>
#include <stdio.h>
#include <tchar.h>

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <Windows.h>
#include <string>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define N 4
#define PI 3.1415926
#define MAXITERATIONS 8
#define PRECISION 3.0e-5

using namespace std;

// -----------------------------------------------------------------------------
class Matrix {

public:
	int rows;
	int cols;
	int size;
	vector<vector<double>> data;

public:
	Matrix() {};
	Matrix(int m, int n);
	Matrix(int n);
	virtual ~Matrix();
	double get(int row, int col) const;
	void set(int row, int col, double value);

	bool operator=(const Matrix&);
	friend Matrix operator+(const Matrix&, const Matrix&);
	friend Matrix operator-(const Matrix&, const Matrix&);
	friend Matrix operator*(const Matrix&, const Matrix&);
	friend Matrix operator*(double, const Matrix&);
	friend Matrix operator*(const Matrix&, double);
	friend Matrix operator/(const Matrix&, double);

	double det();
	Matrix t() const;
	Matrix inv();
	void print();
};

Matrix::Matrix(int m, int n) {
	assert(m > 0);
	assert(n > 0);

	rows = m;
	cols = n;
	size = m * n;

	vector<double> tempRow(n, 0);
	for (size_t i = 0; i < m; i++) {
		data.push_back(tempRow);
	}
}

Matrix::Matrix(int n) {
	assert(n > 0);

	rows = n;
	cols = n;
	size = n * n;

	vector<double> tempRow(n, 0);
	for (size_t i = 0; i < n; i++) {
		data.push_back(tempRow);
	}
}

Matrix::~Matrix() {
	//delete[] p;
	//p = NULL;
}

double Matrix::get(int row, int col) const {
	assert(row >= 0 && row < this->rows);
	assert(col >= 0 && col < this->cols);
	return data[row][col];
}

void Matrix::set(int row, int col, double value) {
	assert(row >= 0 && row < rows);
	assert(col >= 0 && col < cols);
	data[row][col] = value;
}

bool Matrix::operator=(const Matrix& m) {
	assert(this->rows == m.rows);
	assert(this->cols == m.cols);

	for (int i = 0; i < this->rows; i++) {
		for (int j = 0; j < this->cols; j++) {
			data[i][j] = m.data[i][j];
		}
	}

	return true;
}

Matrix operator+(const Matrix& m1, const Matrix& m2) {
	assert(m1.rows == m2.rows);
	assert(m1.cols == m2.cols);

	Matrix ret(m1.rows, m1.cols);

	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {
			double val = m1.get(i, j) + m2.get(i, j);
			ret.set(i, j, val);
		}
	}
	return ret;
}

Matrix operator-(const Matrix& m1, const Matrix& m2) {
	assert(m1.rows == m2.rows);
	assert(m1.cols == m2.cols);

	Matrix ret(m1.rows, m1.cols);

	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {
			double val = m1.get(i, j) - m2.get(i, j);
			ret.set(i, j, val);
		}
	}
	return ret;
}

Matrix operator*(const Matrix& m1, const Matrix& m2) {
	assert(m1.size > 0);
	assert(m2.size > 0);
	assert(m1.cols == m2.rows);

	Matrix ret(m1.rows, m2.cols);
	double sum;

	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m2.cols; j++) {
			sum = 0.0;
			for (int k = 0; k < m1.cols; k++) {
				sum += m1.data[i][k] * m2.data[k][j];
			}
			ret.data[i][j] = sum;
		}
	}
	return ret;
}

vector<double> operator*(double n, vector<double> v) {
	vector<double> tempVec(v.size());
	for (size_t i = 0; i < (int)v.size(); i++) {
		tempVec[i] = v[i] * n;
	}

	return tempVec;
}

vector<double> operator/(vector<double> v, double n) {
	vector<double> tempVec(v.size());
	for (size_t i = 0; i < (int)v.size(); i++) {
		tempVec[i] = v[i] / n;
	}

	return tempVec;
}

Matrix operator*(double value, const Matrix& m1) {
	Matrix ret(m1.rows, m1.cols);

	for (int i = 0; i < m1.size; i++) {
		// ret.p[i] = m1.p[i] * value;
		ret.data[i] = value * m1.data[i];
	}

	return ret;
}

Matrix operator*(const Matrix& m1, double value) {
	Matrix ret(m1.rows, m1.cols);

	for (int i = 0; i < m1.size; i++) {
		// ret.p[i] = m1.p[i] * value;
		ret.data[i] = value * m1.data[i];
	}

	return ret;
}

Matrix operator/(const Matrix& m1, double value) {
	Matrix ret(m1.rows, m1.cols);

	for (int i = 0; i < m1.size; i++) {
		ret.data[i] = m1.data[i] / value;
	}

	return ret;
}

double dets(int n, vector<vector<double>> aa) {
	if (n == 1)
		return aa[0][0];

	vector<vector<double>> bb((n - 1), vector<double>(n - 1));
	int move = 0;
	double sum = 0.0;

	for (int a_row = 0; a_row < n; a_row++) {
		for (int b_row = 0; b_row < n - 1; b_row++) {
			move = a_row > b_row ? 0 : 1;
			for (int j = 0; j < n - 1; j++) {
				bb[b_row][j] = aa[b_row + move][j + 1];
			}
		}
		int flag = (a_row % 2 == 0 ? 1 : -1);
		sum += flag * dets(n - 1, bb) * aa[a_row][0];
	}


	return sum;
}

double Matrix::det() {
	assert(rows == cols);
	return dets(rows, data);
}

Matrix Matrix::t() const {

	Matrix ret(cols, rows);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			ret.data[j][i] = data[i][j];
		}
	}

	return ret;
}

Matrix Matrix::inv() {

	double Det = det();
	assert(Det != 0.0);

	Matrix tmp(rows);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < rows; j++) {
			tmp.data[i][j] = data[i][j];
		}
	}
	Matrix ret(rows);
	for (int i = 0; i < rows; i++) {
		ret.set(i, i, 1.0);
	}

	int r = 0;
	double c = 1.0;

	for (int i = 0; i < rows; i++) {
		if (tmp.data[i][i] == 0) {
			for (int j = i; j < rows; j++) {
				if (j != i) {
					if (tmp.data[j][i] != 0) {
						r = j;
						break;
					}
				}
			}

			for (int k = 0; k < rows; k++) {
				vector<vector<double>> t1(1, vector<double>(rows));
				t1[0][k] = tmp.data[i][k];
				tmp.data[i][k] = tmp.data[r][k];
				tmp.data[r][k] = t1[0][k];
				vector<vector<double>> t2(1, vector<double>(rows));
				t2[0][k] = ret.data[i][k];
				ret.data[i][k] = ret.data[r][k];
				ret.data[r][k] = t2[0][k];
			}
		}

		if (i > 0) {
			for (int j = 0; j < i; j++) {
				if (tmp.data[i][j] != 0) {
					c = tmp.data[i][j] / tmp.data[j][j];
					for (int k = 0; k < rows; k++) {
						tmp.data[i][k] = tmp.data[i][k] - c * tmp.data[j][k];
						ret.data[i][k] = ret.data[i][k] - c * ret.data[j][k];
					}
				}
			}
		}
	}

	for (int i = 0; i < rows; i++) {
		c = 1 / tmp.data[i][i];
		for (int j = 0; j < rows; j++) {
			tmp.data[i][j] *= c;
			ret.data[i][j] *= c;
		}
	}

	for (int i = 0; i < rows; i++) {
		for (int j = i + 1; j < rows; j++) {
			c = tmp.data[i][j];
			for (int k = 0; k < rows; k++) {
				tmp.data[i][k] = tmp.data[i][k] - c * tmp.data[j][k];
				ret.data[i][k] = ret.data[i][k] - c * ret.data[j][k];
			}
		}
	}

	return ret;
}

void Matrix::print() {

	cout << endl;
	for (size_t i = 0; i < (int)data.size(); i++) {
		cout << "<" << i << ">";
		for (size_t j = 0; j < (int)data[0].size(); j++) {
			cout << "\t|  " << data[i][j];
		}
		cout << endl;
	}
}
// -----------------------------------------------------------------------------

// elements of exterior orientation
struct EOEO {
	double Xs;
	double Ys;
	double Zs;
	double phi;
	double omega;
	double kappa;
};

bool CheckPrecison(Matrix& Correction);

vector<double> revolving(
	string input,
	double Xs,
	double Ys,
	double Zs,
	double phi,
	double omega,
	double kappa,
	double m,
	double f);

bool CheckPrecison(Matrix& X) {
	bool Boolean;
	Boolean = { fabs(X.data[3][0]) < PRECISION && fabs(X.data[4][0]) < PRECISION && fabs(X.data[5][0]) < PRECISION };
	return Boolean;
}

vector<double> slicing(vector<double>& arr, int X, int Y) {
	auto start = arr.begin() + X;
	auto end = arr.begin() + Y + 1;
	vector<double> result(Y - X + 1);
	copy(start, end, result.begin());
	return result;
};

vector<double> revolving(
	string input,
	double Xs,
	double Ys,
	double Zs,
	double phi,
	double omega,
	double kappa,
	double m,
	double f) {

	auto iss = istringstream(input);
	auto vec = vector<double>(istream_iterator<double>(iss), istream_iterator<double>());

	vector<vector<double>> sd;
	for (int j = 0; j < 20; j++) {
		if (j % 5 == 0) {
			vector<double> a = slicing(vec, j, j + 4);
			sd.push_back(a);
		};
	}

	/*
	double sd[N][5] = {
		{vec[0], vec[1], vec[2], vec[3], vec[4]},
		{vec[5], vec[6], vec[7], vec[8], vec[9]},
		{vec[10], vec[11], vec[12], vec[13], vec[14]},
		{vec[15], vec[16], vec[17], vec[18], vec[19]}
	};
	*/

	// units mm -> m
	for (int i = 0; i < N; i++) {
		double sdi0 = sd[i][0] / 1000;
		double sdi1 = sd[i][1] / 1000;
		Xs += sd[i][2];
		Ys += sd[i][3];
	}

	Xs /= N;
	Ys /= N;
	Zs = m * f;

	double x0(0), y0(0);
	double X0[N] = { 0.0 };
	double Y0[N] = { 0.0 };
	double Z0[N] = { 0.0 };

	Matrix R(3, 3);
	Matrix Correction(6, 1);

	Matrix V_i(2, 1);
	Matrix A_i(2, 6);
	Matrix L_i(2, 1);

	Matrix V(8, 1);
	Matrix A(8, 6);
	Matrix L(8, 1);

	Matrix ATA(6, 6);
	Matrix ATL(6, 1);

	int iCount = 0;
	while (iCount == 0 || !CheckPrecison(Correction)) {
		++iCount;
		if (iCount == MAXITERATIONS) {
			break;
		}

		R.data[0][0] = cos(phi) * cos(kappa) - sin(phi) * sin(omega) * sin(kappa);
		R.data[0][1] = -cos(phi) * sin(kappa) - sin(phi) * sin(omega) * cos(kappa);
		R.data[0][2] = -sin(phi) * cos(omega);
		R.data[1][0] = cos(omega) * sin(kappa);
		R.data[1][1] = cos(omega) * cos(kappa);
		R.data[1][2] = -sin(omega);
		R.data[2][0] = sin(phi) * cos(kappa) + cos(phi) * sin(omega) * sin(kappa);
		R.data[2][1] = -sin(phi) * sin(kappa) + cos(phi) * sin(omega) * cos(kappa);
		R.data[2][2] = cos(phi) * cos(omega);

		for (int i = 0; i < N; i++) {
			Z0[i] = R.data[0][2] * (sd[i][2] - Xs) + R.data[1][2] * (sd[i][3] - Ys) + R.data[2][2] * (sd[i][4] - Zs);
			X0[i] = x0 - f * (R.data[0][0] * (sd[i][2] - Xs) + R.data[1][0] * (sd[i][3] - Ys) + R.data[2][0] * (sd[i][4] - Zs)) / Z0[i];
			Y0[i] = y0 - f * (R.data[0][1] * (sd[i][2] - Xs) + R.data[1][1] * (sd[i][3] - Ys) + R.data[2][1] * (sd[i][4] - Zs)) / Z0[i];

			A_i.data[0][0] = (R.data[0][0] * f + R.data[0][2] * (sd[i][0] - x0)) / Z0[i];
			A_i.data[0][1] = (R.data[1][0] * f + R.data[1][2] * (sd[i][0] - x0)) / Z0[i];
			A_i.data[0][2] = (R.data[2][0] * f + R.data[2][2] * (sd[i][0] - x0)) / Z0[i];
			A_i.data[0][3] = (sd[i][1] - y0) * sin(omega) - ((sd[i][0] - x0) * ((sd[i][0] - x0) * cos(kappa) -
				(sd[i][1] - y0) * sin(kappa)) / f + f * cos(kappa)) * cos(omega);
			A_i.data[0][4] = -f * sin(kappa) - (sd[i][0] - x0) * ((sd[i][0] - x0) * sin(kappa) + (sd[i][1] - y0) * cos(kappa)) / f;
			A_i.data[0][5] = sd[i][1] - y0;

			A_i.data[1][0] = (R.data[0][1] * f + R.data[0][2] * (sd[i][1] - y0)) / Z0[i];
			A_i.data[1][1] = (R.data[1][1] * f + R.data[1][2] * (sd[i][1] - y0)) / Z0[i];
			A_i.data[1][2] = (R.data[2][1] * f + R.data[2][2] * (sd[i][1] - y0)) / Z0[i];
			A_i.data[1][3] = -(sd[i][0] - x0) * sin(omega) - ((sd[i][1] - y0) * ((sd[i][0] - x0) * cos(kappa) -
				(sd[i][1] - y0) * sin(kappa)) / f - f * sin(kappa)) * cos(omega);
			A_i.data[1][4] = -f * cos(kappa) - (sd[i][1] - y0) * ((sd[i][0] - x0) * sin(kappa) + (sd[i][1] - y0) * cos(kappa)) / f;
			A_i.data[1][5] = -(sd[i][0] - x0);

			L_i.data[0][0] = sd[i][0] - X0[i];
			L_i.data[1][0] = sd[i][1] - Y0[i];

			for (int j = 0; j < 2; j++) {
				L.data[2 * i + j][0] = L_i.data[j][0];
				for (int k = 0; k < 6; k++) {
					A.data[2 * i + j][k] = A_i.data[j][k];
				}
			}
		}

		ATA = A.t() * A;
		ATL = A.t() * L;

		Correction = ATA.inv() * ATL;

		Xs = Xs + Correction.data[0][0];
		Ys = Ys + Correction.data[1][0];
		Zs = Zs + Correction.data[2][0];
		phi = phi + Correction.data[3][0];
		omega = omega + Correction.data[4][0];
		kappa = kappa + Correction.data[5][0];
	}

	EOEO eoeo;
	eoeo.Xs = Xs;
	eoeo.Ys = Ys;
	eoeo.Zs = Zs;
	eoeo.phi = phi;
	eoeo.omega = omega;
	eoeo.kappa = kappa;

	vector<vector<double>> Q(6, vector<double>(1));
	for (int i = 0; i < 6; i++) {
		Q[i][0] = ATA.data[i][i];
	}

	double m0(0);
	double vv(0);

	V = A * Correction - L;

	for (int i = 0; i < 8; i++) {
		vv = vv + V.data[i][0] * V.data[i][0];
	}
	m0 = sqrt(vv / (2 * N - 6));

	double M[6] = { 0.0 };

	for (int i = 0; i < 6; i++) {
		double Qi = Q[i][0];
		M[i] = m0 * sqrt(Qi);
		if (i > 2) {
			M[i] = M[i] * 180 * 3600 / PI;
		}
	}

	vector<double> result = { Xs, Ys, Zs, phi, omega, kappa };
	return result;
	
	/*
	cout << result[0] << endl;
	cout << result[1] << endl;
	cout << result[2] << endl;
	cout << result[3] << endl;
	cout << result[4] << endl;
	cout << result[5] << endl;
	*/
}

namespace py = pybind11;

PYBIND11_MODULE(leastsquares, m) {
	m.def("leastsq_cpp", &revolving, R"pbdoc(
        Optimize elements of exterior orientation.
    )pbdoc");

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}