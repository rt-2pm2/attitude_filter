#pragma once

#include "math.hpp"

const size_t Nx = 10;
const size_t Nq = 4;
const size_t Nbg = 3;
const size_t Nbm = 3;
const size_t Nz = 6;
const size_t Nu = 3;


const size_t N_dyn {3};
const size_t N_meas {2};

using namespace matrix;

/**
 * Class implementing the attitude extended kalman filter
 *
 * The model is in the form:
 * 	x_ = f(x_, u, w)
 * 	z_ = g(x_, u, v)
 *
 * The linearized part requires 4 matrices:
 *	F = df/dx
 *	W = df/dw
 *	
 *	H = dg/dx
 *	V = dg/dv
 *
 *	x_ = Fx_ + Ww
 *	z_ = Hx_ + Vv 
 */
class AttEkf {
	float Ts;

	Vector<float, N_dyn> proc_stds;
	Vector<float, N_meas> meas_stds;

	Vector<float, 3> g;
	Vector<float, 3> m;

	Vector<float, Nx> x_;
	Vector<float, Nx> x_hat;
	SquareMatrix<float, Nx> P_;
	SquareMatrix<float, Nx> P_hat;

	Matrix<float, Nx, Nx> F; 
	Matrix<float, Nx, Nx> W; 	
	Matrix<float, Nx, Nx> Qx;

	Matrix<float, Nz, Nx> H;
	Matrix<float, Nz, Nz> R;
	Matrix<float, Nz, Nz> V;

	Matrix<float, Nx, Nz> K; 
	Vector<float, Nz> z_;	
	Vector<float, Nz> residual;

	SquareMatrix<float, Nz> S;

	// Evaluation of filter Matrices
	void F_eval(const Vector<float, Nx>& x,
			const Vector<float, Nu>& u,
			const float Ts);
	void W_eval(const Vector<float, Nx>& x);
	void Qx_eval();
	void H_eval(const Vector<float, Nx>& x);
	void R_eval();
	void V_eval();	

public:
	AttEkf(float SamplingTime,
		Vector<float, N_dyn> proc_stds,
		Vector<float, N_dyn> P0s,
		Vector<float, N_meas> meas_stds);

	void setG(Vector<float, 3> g);
	void setM(Vector<float, 3> m);
	void setTs(float dt);


	void predict(Vector3<float> u);
	void update(Vector<float, Nz> z);

	Vector<float, Nq> getQuat() const;
	Vector<float, Nx> getEst() const;
	Vector<float, 3> getEul() const; 
	Vector<float, Nz> getRes() const;
	Vector<float, Nz> getZpred() const;
};
