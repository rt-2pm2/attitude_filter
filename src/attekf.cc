#include "attekf.hpp" 
#include <cmath>

using namespace matrix;

// Filter Helpers
static Vector<float, 4> xq(const Vector<float, Nx>& x);
static Vector<float, 3> xbias_gyro(const Vector<float, Nx>& x);
static Vector<float, 3> xbias_mgn(const Vector<float, Nx>& x);

static Matrix<float, 4, 3> chiMatrix(const Vector<float, 4>& q);
static Matrix<float, 4, 4> omegaMatrix(const Vector<float, 3>& w);

static Vector<float, 3> quat2eul(const Vector<float, Nq>& q);

// Eye Matrix
static SquareMatrix<float, 4> const I4 = eye<float, 4>();


/**
 * Constructor
 * \param SamplingTime Sampling time of the filter
 * \param proc_stds Vector of the standard deviations of the process noises
 * \param P0s Vector of the initial convariances of the state variables
 * \param meas_stds Vector of the standard deviations of the measurements
 */
AttEkf::AttEkf(float SamplingTime,
		Vector<float, N_dyn> proc_stds,
		Vector<float, N_dyn> P0s,
		Vector<float, N_meas> meas_stds) {

	x_hat.zero();
	// Init the quaternion
	x_hat(0) = 1;

	this->Ts = SamplingTime;

	// Initialize the proces standard deviations
	this->proc_stds = proc_stds;

	// Initialize the measurement standard deviations
	this->meas_stds = meas_stds;

	// Initialize the initial covariance
	//
	//	   | Pq   0    0  |
	// P_hat = | 0   Pbg   0  |
	//	   | 0   0    Pbm |
	//
	P_hat = eye<float, Nx>();
	P_hat.set<Nq, Nq>(eye<float, Nq>() * P0s(0), 0, 0);
	P_hat.set<Nbg, Nbg>(eye<float, Nbg>() * P0s(1), Nq, Nq);
	P_hat.set<Nbm, Nbm>(eye<float, Nbm>() * P0s(2), Nq + Nbg, Nq + Nbg);
	
	V_eval();
	R_eval();
	Qx_eval();

	// Gravity Vector
	g(2) = 9.81;

	// Local Magnetic Field Vector (normalized)
	m(0) = 0.5;
	m(2) = 0.866;

}

void AttEkf::setTs(float dt) {
	this->Ts = dt;
}

/**
 * Set Navigation frame Gravity Vector
 * \param g Gravity vector in Navigation Frame (NED)
 */
void AttEkf::setG(Vector<float, 3> g) {
	this->g = g;
}

/**
 * Set Navigation frame  
 * \param m Magnetic Vector in Navigation Frame (NED)
 */
void AttEkf::setM(Vector<float, 3> m) {
	this->m = m;
}

void AttEkf::predict(Vector3<float> u) {
	int i;
	Vector<float, 3> z_acc;
	Vector<float, 3> z_mgn;
	Vector<float, 4> q_;
	Vector<float, 3> bg;

	// State prediction: 
	bg = xbias_gyro(x_hat);
	q_ = xq(x_hat);

	// Only the attitude part is predicted, since the bias are considered random walks	
	q_ = (I4 + Ts/2 * omegaMatrix(u)) * q_ - Ts/2 * chiMatrix(q_) * bg;

	for (i = 0; i < Nq; i++)
		x_(i) = q_(i);

	// Covariance prediction 
	F_eval(x_hat, u, Ts);
	W_eval(x_hat);
	P_ = F * P_hat * F.T() + W * Qx * W.T();

	// Measurement prediction
	Dcm<float> C_bn(q_);
	z_acc = C_bn.T() * (-g),
	z_mgn = C_bn.T() * m + xbias_mgn(x_hat);
	
	for (i = 0; i < 3; i++) {
		z_(i) = z_acc(i);
		z_(i + 3) = z_mgn(i);
	}

}

void AttEkf::update(Vector<float, Nz> z) {
	Vector<float, 4> q_;

	H_eval(x_);
	S = (H * P_ * H.T() + V * R * V.T());
	K = P_ * H.T() * S.I(); 

	residual = z - z_;

	x_hat = x_ + K * (residual);

	// Normalize the quaternion
	q_ = xq(x_hat);
	q_ = q_.unit();
	x_hat.set<4,1>(q_, 0, 0); 

	// Update the Covariance Matrix
	P_hat = P_ - K * H * P_;
}


void AttEkf::W_eval(const Vector<float, Nx>& x) {
	W = (eye<float, Nx>());
	W.set<4,3> (chiMatrix(xq(x)) * (-Ts / 2.0), 0, 0);
}

// V Matrix
void AttEkf::V_eval() {
	V = (eye<float, Nz>());
}

// R Matrix
void AttEkf::R_eval() {
	R = (eye<float, Nz>());
	R.set<3,3>(eye<float, 3>() * pow(meas_stds(0), 2.0), 0, 0);
	R.set<3,3>(eye<float, 3>() * pow(meas_stds(1), 2.0), 3, 3);
}


// F Matrix
void AttEkf::F_eval(const Vector<float, Nx>& x, const Vector<float, Nu>& u,
		const float Ts) {

	F = eye<float, Nx>();
	Vector<float, Nbg> b = xbias_gyro(x);

	float data_[16] = {
		0, -b(0), -b(1), -b(2),
		b(0), 0, b(2), -b(1),
		b(1), -b(2), 0, b(0),
		b(2), b(1), -b(0), 0
	};

	Matrix<float, 4, 4> Dchi_dq(data_);
	Matrix<float, 4, 4> D1 = (I4 + Ts/2 * omegaMatrix(u)) - Ts/2 * Dchi_dq;

	Matrix<float, 4, 3> Dchi_db = -Ts/2 * chiMatrix(xq(x)); 
	
	F.set<4,4>(D1, 0, 0);
	F.set<4,3>(Dchi_db, 0, 4);
}

void AttEkf::Qx_eval() {
	Qx = eye<float, Nx>();
	Qx.set<Nq, Nq> (eye<float, Nq>() * pow(Ts * proc_stds(0), 2.0), 0, 0);
	Qx.set<Nbg, Nbg> (eye<float, Nbg>() * pow(Ts * proc_stds(1), 2.0), Nq, Nq); 
	Qx.set<Nbm, Nbm> (eye<float, Nbm>() * pow(Ts * proc_stds(2), 2.0), (Nq + Nbg), (Nq + Nbg)); 
}

void AttEkf::H_eval(const Vector<float, Nx>& x) {
       	H.zero();

	Vector<float, 4> q = xq(x);

	float q0 = q(0);
	float q1 = q(1);
	float q2 = q(2);
	float q3 = q(3);

	float dRdq0_data[] = {
		q0, q3, -q2,
		-q3, q0, q1,
		q2, -q1, q0
	};	
	Matrix<float, 3, 3> dRdq0(dRdq0_data);
	dRdq0 = dRdq0 * 2.0;


	float dRdq1_data[] = {
		q1, q2, q3,
		q2, -q1, q0,
		q3, -q0, -q1
	};
	Matrix<float, 3, 3> dRdq1(dRdq1_data);
	dRdq1 = dRdq1 * 2.0;

	float dRdq2_data[] = {
		-q2, q1, -q0,
		q1, q2, q3,
		q0, q3, -q2
	};
	Matrix<float, 3, 3> dRdq2(dRdq2_data);
	dRdq2 = dRdq2 * 2.0;

	float dRdq3_data[] = {
		-q3, q0, q1,
		-q0, -q3, q2,
		q1, q2, q3
	};
	Matrix<float, 3, 3> dRdq3(dRdq3_data);
	dRdq3 = dRdq3 * 2.0;

	Vector<float, 3> dRdq0xg = dRdq0 * (-g);
	Vector<float, 3> dRdq1xg = dRdq1 * (-g);
	Vector<float, 3> dRdq2xg = dRdq2 * (-g);
	Vector<float, 3> dRdq3xg = dRdq3 * (-g);
	
	H.set<3, 1>(dRdq0xg, 0, 0);
	H.set<3, 1>(dRdq1xg, 0, 1);
	H.set<3, 1>(dRdq2xg, 0, 2);
	H.set<3, 1>(dRdq3xg, 0, 3);

	Vector<float, 3> dRdq0xm = dRdq0 * m;
	Vector<float, 3> dRdq1xm = dRdq1 * m;
	Vector<float, 3> dRdq2xm = dRdq2 * m;
	Vector<float, 3> dRdq3xm = dRdq3 * m;

	H.set<3, 1>(dRdq0xm, 3, 0);
	H.set<3, 1>(dRdq1xm, 3, 1);
	H.set<3, 1>(dRdq2xm, 3, 2);
	H.set<3, 1>(dRdq3xm, 3, 3);

	H.set<3,3>(eye<float, 3>(), 3, 7);
}



Vector<float, Nq> AttEkf::getQuat() const {
	 Vector<float, Nq> vout = xq(x_hat);
	 return vout;	
}

Vector<float, Nx> AttEkf::getEst() const {
	 Vector<float, Nx> vout = x_hat;
	 return vout;	
}


Vector<float, 3> AttEkf::getEul() const {
	Vector<float, 4> q = xq(x_hat);
	Vector<float, 3> vout = quat2eul(q);

	return vout;
}

Vector<float, Nz> AttEkf::getRes() const {
	Vector<float, Nz> out = residual;
	return out;
}

Vector<float, Nz> AttEkf::getZpred() const {
	Vector<float, Nz> out = z_;
	return out;
}



///////////////////////////////////////////////////////////
// Helpers definition
/**
 * Extract the quaternion from the state Vector
 */
Vector<float, 4> xq(const Vector<float, Nx>& x) {
	Vector<float, 4> q;
	for (size_t i = 0; i < Nq; i++)
		q(i) = x(i);
	return q;
} 

/**
 * Extract the gyro bias from the state Vector
 */
Vector<float, 3> xbias_gyro(const Vector<float, Nx>& x) {
	Vector<float, 3> bg;
	for (size_t i = 0; i < Nbg; i++) {
		bg(i) = x(i + Nq);
	}

	return bg;
}

/**
 * Extract the magnetometer bias from the state Vector
 */
Vector<float, 3> xbias_mgn(const Vector<float, Nx>& x) {
	Vector<float, 3> bm;
	for (size_t i = 0; i < Nbm; i++)
		bm(i) = x(i + Nq + Nbg);
	return bm;
}

// SUPPORT MATRIX
/** Chi Matrix
 * Given a quaterion q = [q0; e] in R^4, the matrix has the form
 *
 *        [      -e.T()      ]
 * Chi =  [ I*q0 + [e.hat()] ]
*/
Matrix<float, 4, 3> chiMatrix(const Vector<float, 4>& q) { 
	float q0 = q(0);
	Vector3<float> e = {q(1), q(2), q(3)};

	Matrix<float, 4, 3> ret;
	ret.setRow(0, -e);	
	ret.set<3,3>(q0 * eye<float, 3>() + e.hat(), 1, 0);
	return ret;
}

/** Omega Matrix
 * Give the vector of the angular velocity
 *     [  0    -w.T()   ]
 * O = [  w   -w.hat()  ]
 *
 */
Matrix<float, 4, 4> omegaMatrix(const Vector<float, 3>& w) {
	Matrix<float, 4, 4> ret;

	Vector<float, 4> v;
	for (size_t i = 0; i < 3; i++)
	       v(i+1) = w(i);	

	ret.setCol(0, v);
	v = -v;
	ret.setRow(0, v);	

	Vector3<float> vv = -w;
	ret.set<3,3>(vv.hat(), 1, 1);
	return ret;
}

/** 
 * Conversion from quaternion to euler angles
 */
Vector<float, 3> quat2eul(const Vector<float, Nq>& q) {
	Vector<float, 3> out;

	float q0 = q(0);
	float q1 = q(1);
	float q2 = q(2);
	float q3 = q(3);

	float t1 = 2.0 * (q2 * q3 + q0 * q1);
	float t2 = (q3 * q3) - (q2 * q2) - (q1 * q1) + (q0 * q0) ;

	float d1 = 2.0 * (q1 * q3 - q0 * q2);
	
	float e1 = 2.0 * (q1 * q2 + q0 * q3);
	float e2 = (q1 * q1) + (q0 * q0) - (q3 * q3) - (q2 * q2);

	out(0) = atan2(t1, t2);
	out(1) = -asin(d1);
	out(2) = atan2(e1, e2);

	return out;
}
