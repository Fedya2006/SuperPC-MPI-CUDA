#pragma once
#include "utils.h"
#include <cmath>
#include "math.h"
#include <mpi.h>
#include <chrono>
#include <cuda.h>

#include "Array3D.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "device_launch_parameters.h"

//const double M_PI = 3.1415;



const bool GPU = true;
const bool CPU = false;

const bool DEBUG_KERNELS = false;



class Sequence
{

private:
    double Lx, Ly, Lz;
    double T, dt;

    double hx;
    double hy;
    double hz;
    
    int K, N;
    int local_start_x, local_end_x;
    int local_start_y, local_end_y;
    int local_start_z, local_end_z;

    int global_start_x, global_end_x;
    int global_start_y, global_end_y;
    int global_start_z, global_end_z;



    Array3D analitical(double t);
    double laplacian(const Array3D& u, int i, int j, int k);
    static Array3D U0(const Array3D& u) ;
    Array3D U1(const Array3D& u);

    void exchange_borders(double *  data);
void exchange_borders(MPI_Comm  cart_comm, Array3D & data) const;

void time_from_that_moment(std::string comment, std::chrono::steady_clock::time_point moment);


    void init();
    void free();



    //MPI navigation variables
    int cart_size, cart_rank;
    MPI_Comm cart_comm;


    //Added to support CUDA computation
    thrust::host_vector<double> h_a0, h_u_prev, h_u_curr;
    thrust::device_vector<double> d_u_prev, d_u_curr, d_u_next, d_a0;


public:
    Sequence(double T,  int K, int N,
             double Lx,  double Ly,  double Lz,
             int global_start_x, int global_end_x,
             int global_start_y, int global_end_y,
             int global_start_z, int global_end_z, MPI_Comm cart_comm);

    ~Sequence();

    
    double * send_buffer_x_left = nullptr;
    double * recv_buffer_x_right = nullptr;
    double * send_buffer_x_right = nullptr;
    double * recv_buffer_x_left = nullptr;

    double * send_buffer_y_left = nullptr;
    double * recv_buffer_y_right = nullptr;
    double * send_buffer_y_right = nullptr;
    double * recv_buffer_y_left = nullptr;

    double * send_buffer_z_left = nullptr;
    double * recv_buffer_z_right = nullptr;
    double * send_buffer_z_right = nullptr;
    double * recv_buffer_z_left = nullptr;

    double * send_buffer_x_left2 = nullptr;
    double * recv_buffer_x_right2 = nullptr;
    double * send_buffer_x_right2 = nullptr;
    double * recv_buffer_x_left2 = nullptr;

    double * send_buffer_y_left2 = nullptr;
    double * recv_buffer_y_right2 = nullptr;
    double * send_buffer_y_right2 = nullptr;
    double * recv_buffer_y_left2 = nullptr;

    double * send_buffer_z_left2 = nullptr;
    double * recv_buffer_z_right2 = nullptr;
    double * send_buffer_z_right2 = nullptr;
    double * recv_buffer_z_left2 = nullptr;

    double * device_send_buffer_x_left = nullptr;
    double * device_recv_buffer_x_right = nullptr;
    double * device_send_buffer_x_right = nullptr;
    double * device_recv_buffer_x_left = nullptr;

    double * device_send_buffer_y_left = nullptr;
    double * device_recv_buffer_y_right = nullptr;
    double * device_send_buffer_y_right = nullptr;
    double * device_recv_buffer_y_left = nullptr;

    double * device_send_buffer_z_left = nullptr;
    double * device_recv_buffer_z_right = nullptr;
    double * device_send_buffer_z_right = nullptr;
    double * device_recv_buffer_z_left = nullptr;


    void run();
};
