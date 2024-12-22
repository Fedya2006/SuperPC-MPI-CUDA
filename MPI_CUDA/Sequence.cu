#include "Sequence.h"
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


double diff(double a, double b) {
    double d = a - b;

    if (d > 0)
        return d;
    else
        return -d;

}

void compare(const thrust::device_vector<double>& d_u, Array3D& arr)
{
    thrust::host_vector<double> u = d_u;

    cudaDeviceSynchronize();

    int Nx = arr.Nx;
    int Ny = arr.Ny;
    int Nz = arr.Nz;

    std::cout << "COMPARE " << Nx << " " << Ny << " " << Nz << std::endl;

    int cnt = 0;

    for (int i = 0; i <= Ny; ++i)
        for (int j = 0; j <= Nx; ++j)
            for (int k = 0; k <= Nz; ++k) {

                if (i == k_i && j == k_j && k == k_k) {
                    printf("COMPARE: %.15f %.15f \n", u[ind(i, j, k)], arr(i, j, k));
                }

                if (diff(u[ind(i, j, k)], arr(i, j, k)) > 0.0000000001) {
                    std::cout << "difference was spot in " << i << " " << j << " " << k << std::endl;
                    std::cout << "diff = " << diff(u[ind(i, j, k)], arr(i, j, k)) << std::endl;
                    cnt++;
                }

                if (cnt > 0)
                    throw std::runtime_error("bad");
            }

}



__global__ void print( double* device_recv_buffer_x_right, int N){

  int i = blockIdx.y * blockDim.y;
  int j = blockIdx.z * blockDim.x + threadIdx.x;
  int k = blockIdx.x * blockDim.z + threadIdx.z;


    if(i == k_i && j == k_j && k == k_k ){
        printf("kernel_my_print-->>> here %d %d %d \n", i, j, k);
        printf("kernel_my_print = %.15f\n", device_recv_buffer_x_right[k_k * (N + 1) + k_i]);
    }
}

Array3D Sequence::analitical(const double t) 
{
    int Nx = local_end_x, Ny = local_end_y, Nz = local_end_z;
    Array3D array(Nx, Ny, Nz);

    double hx = Lx / N;
    double hy = Ly / N;
    double hz = Lz / N;

    double at = M_PI * sqrt(4.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 4.0 / (Lz * Lz));

    for (int i = 0; i <= Ny; ++i)
        for (int j = 0; j <= Nx; ++j)
            for (int k = 0; k <= Nz; ++k)
            {
                double x = (j + global_start_x) * hx;
                double y = (i + global_start_y) * hy;
                double z = (k + global_start_z) * hz;


                const double value = sin(2.0 * M_PI * x / Lx) *
                                     sin((M_PI * y / Ly) + M_PI) *
                                     sin((2.0 * M_PI * z / Lz) + 2.0 * M_PI) *
                                     cos(at * t + M_PI);


                array.set(value, i, j, k);
            }


    return array;
}




Sequence::~Sequence(){
  cudaFree(device_recv_buffer_x_right);
  cudaFree( device_recv_buffer_x_left);
  cudaFree(device_recv_buffer_y_right);
  cudaFree( device_recv_buffer_y_left);
  cudaFree(device_recv_buffer_z_right);
  cudaFree( device_recv_buffer_z_left);
}




__device__ double laplacian_inside(double* u, int i, int j, int k,
    double hx, double hy, double hz,
    int Nx, int Ny, int Nz)
{
    double y1 = u[ind(i + 1, j, k)];
    double y0 = u[ind(i, j, k)];
    double y_1 = u[ind(i - 1, j, k)];

    double x1 =  u[ind(i, j + 1, k)];
    double x0 = u[ind(i, j, k)];
    double x_1 = u[ind(i, j - 1, k)];

    double z1 = u[ind(i, j, k + 1)];
    double z0 = u[ind(i, j, k)];
    double z_1 = u[ind(i, j, k - 1)];

    return (y1 - 2 * y0 + y_1) / (hy * hy) + (x1 - 2 * x0 + x_1) / (hx * hx) + (z1 - 2 * z0 + z_1) / (hz * hz);
}



__device__ double laplacian(double *u, int i, int j, int k,
                            int Nx, int Ny, int Nz,
                            double Lx, double Ly, double Lz,
                            int N,
                                                        
                            double * device_recv_buffer_x_right,
                            double * device_recv_buffer_x_left,
                            double * device_recv_buffer_y_right,
                            double * device_recv_buffer_y_left,
                            double * device_recv_buffer_z_right,
                            double * device_recv_buffer_z_left) 
{
    const double hx = Lx / N;
    const double hy = Ly / N;
    const double hz = Lz / N;

    double y1 = (i == Ny ? device_recv_buffer_y_right[k * (Nx + 1) + j] : u[ind(i + 1, j, k)]);
    double y0 = u[ind(i, j, k)];
    double y_1 = (i == 0 ? device_recv_buffer_y_left[k * (Nx + 1) + j] : u[ind(i - 1, j, k)]);

    double x1 = (j == Nx ? device_recv_buffer_x_right[k * (Ny + 1) + i] : u[ind(i,j + 1,k)]);
    double x0 = u[ind(i, j, k)];
    double x_1 = (j == 0 ? device_recv_buffer_x_left[k * (Ny + 1) + i] : u[ind(i,j - 1,k)]);

    double z1 = (k == Nz ? device_recv_buffer_z_right[i * (Nx + 1) + j] : u[ind(i, j,k + 1)]);
    double z0 = u[ind(i, j, k)];
    double z_1 = (k == 0 ? device_recv_buffer_z_left[i * (Nx + 1) + j] : u[ind(i, j,k - 1)]);


    if(DEBUG_KERNELS){
        if(i == k_i && j == k_j && k == k_k ){
            printf("kernel laplacian output: rb_zl[%d,%d,%d] = %.15f\n", i, j, k, device_recv_buffer_z_left[i * (Nx + 1) + j]);
            //printf("kernel laplacian output: rb_xr[1,128,1] = %.15f\n", device_recv_buffer_x_right[k_k * (Ny + 1) + k_i]);
            printf("kernel laplacian output: %.15f %.15f %.15f | %.15f %.15f %.15f | %.15f %.15f %.15f  \n", y1, y0, y_1, x1, x0, x_1, z1, z0, z_1);
        }
    }

    return (y1 - 2 * y0 + y_1) / (hy * hy) + (x1 - 2 * x0 + x_1) / (hx * hx) + (z1 - 2 * z0 + z_1) / (hz * hz);
}


double Sequence::laplacian(const Array3D & u, int i, int j, int k) 
{
    const double hx = Lx / N;
    const double hy = Ly / N;
    const double hz = Lz / N;

    if (i < 0 || i > u.Ny || j < 0 || j > u.Nx || k < 0 || k > u.Nz)
        throw std::out_of_range("Index out of bounds in laplacian");

    double y1 = (i == u.Ny ? recv_buffer_y_right[k * (u.Nx + 1) + j] : u(i + 1, j, k));
    double y0 = u(i, j, k);
    double y_1 = (i == 0 ? recv_buffer_y_left[k * (u.Nx + 1) + j] : u(i - 1, j, k));

    double x1 = (j == u.Nx ? recv_buffer_x_right[k * (u.Ny + 1) + i] : u(i,j + 1,k));
    double x0 = u(i, j, k);
    double x_1 = (j == 0 ? recv_buffer_x_left[k * (u.Ny + 1) + i] : u(i,j - 1,k));

    double z1 = (k == u.Nz ? recv_buffer_z_right[i * (u.Nx + 1) + j] : u(i, j,k + 1));
    double z0 = u(i, j, k);
    double z_1 = (k == 0 ? recv_buffer_z_left[i * (u.Nx + 1) + j] : u(i, j,k - 1));

    if(i == k_i && j == k_j && k == k_k ){
        //printf("host laplacian output: rb_xr[1,128,1] = %.15f\n", recv_buffer_x_right[k * (u.Ny + 1) + i]);
        printf("host laplacian output: rb_zl[%d,%d,%d] = %.15f\n", i, j, k, recv_buffer_z_left[i * (u.Nx + 1) + j]);
        printf("host laplacian output: %.15f %.15f %.15f | %.15f %.15f %.15f | %.15f %.15f %.15f  \n", y1, y0, y_1, x1, x0, x_1, z1, z0, z_1);
    }

    return (y1 - 2 * y0 + y_1) / (hy * hy) + (x1 - 2 * x0 + x_1) / (hx * hx) + (z1 - 2 * z0 + z_1) / (hz * hz);
}

Array3D Sequence::U0(const Array3D & u) {
    auto u0 = Array3D(u.Nx, u.Ny, u.Nz);

    for (int i = 0; i <= u.Ny; ++i)
        for (int j = 0; j <= u.Nx; ++j)
            for (int k = 0; k <= u.Nz; ++k)
                u0.set(u(i,j,k), i,j,k);
    return u0;
}

Array3D Sequence::U1(const Array3D & u)
{
    auto u1 = Array3D(u.Nx, u.Ny, u.Nz);
    for (int i = 0; i <= u.Ny; ++i)
        for (int j = 0; j <= u.Nx; ++j)
            for (int k = 0; k <= u.Nz; ++k)
            {
                if (!((i + global_start_y == 0) || (i + global_start_y == N) ||
                (j + global_start_x == 0) || (j + global_start_x == N) ||
                (k + global_start_z == 0) || (k + global_start_z == N)))
                    u1.set(u(i,j,k) + (dt * dt / 2) * laplacian(u, i, j, k), i, j, k);
                else
                    u1.set(0.0, i, j, k);

            }

    return u1;
}








__global__ void foul_fill_buffers(double* data, int Nx, int Ny, int Nz, 

                                int c0, int c1, int c2, 
                                int d0, int d1, int d2,
                                 double* send_buffer_x_left,  double* send_buffer_x_right, 
                                   double* send_buffer_y_left,  double* send_buffer_y_right, 
                                double* send_buffer_z_left,  double* send_buffer_z_right   ) {
    
    
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;


    if( i >= 0 && i <= Ny  && 
        j >= 0 && j <= Nx  && 
        k >= 0 && k <= Nz  )
        {

        // x-:
        if(j == 0)
            send_buffer_x_left[k * (Ny + 1) + i] = data[ind(i, (c0 == 0 ? 1 : 0), k) ];

        // x+:
        if(j == Nx)
            send_buffer_x_right[k * (Ny + 1) + i] = data[ind(i, (c0 == d0 - 1 ? Nx - 1 : Nx), k)];


        // y-:
        if(i == 0)
            send_buffer_y_left[k * (Nx + 1) + j] = data[ind(0, j, k)];


        // y+:
        if(i == Ny)
                send_buffer_y_right[k * (Nx + 1) + j] = data[ind(Ny, j, k)];

        // z-:
        if(k == 0)
            send_buffer_z_left[i * (Nx + 1) + j] = data[ind(i, j, (c2 == 0 ? 1 : 0))];

        // z+:
        if(k == Nz)
            send_buffer_z_right[i * (Nx + 1) + j] = data[ind(i, j, (c2 == d2 - 1 ? Nz - 1 : Nz))];


        }

}







void Sequence::exchange_borders(MPI_Comm  cart_comm, Array3D & data) const
{

    int world_size;
    int world_rank;
    MPI_Comm_size(cart_comm, &world_size);
    MPI_Comm_rank(cart_comm, &world_rank);

    int ndims = 3;
    int dims[3], periods[3], notused[3], coords[3];

    MPI_Cart_get(cart_comm, ndims, dims, periods, notused);
    MPI_Cart_coords(cart_comm, world_rank, 3, coords);

    int neighbors[6];
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[0], &neighbors[1]); // x- и x+
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[2], &neighbors[3]); // y- и y+
    MPI_Cart_shift(cart_comm, 2, 1, &neighbors[4], &neighbors[5]); // z- и z+


    int Nx = local_end_x , Ny = local_end_y, Nz = local_end_z;


    // x-:
    for (int k = 0; k <= data.Nz; ++k)
        for (int i = 0; i <= data.Ny; ++i)
            send_buffer_x_left[k * (Ny + 1) + i] = data(i, (coords[0] == 0 ? 1 : 0), k);

    //printf("host_my_print-->>> here %d %d %d \n", i, j, k);
    printf("host_my_print = %.15f\n", send_buffer_x_left[k_k * (N + 1) + k_i]);


    MPI_Sendrecv(send_buffer_x_left, (Nz + 1) * (Ny + 1), MPI_DOUBLE, neighbors[0], 1,
                     recv_buffer_x_right, (Nz + 1) * (Ny + 1), MPI_DOUBLE, neighbors[1], 1,
                     cart_comm, MPI_STATUS_IGNORE);


    // x+:
    for (int k = 0; k <= data.Nz; ++k)
        for (int i = 0; i <= data.Ny; ++i)
                send_buffer_x_right[k * (Ny + 1) + i] = data(i, (coords[0] == dims[0] - 1 ? data.Nx - 1 : data.Nx), k);

    MPI_Sendrecv(send_buffer_x_right, (Nz + 1) * (Ny + 1), MPI_DOUBLE, neighbors[1], 2,
                     recv_buffer_x_left, (Nz + 1) * (Ny + 1), MPI_DOUBLE, neighbors[0], 2,
                     cart_comm, MPI_STATUS_IGNORE);



    // y-:
    for (int k = 0; k <= data.Nz; ++k)
        for (int j = 0; j <= data.Nx; ++j)
            send_buffer_y_left[k * (Nx + 1) + j] = data(0, j, k);

    MPI_Sendrecv(send_buffer_y_left, (Nz + 1) * (Nx + 1), MPI_DOUBLE,
                                neighbors[2], 3,
                                recv_buffer_y_right, (Nz + 1) * (Nx + 1), MPI_DOUBLE,
                                neighbors[3], 3,
                                cart_comm, MPI_STATUS_IGNORE);


    // y+:
    for (int k = 0; k <= data.Nz; ++k)
        for (int j = 0; j <= data.Nx; ++j)
            send_buffer_y_right[k * (Nx + 1) + j] = data(data.Ny, j, k);

    MPI_Sendrecv(send_buffer_y_right, (Nz + 1) * (Nx + 1), MPI_DOUBLE,
                                neighbors[3], 4,
                                recv_buffer_y_left, (Nz + 1) * (Nx + 1), MPI_DOUBLE,
                                neighbors[2], 4,
                                cart_comm, MPI_STATUS_IGNORE);



    // z-:
    for (int i = 0; i <= data.Ny; ++i)
        for (int j = 0; j <= data.Nx; ++j)
            send_buffer_z_left[i * (Nx + 1) + j] = data(i, j, (coords[2] == 0 ? 1 : 0));

    MPI_Sendrecv(send_buffer_z_left, (Ny + 1) * (Nx + 1), MPI_DOUBLE, neighbors[4], 5,
                     recv_buffer_z_right, (Ny + 1) * (Nx + 1), MPI_DOUBLE, neighbors[5],5,
                     cart_comm, MPI_STATUS_IGNORE);



    // z+:
    for (int i = 0; i <= data.Ny; ++i)
        for (int j = 0; j <= data.Nx; ++j)
                send_buffer_z_right[i * (Nx + 1) + j] = data(i, j, (coords[2] == dims[2] - 1 ? data.Nz - 1 : data.Nz));

    MPI_Sendrecv(send_buffer_z_right, (Ny + 1) * (Nx + 1), MPI_DOUBLE, neighbors[5], 6,
                     recv_buffer_z_left, (Ny + 1) * (Nx + 1), MPI_DOUBLE, neighbors[4], 6,
                     cart_comm, MPI_STATUS_IGNORE);

    printf("host exchange: rb_xr[1,128,1] = %.15f\n", recv_buffer_x_right[1 * (Ny + 1) + 1]);

}





void Sequence::exchange_borders(double * data)
{
   int ndims = 3;
    int dims[3], periods[3], notused[3], coords[3];

    MPI_Cart_get(cart_comm, ndims, dims, periods, notused);
    MPI_Cart_coords(cart_comm, cart_rank, 3, coords);

    int neighbors[6];

    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[0], &neighbors[1]); // x- и x+
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[2], &neighbors[3]); // y- и y+
    MPI_Cart_shift(cart_comm, 2, 1, &neighbors[4], &neighbors[5]); // z- и z+
    int Nx = local_end_x , Ny = local_end_y, Nz = local_end_z;


    


    foul_fill_buffers<<<dim3((Nz + 1) / 32 + 1, Ny, (Nx + 1) / 32 + 1), dim3(32, 1, 32)>>>( data,  Nx,  Ny,  Nz, 

                                 coords[0], coords[1], coords[2],
                                 dims[0], dims[1], dims[2],

                                  device_send_buffer_x_left,   device_send_buffer_x_right, 
                                    device_send_buffer_y_left,   device_send_buffer_y_right, 
                                 device_send_buffer_z_left,   device_send_buffer_z_right   );

    cudaDeviceSynchronize();

    cudaMemcpy(send_buffer_x_right2, device_send_buffer_x_right , (Nz + 1) * (Ny + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(send_buffer_x_left2, device_send_buffer_x_left , (Nz + 1) * (Ny + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(send_buffer_y_right2, device_send_buffer_y_right , (Nz + 1) * (Nx + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(send_buffer_y_left2, device_send_buffer_y_left , (Nz + 1) * (Nx + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(send_buffer_z_right2, device_send_buffer_z_right , (Ny + 1) * (Nx + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(send_buffer_z_left2, device_send_buffer_z_left , (Ny + 1) * (Nx + 1) * sizeof(double), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();

    if(DEBUG_KERNELS){
        print<<<dim3((Nz + 1) / 16 + 1, Ny + 1, (Nx + 1) / 32 + 1), dim3(32, 1, 16)>>>(device_send_buffer_x_left, N );
    }

    MPI_Sendrecv(send_buffer_x_left2, (Nz + 1) * (Ny + 1), MPI_DOUBLE, neighbors[0], 1,
                     recv_buffer_x_right2, (Nz + 1) * (Ny + 1), MPI_DOUBLE, neighbors[1], 1,
                     cart_comm, MPI_STATUS_IGNORE);


    cudaMemcpy(device_recv_buffer_x_right, recv_buffer_x_right2, (Nz + 1) * (Ny + 1) * sizeof(double), cudaMemcpyHostToDevice);



    MPI_Sendrecv(send_buffer_x_right2, (Nz + 1) * (Ny + 1), MPI_DOUBLE, neighbors[1], 2,
                     recv_buffer_x_left2, (Nz + 1) * (Ny + 1), MPI_DOUBLE, neighbors[0], 2,
                     cart_comm, MPI_STATUS_IGNORE);


    cudaMemcpy(device_recv_buffer_x_left, recv_buffer_x_left2, (Nz + 1) * (Ny + 1) * sizeof(double), cudaMemcpyHostToDevice);



    MPI_Sendrecv(send_buffer_y_left2, (Nz + 1) * (Nx + 1), MPI_DOUBLE,
                                neighbors[2], 3,
                                recv_buffer_y_right2, (Nz + 1) * (Nx + 1), MPI_DOUBLE,
                                neighbors[3], 3,
                                cart_comm, MPI_STATUS_IGNORE);

    cudaMemcpy(device_recv_buffer_y_right, recv_buffer_y_right2, (Nz + 1) * (Nx + 1) * sizeof(double), cudaMemcpyHostToDevice);



    MPI_Sendrecv(send_buffer_y_right2, (Nz + 1) * (Nx + 1), MPI_DOUBLE,
                                neighbors[3], 4,
                                recv_buffer_y_left2, (Nz + 1) * (Nx + 1), MPI_DOUBLE,
                                neighbors[2], 4,
                                cart_comm, MPI_STATUS_IGNORE);

    cudaMemcpy(device_recv_buffer_y_left, recv_buffer_y_left2, (Nz + 1) * (Nx + 1) * sizeof(double), cudaMemcpyHostToDevice);



    MPI_Sendrecv(send_buffer_z_left2, (Ny + 1) * (Nx + 1), MPI_DOUBLE, neighbors[4], 5,
                     recv_buffer_z_right2, (Ny + 1) * (Nx + 1), MPI_DOUBLE, neighbors[5],5,
                     cart_comm, MPI_STATUS_IGNORE);

    cudaMemcpy(device_recv_buffer_z_right, recv_buffer_z_right2, (Ny + 1) * (Nx + 1) * sizeof(double), cudaMemcpyHostToDevice);


    MPI_Sendrecv(send_buffer_z_right2, (Ny + 1) * (Nx + 1), MPI_DOUBLE, neighbors[5], 6,
                     recv_buffer_z_left2, (Ny + 1) * (Nx + 1), MPI_DOUBLE, neighbors[4], 6,
                     cart_comm, MPI_STATUS_IGNORE);

    cudaMemcpy(device_recv_buffer_z_left, recv_buffer_z_left2, (Ny + 1) * (Nx + 1) * sizeof(double), cudaMemcpyHostToDevice);



    //printf("kernel laplacian output: rb_xr[1,128,1] = %.15f\n", device_recv_buffer_x_right[k_k * (Ny + 1) + k_i]);
    printf("kernel exchange: rb_xr[1,128,1] = %.15f\n", recv_buffer_x_right2[1 * (Ny + 1) + 1]);

    
    cudaDeviceSynchronize();


    if(DEBUG_KERNELS){
        print<<<dim3((Nz + 1) / 16 + 1, Ny + 1, (Nx + 1) / 32 + 1), dim3(32, 1, 16)>>>(device_recv_buffer_x_right, N );
    }
}


Sequence::Sequence(const double T, const int K, const int N,
                   const double Lx, const double Ly, const double Lz,
                   int global_start_x, int global_end_x,
                   int global_start_y, int global_end_y,
                   int global_start_z, int global_end_z, MPI_Comm cart_comm) :

                   Lx(Lx), Ly(Ly), Lz(Lz),
                   global_start_x(global_start_x), global_end_x(global_end_x),
                   global_start_y(global_start_y), global_end_y(global_end_y),
                   global_start_z(global_start_z), global_end_z(global_end_z),
                   T(T), K(K), N(N), cart_comm(cart_comm)
                   {
                        dt = T / K;

                       local_end_x = global_end_x - global_start_x;
                       local_end_y = global_end_y - global_start_y;
                       local_end_z = global_end_z - global_start_z;

                       local_start_x = 0;
                       local_start_y = 0;
                       local_start_z = 0;

    MPI_Comm_size(cart_comm, &cart_size);
    MPI_Comm_rank(cart_comm, &cart_rank);

                       init();

    hx = Lx / N;
    hy = Ly / N;
     hz = Lz / N;

                   }


void Sequence::init() 
{
    int Nx = local_end_x, Ny = local_end_y, Nz = local_end_z;

    // Размеры граней
    int face_size_x = (Nz + 1) * (Ny + 1);
    int face_size_y = (Nz + 1) * (Nx + 1);
    int face_size_z = (Ny + 1) * (Nx + 1);


    send_buffer_x_left = static_cast<double *>(malloc(face_size_x * sizeof(double)));
    recv_buffer_x_right = static_cast<double *>(malloc(face_size_x * sizeof(double)));
    send_buffer_x_right = static_cast<double *>(malloc(face_size_x * sizeof(double)));
    recv_buffer_x_left = static_cast<double *>(malloc(face_size_x * sizeof(double)));

    send_buffer_y_left = static_cast<double *>(malloc(face_size_y * sizeof(double)));
    recv_buffer_y_right = static_cast<double *>(malloc(face_size_y * sizeof(double)));
    send_buffer_y_right = static_cast<double *>(malloc(face_size_y * sizeof(double)));
    recv_buffer_y_left = static_cast<double *>(malloc(face_size_y * sizeof(double)));

    send_buffer_z_left = static_cast<double *>(malloc(face_size_z * sizeof(double)));
    recv_buffer_z_right = static_cast<double *>(malloc(face_size_z * sizeof(double)));
    send_buffer_z_right = static_cast<double *>(malloc(face_size_z * sizeof(double)));
    recv_buffer_z_left = static_cast<double *>(malloc(face_size_z * sizeof(double)));


    //cuda
    send_buffer_x_left2 = static_cast<double *>(malloc(face_size_x * sizeof(double)));
    recv_buffer_x_right2 = static_cast<double *>(malloc(face_size_x * sizeof(double)));
    send_buffer_x_right2 = static_cast<double *>(malloc(face_size_x * sizeof(double)));
    recv_buffer_x_left2 = static_cast<double *>(malloc(face_size_x * sizeof(double)));

    send_buffer_y_left2 = static_cast<double *>(malloc(face_size_y * sizeof(double)));
    recv_buffer_y_right2 = static_cast<double *>(malloc(face_size_y * sizeof(double)));
    send_buffer_y_right2 = static_cast<double *>(malloc(face_size_y * sizeof(double)));
    recv_buffer_y_left2 = static_cast<double *>(malloc(face_size_y * sizeof(double)));

    send_buffer_z_left2 = static_cast<double *>(malloc(face_size_z * sizeof(double)));
    recv_buffer_z_right2 = static_cast<double *>(malloc(face_size_z * sizeof(double)));
    send_buffer_z_right2 = static_cast<double *>(malloc(face_size_z * sizeof(double)));
    recv_buffer_z_left2 = static_cast<double *>(malloc(face_size_z * sizeof(double)));


    cudaMalloc( (void**)& device_send_buffer_x_left, face_size_x * sizeof(double));
    cudaMalloc( (void**)& device_recv_buffer_x_left, face_size_x * sizeof(double));
    cudaMalloc( (void**)& device_send_buffer_x_right, face_size_x * sizeof(double));
    cudaMalloc( (void**)& device_recv_buffer_x_right, face_size_x * sizeof(double));

    cudaMalloc( (void**)& device_send_buffer_y_left, face_size_y * sizeof(double));
    cudaMalloc( (void**)& device_recv_buffer_y_left, face_size_y * sizeof(double));
    cudaMalloc( (void**)& device_send_buffer_y_right, face_size_y * sizeof(double));
    cudaMalloc( (void**)& device_recv_buffer_y_right, face_size_y * sizeof(double));

    cudaMalloc( (void**)& device_send_buffer_z_left, face_size_z * sizeof(double));
    cudaMalloc( (void**)& device_recv_buffer_z_left, face_size_z * sizeof(double));
    cudaMalloc( (void**)& device_send_buffer_z_right, face_size_z * sizeof(double));
    cudaMalloc( (void**)& device_recv_buffer_z_right, face_size_z * sizeof(double));
}

void Sequence::free()
{
    ::free(send_buffer_x_left);
    ::free(recv_buffer_x_right);
    ::free(send_buffer_x_right);
    ::free(recv_buffer_x_left);

    ::free(send_buffer_y_left);
    ::free(recv_buffer_y_right);
    ::free(send_buffer_y_right);
    ::free(recv_buffer_y_left);

    ::free(send_buffer_z_left);
    ::free(recv_buffer_z_right);
    ::free(send_buffer_z_right);
    ::free(recv_buffer_z_left);

    free();
}






__global__ void time_step_inside(double* u_prev, double* u_curr, double* u_next,
    double hx, double hy, double hz,
    double N,
    int Nx, int Ny, int Nz,
    double Lx, double Ly, double Lz,
    double dt) {


    int i = blockIdx.y * blockDim.y;
    int j = blockIdx.z * blockDim.x + threadIdx.x;
    int k = blockIdx.x * blockDim.z + threadIdx.z;


    //only in center
    if (k > 0 && k < Nz &&
        i > 0 && i < Ny &&
        j > 0 && j < Nx) {

            //double laplacian_val = 0.557;
            double laplacian_val = laplacian_inside(u_curr, i, j, k,
                hx, hy, hz,
                Nx, Ny, Nz );

            double value = 2.0 * u_curr[ind(i, j, k)] - u_prev[ind(i, j, k)] + dt * dt * laplacian_val;

            u_next[ind(i, j, k)] = value;
        }
}







__global__ void time_step_border(double* u_prev, double* u_curr, double* u_next,
    double hx, double hy, double hz,
    double N,
    int Nx, int Ny, int Nz,
    double Lx, double Ly, double Lz,
    int global_start_y,

    double* device_recv_buffer_x_right,
    double* device_recv_buffer_x_left,
    double* device_recv_buffer_y_right,
    double* device_recv_buffer_y_left,
    double* device_recv_buffer_z_right,
    double* device_recv_buffer_z_left,
    double dt) {


    int i = blockIdx.y * blockDim.y;
    int j = blockIdx.z * blockDim.x + threadIdx.x;
    int k = blockIdx.x * blockDim.z + threadIdx.z;



    //only on borders
    if (k == 0 || k == Nz ||
        i == 0 || i == Ny ||
        j == 0 || j == Nx) {

        if (i <= Ny &&
            j <= Nx &&
            k <= Nz) {


            if ((i + global_start_y != 0) && (i + global_start_y != N))
            {
                //double laplacian_val = 0.557;
                double laplacian_val = laplacian(u_curr, i, j, k,
                    Nx, Ny, Nz,
                    Lx, Ly, Lz, N, device_recv_buffer_x_right,
                    device_recv_buffer_x_left,
                    device_recv_buffer_y_right,
                    device_recv_buffer_y_left,
                    device_recv_buffer_z_right,
                    device_recv_buffer_z_left);

                double value = 2.0 * u_curr[ind(i, j, k)] - u_prev[ind(i, j, k)] + dt * dt * laplacian_val;

                u_next[ind(i, j, k)] = value;
            }
            else {
                u_next[ind(i, j, k)] = 0;
            }
        }
    }

}








__global__ void time_step(double* u_prev, double* u_curr, double* u_next,
    double hx, double hy, double hz,
    double N,
    int Nx, int Ny, int Nz,
    double Lx, double Ly, double Lz,
    int global_start_y,

    double* device_recv_buffer_x_right,
    double* device_recv_buffer_x_left,
    double* device_recv_buffer_y_right,
    double* device_recv_buffer_y_left,
    double* device_recv_buffer_z_right,
    double* device_recv_buffer_z_left,
    double dt) {


    int i = blockIdx.y * blockDim.y;
    int j = blockIdx.z * blockDim.x + threadIdx.x;
    int k = blockIdx.x * blockDim.z + threadIdx.z;


    /*if(blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0){
        printf("offsets %d, %d, %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    }*/

    //if(i == k_i  ){

    if (DEBUG_KERNELS) {
        if (i == k_i && j == k_j && k == k_k) {
            printf("-->>> here %d %d %d \n", i, j, k);
            printf("kernel time_step  start output: rb_xr[1,128,1] = %.15f\n", device_recv_buffer_x_right[k_k * (Ny + 1) + k_i]);
        }
    }


    if (k >= 0 && k <= Nz &&
        i >= 0 && i <= Ny &&
        j >= 0 && j <= Nx) {

        if ((i + global_start_y != 0) && (i + global_start_y != N))
        {
            //double laplacian_val = 0.557;
            double laplacian_val = laplacian(u_curr, i, j, k,
                Nx, Ny, Nz,
                Lx, Ly, Lz, N, device_recv_buffer_x_right,
                device_recv_buffer_x_left,
                device_recv_buffer_y_right,
                device_recv_buffer_y_left,
                device_recv_buffer_z_right,
                device_recv_buffer_z_left);


            if (DEBUG_KERNELS) {
                if (i == k_i && j == k_j && k == k_k) {
                    printf("kernel output: laplacian = %.15f \n", laplacian_val);
                    printf("kernel output: ucurr = %.15f \n", u_curr[ind(i, j, k)]);
                    printf("kernel output: uprev = %.15f \n", u_prev[ind(i, j, k)]);
                }
            }

            double value = 2.0 * u_curr[ind(i, j, k)] - u_prev[ind(i, j, k)] + dt * dt * laplacian_val;



            u_next[ind(i, j, k)] = value;
        }
        else {
            u_next[ind(i, j, k)] = 0;
        }

        if (DEBUG_KERNELS) {
            if (i == k_i && j == k_j && k == k_k) {
                printf("kernel output: %.15f \n", u_next[ind(i, j, k)]);
            }
        }

    }

}






void Sequence::time_from_that_moment(std::string comment, std::chrono::steady_clock::time_point moment){
    cudaDeviceSynchronize();

    if (cart_rank > 0)
        return;
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::cout << comment  << std::chrono::duration_cast<std::chrono::milliseconds>(    now -     moment ).count() << "[ms]" << std::endl;
}











void Sequence::run() 
{
    std::chrono::steady_clock::time_point begin;

    //local domain sizes
    int Nx = local_end_x, Ny = local_end_y, Nz = local_end_z;
    int domain_size = (Nx + 1) * (Ny + 1) * (Nz + 1);


    Array3D a0 = analitical(0 * dt);
    Array3D u_prev = U0(a0);

    exchange_borders(cart_comm, u_prev);
    Array3D u_curr = U1(u_prev);
    
    Array3D a1 = analitical(1 * dt);
    Array3D u_next(Nx, Ny, Nz);



    h_u_prev.resize(domain_size); 
    h_u_curr.resize(domain_size);


    //Moving data to host arrays
    for (int i = 0; i <= Ny; ++i)
        for (int j = 0; j <= Nx; ++j)
            for (int k = 0; k <= Nz; ++k){
                h_u_prev[ind(i, j, k)] = u_prev(i, j, k);
                h_u_curr[ind(i, j, k)] = u_curr(i, j, k);
            }

    //Cuda needs two arrays in the device memory to be filled with data
    d_u_prev.resize(domain_size); 
    d_u_curr.resize(domain_size);
    d_u_next.resize(domain_size);

    d_u_prev = h_u_prev;
    d_u_curr = h_u_curr;



    //-------------------------------------


    cudaDeviceSynchronize();
    MPI_Barrier(cart_comm);
    
    time_from_that_moment("got to cycle", begin);


    begin = std::chrono::steady_clock::now();



    for (int t = 2; t <= K; ++t)
    {
        double *d_u_prev_ptr = thrust::raw_pointer_cast(d_u_prev.data());
        double *d_u_curr_ptr = thrust::raw_pointer_cast(d_u_curr.data());
        double *d_u_next_prev_ptr = thrust::raw_pointer_cast(d_u_next.data());
        double *d_a0_ptr;


        if(GPU && CPU){
        compare(d_u_prev, u_prev);
        compare(d_u_curr, u_curr);
        }

        std::cout << "********************************************" << std::endl;

        time_from_that_moment("started borders", begin);

        if(CPU){
        exchange_borders( cart_comm, u_curr);
        }


        if(GPU){
        exchange_borders( d_u_curr_ptr);
        cudaDeviceSynchronize();
        }


        if(GPU && CPU){
        compare(d_u_prev, u_prev);
        compare(d_u_curr, u_curr);  
        cudaDeviceSynchronize(); 
        }

        std::cout << "____________________________________________" << std::endl;



        //Есть проблема! что с правым буфером? в 1 121 1
        //проверим!
        if(DEBUG_KERNELS){
        print<<<dim3((Nz + 1) / 16 + 1, Ny + 1, (Nx + 1) / 32 + 1), dim3(32, 1, 16)>>>(device_recv_buffer_x_right, N );
        cudaDeviceSynchronize();
        }



        
        time_from_that_moment("ended borders", begin);

        time_from_that_moment("started kernel", begin);

        if(GPU){

        //cudaDeviceSynchronize();
        //printf("kernel time_step  start output: rb_xr[1,128,1] = %.15f\n", device_recv_buffer_x_right[k_k * (Ny + 1) + k_i]);
        //cudaDeviceSynchronize();

         /*   time_step << <dim3((Nz + 1) / 16 + 1, Ny + 1, (Nx + 1) / 32 + 1), dim3(32, 1, 16) >> > (d_u_prev_ptr, d_u_curr_ptr, d_u_next_prev_ptr,
                hx, hy, hz,
                N,
                Nx, Ny, Nz,
                Lx, Ly, Lz,
                global_start_y,

                device_recv_buffer_x_right,
                device_recv_buffer_x_left,
                device_recv_buffer_y_right,
                device_recv_buffer_y_left,
                device_recv_buffer_z_right,
                device_recv_buffer_z_left,
                dt);*/

         time_step_inside <<<dim3((Nz + 1) / 32 + 1, Ny + 1, (Nx + 1) / 32 + 1), dim3(32, 1, 32) >> > (d_u_prev_ptr, d_u_curr_ptr, d_u_next_prev_ptr,
                hx, hy, hz,
                N,
                Nx, Ny, Nz,
                Lx, Ly, Lz,
                dt);

         cudaDeviceSynchronize();

         time_from_that_moment("kernel 1st part ", begin);

        time_step_border<<<dim3((Nz + 1) / 16 + 1, Ny + 1, (Nx + 1) / 32 + 1), dim3(32, 1, 16)>>>( d_u_prev_ptr, d_u_curr_ptr, d_u_next_prev_ptr, 
                                                                            hx, hy, hz,
                                                                            N,
                                                                            Nx, Ny, Nz,
                                                                            Lx, Ly, Lz,
                                                                            global_start_y, 
                                                                            
                                                                            device_recv_buffer_x_right,
                                                                            device_recv_buffer_x_left,
                                                                            device_recv_buffer_y_right,
                                                                            device_recv_buffer_y_left,
                                                                            device_recv_buffer_z_right,
                                                                            device_recv_buffer_z_left,
                                                                            dt );

                                                                            

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        time_from_that_moment("kernel 2nd part ", begin);

        }


        if(CPU){
            for (int k = 0; k <= Nz; ++k)
                for (int i = 0; i <= Ny; ++i)
                    for (int j = 0; j <= Nx; ++j)
                    {
                        if ((i + global_start_y != 0) && (i + global_start_y != N))
                        {
                            double laplacian_value =  laplacian(u_curr, i, j, k);


                            if(i == k_i && j == k_j && k == k_k ){
                                printf("host output: laplacian = %.15f \n", laplacian_value);
                                printf("host output: ucurr = %.15f \n", u_curr(i, j, k));
                                printf("host output: uprev = %.15f \n", u_prev(i, j, k));
                            }

                            double value;
                            value = 2.0 * u_curr(i, j, k) - u_prev(i, j, k) + dt * dt * laplacian_value;


                            u_next.set(value, i, j, k);
                        }
                        else
                            u_next.set(0.0, i, j, k);

                        if(i == k_i && j == k_j && k == k_k ){
                            printf("host output: %.15f \n", u_next( i, j, k));
                        }

                    }
            }

        cudaDeviceSynchronize();
        time_from_that_moment("kernel", begin);

        if(GPU && CPU){
        compare(d_u_next, u_next);
        }
        
        std::cout << "================================================" << std::endl;
        //break;

        d_a0 = std::move(d_u_prev);
        d_u_prev = std::move(d_u_curr);
        d_u_curr = std::move(d_u_next);
        d_u_next = std::move(d_a0);

        a0 = std::move(u_prev);
        u_prev = std::move(u_curr);
        u_curr = std::move(u_next);
        u_next = std::move(a0);



        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    }
    
    time_from_that_moment("exited cycle", begin);


    //Moving back
    h_u_curr = d_u_curr;

    //writing to old formatted array

    if(GPU){
    for (int i = 0; i <= Ny; ++i)
        for (int j = 0; j <= Nx; ++j)
            for (int k = 0; k <= Nz; ++k){
                 u_curr.set(h_u_curr[ind(i, j, k)], i, j, k);
            }
    }

    const Array3D at = analitical(K * dt);

    calculateError(cart_comm, at, u_curr, K, cart_rank,
                   global_start_x, global_end_x,
                   global_start_y, global_end_y,
                   global_start_z, global_end_z);

    time_from_that_moment("exited function", begin);

}
