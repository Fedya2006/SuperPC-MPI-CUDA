#include "Array3D.h"

double diff(double a, double b){
    double d = a - b;

    if(d > 0)
        return d;
    else 
        return -d;

}

void compare(const thrust::device_vector<double>& d_u, Array3D& arr)
{
    thrust::host_vector<double> u  = d_u;

    cudaDeviceSynchronize();

    int Nx = arr.Nx;
    int Ny = arr.Ny;
    int Nz = arr.Nz;

    std::cout << "COMPARE " << Nx << " " << Ny << " " << Nz << std::endl; 

    int cnt = 0;
    
    for (int i = 0; i <= Ny; ++i)
        for (int j = 0; j <= Nx; ++j)
            for (int k = 0; k <= Nz; ++k){

                    if(i == k_i && j == k_j && k == k_k ){
                        printf("COMPARE: %.15f %.15f \n", u[ind(i, j, k)], arr( i, j, k));
                    }

                if( diff(u[ind(i, j, k)], arr(i,j,k)) > 0.0000000001 ){
                    std::cout << "difference was spot in " << i << " " << j << " " << k << std::endl; 
                    std::cout << "diff = " << diff(u[ind(i, j, k)], arr(i,j,k)) << std::endl;
                    cnt++;  
                }
                    
                if(cnt > 0)
                    throw std::runtime_error("bad");           
                }

}


