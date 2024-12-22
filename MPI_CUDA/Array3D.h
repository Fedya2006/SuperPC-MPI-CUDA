#pragma once
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include "string"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>



#define ind(i,j,k) ((k) * (Nx+1) * (Ny+1) + (i) * (Nx+1) + (j))
const int k_i = 1, k_j = 1, k_k = 0; 




constexpr double threshold_value = 1e-15;

class Array3D
{
public:
    double* data = nullptr;
    int Nx, Ny, Nz;


    explicit Array3D(const int Nx, const int Ny, const int Nz) : Nx(Nx), Ny(Ny), Nz(Nz)
    {
        data = static_cast<double *>(malloc((Nx + 1) * (Ny + 1) * (Nz + 1) * sizeof(double)));

        for (int i = 0; i <= Ny; ++i)
            for (int j = 0; j <= Nx; ++j)
                for (int k = 0; k <= Nz; ++k)
                    set(0.0, i, j, k);

    }


    ~Array3D()
    {
        if (this->data != nullptr)
            free(data);
    }


    double& operator()(int i, int j, int k) const
    {
        return this->data[k * (Nx+1) * (Ny+1) + i * (Nx+1) + j];
    }

    Array3D& operator=(Array3D&& other) noexcept
    {

        if (this->data != nullptr)
            free(this->data);


        this->data = other.data;
        this->Nx = other.Nx;
        this->Ny = other.Ny;
        this->Nz = other.Nz;

        other.data = nullptr;
        other.Nx = 0;
        other.Ny = 0;
        other.Nz = 0;

        return *this;
    }


    Array3D(Array3D&& other)  noexcept : data(other.data), Nx(other.Nx), Ny(other.Ny), Nz(other.Nz)
    {
        other.data = nullptr;
        other.Nx = 0;
        other.Ny = 0;
        other.Nz = 0;
    }


    void set(double value, const int i, const int j, const int k) const
    {
        if (fabs(value) < threshold_value)
            value = 0.0;

        if (this->data != nullptr)
            this->data[k * (Nx+1) * (Ny+1) + i * (Nx+1) + j] = value;
        else
            throw std::invalid_argument("Undefined array pointer");
    }

};



