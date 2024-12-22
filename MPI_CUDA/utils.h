#pragma once
#include "Array3D.h"
#include <iostream>
#include <mpi.h>
#include <chrono>

void printArray(const Array3D& array);

void printSlice(const Array3D& array, int k);

double calculateError(MPI_Comm  cart_comm, const Array3D& u_num,
                      const Array3D& u_exact, int iter, int rank,
                      int global_start_x, int global_end_x,
                      int global_start_y, int global_end_y,
                      int global_start_z, int global_end_z);



