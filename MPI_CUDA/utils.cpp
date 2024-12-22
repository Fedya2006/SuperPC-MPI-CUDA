#include "utils.h"


void printArray(const Array3D& array)
{
    for (int k = 0; k <= array.Nz; ++k)
    {
        std::cout << "Slice k = " << k << ":\n";
        for (int i = 0; i <= array.Ny; ++i)
        {
            for (int j = 0; j <= array.Nx; ++j)
            {
                std::cout << array(i, j, k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}


void printSlice(const Array3D& array, int k)
{
    std::cout << "Slice k = " << k << ":\n";
    for (int i = 0; i <= array.Ny; ++i)
    {
        for (int j = 0; j <= array.Nx; ++j)
            std::cout << array(i, j, k) << " ";

        std::cout << std::endl;
    }
    std::cout << std::endl;

}


double calculateError(MPI_Comm  cart_comm, const Array3D& u_num, const Array3D& u_exact, int iter, int rank,
                      int global_start_x, int global_end_x,
                      int global_start_y, int global_end_y,
                      int global_start_z, int global_end_z)
{
    int i_max = 0, j_max = 0, k_max = 0;

    double max_error = 0.0;

    /*
    std::cout << "calculateError" << " rank: " << rank
              << "  X: [" << local_start_x << ", " << local_end_x << "], "
              << "Y: [" << local_start_y << ", " << local_end_y << "], "
              << "Z: [" << local_start_z << ", " << local_end_z << "]"
              << std::endl;
    */


    for (int i = 0; i <= u_exact.Ny; i++)
        for (int j = 0; j <= u_exact.Nx; j++)
            for (int k = 0; k <= u_exact.Nz; k++)
            {
                double error = fabs(u_num(i, j, k) - u_exact(i, j, k));

                if (error > max_error)
                {
                    //std::cout << "error " << i_max << " " << j_max << " " << k_max << " " << error << std::endl;

                    max_error = error;
                    i_max = i;
                    j_max = j;
                    k_max = k;
                }
            }


    int world_size, world_rank;
    MPI_Comm_size(cart_comm, &world_size);
    MPI_Comm_rank(cart_comm, &world_rank);

    double global_error;

    MPI_Reduce(&max_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (world_rank == 0)
    {
        printf("Rank: %d | Iteration: %d | Max error: %.4e \n", world_rank, iter, global_error);
    }

    return max_error;
}