#include <omp.h>
#include <string>
#include <iostream>
#include "Sequence.h"
#include <mpi.h>
#include <locale>

void calculate_global_indices(int global_N,
    const int dims[3], const int coords[3],
    int& global_start_x, int& global_end_x,
    int& global_start_y, int& global_end_y,
    int& global_start_z, int& global_end_z)
{
    int block_size_x = (global_N + 1) / dims[0] + (coords[0] < ((global_N + 1) % dims[0]));
    int block_size_y = (global_N + 1) / dims[1] + (coords[1] < ((global_N + 1) % dims[1]));
    int block_size_z = (global_N + 1) / dims[2] + (coords[2] < ((global_N + 1) % dims[2]));

    global_start_x = coords[0] * ((global_N + 1) / dims[0]) +
        ((coords[0] < ((global_N + 1) % dims[0])) ? coords[0] : (global_N + 1) % dims[0]);

    global_start_y = coords[1] * ((global_N + 1) / dims[1]) +
        ((coords[1] < ((global_N + 1) % dims[1])) ? coords[1] : (global_N + 1) % dims[1]);

    global_start_z = coords[2] * ((global_N + 1) / dims[2]) +
        ((coords[2] < ((global_N + 1) % dims[2])) ? coords[2] : (global_N + 1) % dims[2]);

    global_end_x = global_start_x + block_size_x - 1;
    global_end_y = global_start_y + block_size_y - 1;
    global_end_z = global_start_z + block_size_z - 1;

}



int main(int argc, char **argv)
{
    setbuf(stdout, NULL);

    int K = 20;
    double T = 0.022;
    double Lx = 1;
    double Ly = 1;
    double Lz = 1;
    int N = 128;


    if (argc >= 3)
    {

    	N = std::stoi(argv[1]);
    	int mode = std::stoi(argv[2]);

        std::cout << "N = " << N << std::endl;
        std::cout << "mode = " << mode << std::endl;

    	if (mode == 1)
        {
            Lx = 1.0;
            Ly = 1.0;
            Lz = 1.0;
        }

       if (mode == 2)
       {
            Lx = M_PI;
            Ly = M_PI;
            Lz = M_PI;
       }

    }


    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Создание 3D декартовой топологии
    MPI_Comm cart_comm;
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(world_size, 3, dims);
    
    std::cout << "Dims: [" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]" << std::endl;



    int periods[3] = {1, 0, 1};
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);


    /* [!] Неоптимальность: используются пересылки по старым MPI-ранкам
                                                 Можно переупорядочить. */

    // Получение координат процесса в 3D сетке процессов
    int coords[3];
    MPI_Cart_coords(cart_comm, world_rank, 3, coords);

    // Вычисление глобальных границ для каждого процесса
    int global_start_x, global_end_x, global_start_y, global_end_y, global_start_z, global_end_z;


    calculate_global_indices(N, dims, coords,
                             global_start_x, global_end_x,
                             global_start_y, global_end_y,
                             global_start_z, global_end_z);


    auto solver = Sequence(T, K, N, Lx, Ly, Lz,
                           global_start_x, global_end_x,
                           global_start_y, global_end_y,
                           global_start_z, global_end_z, cart_comm);


    printf("proc domain: [%d, %d] [%d, %d], [%d, %d]\n", global_start_x, global_end_x, 
                                                        global_start_y, global_end_y,
                                                        global_start_y, global_end_y );


    double time1 =  MPI_Wtime();

    solver.run();

    double time2 =  MPI_Wtime();

    double local_delta_time = time2 - time1;

    double global_delta_time = 0.0;

    MPI_Reduce(&local_delta_time, &global_delta_time, 1, MPI_DOUBLE, MPI_MAX,
               0, cart_comm);

    if (world_rank == 0)
        std::cout << "Elapsed time: " << global_delta_time << std::endl;

    MPI_Finalize();
    return 0;
}
