#include <iostream>
#include "Sequence.h"
using namespace std;

void exchange_borders(MPI_Comm cart_comm)
{
    int world_size;
    int world_rank;
    MPI_Comm_size(cart_comm, &world_size);
    MPI_Comm_rank(cart_comm, &world_rank);

    int ndims = 3;
    int dims[3], periods[3], notused[3], coords[3];

    MPI_Cart_get(cart_comm, ndims, dims, periods, notused);
    MPI_Cart_coords(cart_comm, world_rank, 3, coords);

    std::cout << "Rank " << world_rank << " coords: "
              << "[" << coords[0] << ", " << coords[1] << ", " << coords[2] << "]" << std::endl;


    int neighbors[6];
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[0], &neighbors[1]); // x- и x+
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[2], &neighbors[3]); // y- и y+
    MPI_Cart_shift(cart_comm, 2, 1, &neighbors[4], &neighbors[5]); // z- и z+


    std::cout << "Rank " << world_rank << " neighbors: "
              << "x-=" << neighbors[0] << ", x+=" << neighbors[1] << ", "
              << "y-=" << neighbors[2] << ", y+=" << neighbors[3] << ", "
              << "z-=" << neighbors[4] << ", z+=" << neighbors[5] << std::endl;


    if (neighbors[0] != MPI_PROC_NULL)
    {
        ; // Данные для отправки
        int * send_data0 = reinterpret_cast<int *>(static_cast<double *>(malloc(3 * sizeof(int))));
        send_data0[0] = coords[0];
        send_data0[1] = coords[1];
        send_data0[2] = coords[2];
        int recv_data0 [3];

        int tag = 0; // Тег для x- направления

        int ierr = MPI_Sendrecv(send_data0, 3, MPI_INT, neighbors[0], tag,
                                recv_data0, 3, MPI_INT, neighbors[0], tag,
                     cart_comm, MPI_STATUS_IGNORE);

        cout << "Rank: " << world_rank << " Received recv_data0: " <<  "[" << recv_data0[0] << ", " << recv_data0[1] << ", " << recv_data0[2] << "]" << std::endl;
        if (ierr != MPI_SUCCESS)
        {
            char error_string[MPI_MAX_ERROR_STRING];
            int error_length;
            MPI_Error_string(ierr, error_string, &error_length);
            std::cout << "Rank: " << world_rank<< "   MPI Error: " << error_string << "\n\n\n";
        }

    }


    if (neighbors[1] != MPI_PROC_NULL)
    {
        MPI_Status status;
        int * send_data1 = reinterpret_cast<int *>(static_cast<double *>(malloc(3 * sizeof(int))));
        send_data1[0] = coords[0];
        send_data1[1] = coords[1];
        send_data1[2] = coords[2];
        int recv_data1[3];

        int tag = 1;

        int ierr = MPI_Sendrecv(send_data1, 3, MPI_INT, neighbors[1], tag,
                                recv_data1, 3, MPI_INT, neighbors[1], tag,
                     cart_comm, MPI_STATUS_IGNORE);

        cout << "Rank: " << world_rank << " Received recv_data0: " <<  "[" << recv_data1[0] << ", " << recv_data1[1] << ", " << recv_data1[2] << "]" << std::endl;

        if (ierr != MPI_SUCCESS)
        {
            char error_string[MPI_MAX_ERROR_STRING];
            int error_length;
            MPI_Error_string(status.MPI_ERROR, error_string, &error_length);
            std::cout << "Rank: " << world_rank<< "   MPI Error: " << error_string << "\n\n\n";
        }

    }

}


int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Comm cart_comm;
    int dims[3] = {};
    MPI_Dims_create(world_size, 3, dims);

    int periods[3] = {1, 0, 1};
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, world_rank, 3, coords);


    exchange_borders(cart_comm);

    MPI_Finalize();
    return 0;
}