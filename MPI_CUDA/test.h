void exchange_boundary_layers(MPI_Comm comm, const int rank_prev[3], const int rank_next[3],
                              std::vector<double>& vec_left_x, std::vector<double>& vec_right_x,
                              std::vector<double>& vec_left_y, std::vector<double>& vec_right_y,
                              std::vector<double>& vec_left_z, std::vector<double>& vec_right_z) {

    if (rank_prev[0] != MPI_PROC_NULL) {
        MPI_Sendrecv(vec_left_x.data(), ny * nz, MPI_DOUBLE, rank_prev[0], 1,
                     vec_right_x.data(), ny * nz, MPI_DOUBLE, rank_prev[0], 0,
                     comm, MPI_STATUS_IGNORE);
    }

    if (rank_next[0] != MPI_PROC_NULL) {
        MPI_Sendrecv(vec_right_x.data(), ny * nz, MPI_DOUBLE, rank_next[0], 0,
                     vec_left_x.data(), ny * nz, MPI_DOUBLE, rank_next[0], 1,
                     comm, MPI_STATUS_IGNORE);
    }


    if (rank_next[1] != MPI_PROC_NULL) {
        MPI_Sendrecv(vec_right_y.data(), nx * nz, MPI_DOUBLE, rank_next[1], 2,
                     vec_left_y.data(), nx * nz, MPI_DOUBLE, rank_next[1], 3,
                     comm, MPI_STATUS_IGNORE);
    }
    if (rank_prev[1] != MPI_PROC_NULL) {
        MPI_Sendrecv(vec_left_y.data(), nx * nz, MPI_DOUBLE, rank_prev[1], 3,
                     vec_right_y.data(), nx * nz, MPI_DOUBLE, rank_prev[1], 2,
                     comm, MPI_STATUS_IGNORE);
    }

    if (rank_next[2] != MPI_PROC_NULL) {
        MPI_Sendrecv(vec_right_z.data(), nx * ny, MPI_DOUBLE, rank_next[2], 4,
                     vec_left_z.data(), nx * ny, MPI_DOUBLE, rank_next[2], 5,
                     comm, MPI_STATUS_IGNORE);
    }
    if (rank_prev[2] != MPI_PROC_NULL) {
        MPI_Sendrecv(vec_left_z.data(), nx * ny, MPI_DOUBLE, rank_prev[2], 5,
                     vec_right_z.data(), nx * ny, MPI_DOUBLE, rank_prev[2], 4,
                     comm, MPI_STATUS_IGNORE);
    }
}