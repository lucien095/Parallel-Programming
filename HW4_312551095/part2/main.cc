#include <mpi.h>
#include <cstdio>

// *********************************************
// ** ATTENTION: YOU CANNOT MODIFY THIS FILE. **
// *********************************************

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0){
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);

        int n = *n_ptr;
        int m = *m_ptr;
        int l = *l_ptr;
        *a_mat_ptr = (int *)malloc(n * m * sizeof(int));
        *b_mat_ptr = (int *)malloc(m * l * sizeof(int));

        // construct matrix a
        for (int i = 0; i < n; ++i){
            for (int j = 0; j < m; ++j){
                scanf("%d", *a_mat_ptr + i * m + j);
            }
        }
        // construct matrix b
        for (int i = 0; i < m; ++i){
            for (int j = 0; j < l; ++j){
                scanf("%d", *b_mat_ptr + i * l + j);
            }
        }
    }
}
// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0){
        for (int d = 1; d < world_size; ++d){
            MPI_Send(&n, 1, MPI_INT, d, 1, MPI_COMM_WORLD);
        }
        int *ab_mat;
        ab_mat = (int *)malloc(sizeof(int) * n * l);

        int n_workers, master_rows, worker_rows, remain_rows, real_worker_rows, offset;
        if (n < world_size){
            n_workers = n;
        }
        else{
            n_workers = world_size;
        }
        remain_rows = n % n_workers;
        worker_rows = n / n_workers;
        if(remain_rows == 0){
            master_rows = worker_rows;
        }
        else{
            master_rows = worker_rows + 1;
        }
        offset = master_rows;

        for (int d = 1; d < n_workers; ++d){
            if(d < remain_rows){
                real_worker_rows = worker_rows + 1;
            }else {
                real_worker_rows = worker_rows;
            }
            
            MPI_Send(&m, 1, MPI_INT, d, 1, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, d, 1, MPI_COMM_WORLD);
            MPI_Send(&offset, 1, MPI_INT, d, 1, MPI_COMM_WORLD);
            MPI_Send(&real_worker_rows, 1, MPI_INT, d, 1, MPI_COMM_WORLD);
            
            MPI_Send(a_mat + offset * m, real_worker_rows * m, MPI_INT, d, 1, MPI_COMM_WORLD);
            MPI_Send(b_mat, m * l, MPI_INT, d, 1, MPI_COMM_WORLD);

            offset += real_worker_rows;
        }

        for (int i = 0; i < master_rows; ++i){
            for (int j = 0; j < l; ++j){
                ab_mat[i * l + j] = 0;
                for (int k = 0; k < m; ++k){
                    ab_mat[i * l + j] += a_mat[i * m + k] * b_mat[l * k + j];
                }
            }
        }

        MPI_Status status;
        for (int s = 1; s < n_workers; ++s){
            MPI_Recv(&offset, 1, MPI_INT, s, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&real_worker_rows, 1, MPI_INT, s, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(ab_mat + offset * l, real_worker_rows * l, MPI_INT, s, 2, MPI_COMM_WORLD, &status);
        }
        for (int i = 0; i < n; ++i){
            for (int j = 0; j < l; ++j){
                printf("%d ", ab_mat[i * l + j]);
            }
            printf("\n");
        }
        free(ab_mat);
    }else if (world_rank > 0){
        int n, m, l, offset, real_worker_rows;

        MPI_Status status;
        MPI_Recv(&n, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&m, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&l, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&real_worker_rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        
        int *a = (int *)malloc(sizeof(int) * real_worker_rows * m);
        int *b = (int *)malloc(sizeof(int) * m * l);
        int *c = (int *)malloc(sizeof(int) * real_worker_rows * l);

        MPI_Recv(a, real_worker_rows * m, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(b, m * l, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        for (int i = 0; i < real_worker_rows; ++i){
            for (int j = 0; j < l; ++j){
                c[i * l + j] = 0;
                for (int k = 0; k < m; ++k){
                    c[i * l + j] += a[i * m + k] * b[l * k + j];
                }
            }
        }
        // send result to 0
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&real_worker_rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(c, real_worker_rows * l, MPI_INT, 0, 2, MPI_COMM_WORLD);
        free(a);free(b);free(c);
    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0){
        free(a_mat);
        free(b_mat);
    }
}

int main () {
    int n, m, l;
    int *a_mat, *b_mat;

    MPI_Init(NULL, NULL);
    double start_time = MPI_Wtime();

    construct_matrices(&n, &m, &l, &a_mat, &b_mat);
    matrix_multiply(n, m, l, a_mat, b_mat);
    destruct_matrices(a_mat, b_mat);

    double end_time = MPI_Wtime();
    MPI_Finalize();
    printf("MPI running time: %lf Seconds\n", end_time - start_time);

    return 0;
}
