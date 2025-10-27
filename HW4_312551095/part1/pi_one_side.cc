#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int fnz (long long int *gather, int size)
{
    int diff = 0;

    for (int i = 0; i < size; ++i)
       diff |= (gather[i] != 0);

    if (diff)
    {
        int res = 0;
        for (int i = 0; i < size; ++i)
        {
            if(gather[i] != 0)++res;
        }
       return(res == size-1);
    }
    return 0;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long int num_cycle = 0;
    long long int local_num_tosses = tosses / world_size;

    if (world_rank == 0)
    {
        // Master
        long long int *gather;
        MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &gather);
        *gather = 0;

        unsigned int seed = world_rank * 777;
        double x, y, f1, f2, distance;
        for (int i = 0; i < local_num_tosses; ++i) {
            f1 = (double)rand_r(&seed) / RAND_MAX; /*rand() is much slower than rand_r()*/
            x = -1 + f1 * 2;
            f2 = (double)rand_r(&seed) / RAND_MAX;
            y = -1 + f2 * 2;
            distance = x * x + y * y;
            if (distance <= 1)++num_cycle;
        }

        MPI_Win_create(gather, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        int ready = 0;
        while (!ready)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            ready = fnz(gather, world_size);
            MPI_Win_unlock(0, win);
        }

        for(int i=0; i<world_size; i++)num_cycle += gather[i];

        MPI_Win_free(&win);
        MPI_Free_mem(gather);
    }
    else
    {
        // Workers
        long long int local_num_cycle = 0;
        unsigned int seed = world_rank * time(0);
        double x, y, f1, f2, distance;
        for (int i = 0; i < local_num_tosses; ++i) {
            f1 = (double)rand_r(&seed) / RAND_MAX; /*rand() is much slower than rand_r()*/
            x = -1 + f1 * 2;
            f2 = (double)rand_r(&seed) / RAND_MAX;
            y = -1 + f2 * 2;
            distance = x * x + y * y;
            if (distance <= 1)++local_num_cycle;
        }

        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&local_num_cycle, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG, win);
        MPI_Win_unlock(0, win);

        MPI_Win_free(&win);
    }

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * num_cycle / (double)tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}