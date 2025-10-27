#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long int num_cycle = 0;
    long long int local_num_cycle = 0;
    long long int local_num_tosses = tosses / world_size;
    unsigned int seed = world_rank * 777;
    double x, y, f1, f2, distance;
    for (int i = 0; i < local_num_tosses; ++i) {
        f1 = (double)rand_r(&seed) / RAND_MAX; /*rand() is much slower than rand_r()*/
        x = -1 + f1 * 2;
        f2 = (double)rand_r(&seed) / RAND_MAX;
        y = -1 + f2 * 2;
        distance = x * x + y * y;
        if (distance <= 1)++local_num_cycle;
    }

    // TODO: use MPI_Reduce
    MPI_Reduce(&local_num_cycle, &num_cycle, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
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
