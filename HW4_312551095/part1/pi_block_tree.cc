#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

using namespace std;

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
    MPI_Status status;

    long long int local_num_tosses = tosses / world_size;

    // TODO: binary tree redunction
    unsigned int seed = world_rank * 777;
    long long int local_num_cycle = 0;
    double x, y, f1, f2, distance;
    for (int i = 0; i < local_num_tosses; ++i) {
        f1 = (double)rand_r(&seed) / RAND_MAX; /*rand() is much slower than rand_r()*/
        x = -1 + f1 * 2;
        f2 = (double)rand_r(&seed) / RAND_MAX;
        y = -1 + f2 * 2;
        distance = x * x + y * y;
        if (distance <= 1)++local_num_cycle;
    }

    int pow = 0;
    int index = world_rank;
    while(index % 2 == 0)
    {
        int source = (index + 1) * (1 << pow);
        long long int recv_num_cycle = 0;
        MPI_Recv(&recv_num_cycle, 1, MPI_LONG_LONG, source, 0, MPI_COMM_WORLD, &status);
        local_num_cycle += recv_num_cycle;

        ++pow;
        index = world_rank / (1 << pow);
        if(pow == log2(world_size))break;
    }

    if(world_rank > 0 )
    {
        int dest = (index - 1) * (1 << pow);
        MPI_Send(&local_num_cycle, 1, MPI_LONG_LONG, dest, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 * local_num_cycle / (double)tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
