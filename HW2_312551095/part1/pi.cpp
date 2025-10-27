# include <iostream>
# include <cstdlib>
# include <ctime>
# include <math.h>
#include <pthread.h>

using namespace std;

struct arg_in_thread{
    long long int start;
    long long int end;
    long long int *num_cycle;
};

//
pthread_mutex_t mutex;

void* pi(void *args);

int main(int argc, char** argv){
    long long int num_of_thread, num_toss, per_parse_toss_size, *num_cycle;

    // read and parse arguments
    num_of_thread = atoll(argv[1]);
    num_toss = atoll(argv[2]);
    per_parse_toss_size = floor(num_toss / num_of_thread);

    // Setting threads
    pthread_t threads[num_of_thread]; //create thread

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    arg_in_thread arg[num_of_thread];

    // Initialize mutex lock
    pthread_mutex_init(&mutex, NULL);

    num_cycle = (long long int *) malloc(sizeof(*num_cycle));
    *num_cycle = 0; // Init the number of hit as zero

    for (long long int i = 0;i < num_of_thread; ++i) {
        /* Set argueemnts to each thread */
        arg[i].start = per_parse_toss_size * i;
        arg[i].end = per_parse_toss_size * (i+1);
        arg[i].num_cycle = num_cycle;
        // Create a new thread and run pi() function with correspoding arg[i] arguments
        pthread_create(&threads[i], &attr, pi, (void *) &arg[i]);
    }

    /* Free attribute*/
    pthread_attr_destroy(&attr);
    
    /* Wait for the other threads*/
    void *status;
    for (long long int i = 0;i < num_of_thread; ++i) {
        pthread_join(threads[i], &status);
    }

    pthread_mutex_destroy(&mutex);

    double pi = 4 * ((*num_cycle) / (double)num_toss);
    printf("%.8lf\n", pi);

    return 0;
}

void* pi(void *args){
    arg_in_thread *arg = (arg_in_thread *) args;
    long long int start = arg->start;
    long long int end = arg->end;
    long long int *num_cycle = arg->num_cycle;
    long long int local_num_cycle = 0;

    unsigned int seed = 777;
    double x, y, f1, f2, distance;
    for (int i = start; i < end; ++i){
        f1 = (double)rand_r(&seed) / RAND_MAX; /*rand() is much slower than rand_r()*/
        x = -1 + f1 * 2;
        f2 = (double)rand_r(&seed) / RAND_MAX;
        y = -1 + f2 * 2;
        distance = x * x + y * y;
        if (distance <= 1){
            ++local_num_cycle;
        }
    }
    
    pthread_mutex_lock(&mutex);
    *num_cycle += local_num_cycle;
    pthread_mutex_unlock(&mutex);

    pthread_exit((void *)0);
}