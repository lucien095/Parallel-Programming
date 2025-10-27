#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define PAD 8

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{
  // initialize vertex weights to uniform probability.
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  #pragma omp parallel for
  for(int i = 0; i < numNodes; ++i){
    solution[i] = equal_prob;
  }
  //Double precision scores are used to avoid underflow for large graphs
  int nthreads = omp_get_max_threads();
  //Avoid false sharing
  double local_diff[nthreads][PAD], local_no_outgoing_sum[nthreads][PAD];

  double *score_new = new double[numNodes];
  double global_diff, global_no_outgoing_sum;
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }
   */
  bool converged = false;
  while(!converged){
    // initial global variable
    global_diff = 0.0;
    global_no_outgoing_sum = 0.0;
    // initial local variable
    #pragma omp parallel for
    for(int i = 0; i < nthreads; ++i) {
      local_diff[i][0] = 0.0;
      local_no_outgoing_sum[i][0] = 0.0;
    }

    // Calculate the sum of no outgoing node and intermediate score
    #pragma omp parallel 
    {
      int thread_id = omp_get_thread_num();
      int n_threads = omp_get_num_threads();

      int n_outgoing;
      #pragma omp for schedule(dynamic, 1024)
      for(int i = 0; i < numNodes; ++i) {
        n_outgoing = outgoing_size(g, i);
        if(n_outgoing == 0) {
            local_no_outgoing_sum[thread_id][0] += solution[i];
        }else {
            score_new[i] = solution[i] / n_outgoing; 
        }
      }
    }

    #pragma omp parallel for reduction(+:global_no_outgoing_sum)
    for(int i = 0; i < nthreads; ++i){
      global_no_outgoing_sum += local_no_outgoing_sum[i][0];
    }
    global_no_outgoing_sum *= damping / numNodes;

    #pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      int n_threads = omp_get_num_threads();
      double score_new_sum;
      
      #pragma omp for schedule(dynamic, 1024)
      for(int i = 0; i < numNodes; ++i){
        const Vertex *start = incoming_begin(g, i);
        const Vertex *end = incoming_end(g, i);
        score_new_sum = 0.0;
      
        for(const Vertex *v = start; v != end; ++v){
            score_new_sum += score_new[*v];
        }
        score_new_sum = (damping*score_new_sum) + (1.0-damping)/numNodes + global_no_outgoing_sum;

        local_diff[thread_id][0] += std::abs(solution[i] - score_new_sum);
        solution[i] = score_new_sum;
      }
    }

    #pragma omp parallel for reduction(+:global_diff)
    for(int i = 0; i < nthreads; ++i){
      global_diff += local_diff[i][0];
    }

    converged = (global_diff < convergence) ? true : false;
  }
  delete [] score_new;
}
