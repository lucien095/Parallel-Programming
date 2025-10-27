#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <iostream>

#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
//#define VERBOSE true

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// -----------------------------------------------------------------------------------------------------
// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighbouring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{   
    int new_frontier_distance = distances[frontier->vertices[0]] + 1;
    std::vector<int> partial_new_frontier[omp_get_max_threads()];

    #pragma omp parallel for schedule(monotonic: dynamic, 1024)
    for (int i=0; i<frontier->count; ++i) {

        Vertex node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbours to the new frontier
        for (int neighbour=start_edge; neighbour<end_edge; ++neighbour) {
            Vertex outgoing = g->outgoing_edges[neighbour];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                distances[outgoing] = new_frontier_distance;
                partial_new_frontier[omp_get_thread_num()].push_back(outgoing);
            }
        }
    }

    for (int i = 0; i < omp_get_max_threads(); ++i)
    {
        for (int child_v : partial_new_frontier[i])
        {
            int index = new_frontier->count++;
            new_frontier->vertices[index] = child_v;
        }    
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    bool* visited = (bool*) malloc(graph->num_nodes * sizeof(bool));

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; ++i) {
        sol->distances[i] = NOT_VISITED_MARKER;
        visited[i] = false;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        top_down_step(graph, frontier, new_frontier, sol->distances);
        

       for (size_t i = 0; i < new_frontier->count; ++i) {
            int index = new_frontier->vertices[i];
            visited[index] = true;
        }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(list1.vertices);free(list2.vertices);free(visited);
}

// -----------------------------------------------------------------------------------------------------
void bottom_up_step(
    Graph g,
    bool* visited,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{   
    int new_frontier_distance = distances[frontier->vertices[0]] + 1;
    std::vector<int> partial_new_frontier[omp_get_max_threads()];

    #pragma omp parallel for schedule(monotonic: dynamic, 1024)
    for (Vertex i = 0; i < g->num_nodes; ++i) {
        if (!visited[i]){
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[i + 1];

            for (int neighbour = start_edge; neighbour < end_edge; neighbour++) {
                Vertex incoming = g->incoming_edges[neighbour];
                if (visited[incoming]){
                    distances[i] = new_frontier_distance;
                    partial_new_frontier[omp_get_thread_num()].push_back(i);
                    break;
                }
                
            }
        }
    }

    for (int i = 0; i < omp_get_max_threads(); ++i){
        for (int v : partial_new_frontier[i]){
            int index = new_frontier->count++;
            new_frontier->vertices[index] = v;
        }    
    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier= &list1;
    vertex_set* new_frontier= &list2;

    bool* visited = (bool*) malloc(graph->num_nodes * sizeof(bool));

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; ++i) {
        sol->distances[i] = NOT_VISITED_MARKER;
        visited[i] = false;
    }

    // setup frontier with the root node
    frontier->vertices[0] = ROOT_NODE_ID;
    frontier->count++;
    visited[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        vertex_set_clear(new_frontier);
        bottom_up_step(graph, visited, frontier, new_frontier, sol->distances);
        
        for (size_t i = 0; i < new_frontier->count; ++i) {
            int index = new_frontier->vertices[i];
            visited[index] = true;
        }

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();

        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier= new_frontier;
        new_frontier= tmp;
    }

    free(list1.vertices);free(list2.vertices);free(visited);
}

// -----------------------------------------------------------------------------------------------------

void bfs_hybrid(Graph graph, solution* sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier= &list1;
    vertex_set* new_frontier= &list2;

    bool* visited = (bool*) malloc(graph->num_nodes * sizeof(bool));

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; ++i) {
        sol->distances[i] = NOT_VISITED_MARKER;
        visited[i] = false;
    }

    // setup frontier with the root node
    frontier->vertices[0] = ROOT_NODE_ID;
    frontier->count++;
    visited[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
        vertex_set_clear(new_frontier);

        if ((double)frontier->count / graph->num_nodes <= 0.03) {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        } else {
            bottom_up_step(graph, visited, frontier, new_frontier, sol->distances);
        }
        
        for (size_t i = 0; i < new_frontier->count; ++i) {
            int index = new_frontier->vertices[i];
            visited[index] = true;
        }


        // swap pointers
        vertex_set* tmp = frontier;
        frontier= new_frontier;
        new_frontier= tmp;
    }

    free(list1.vertices);free(list2.vertices);free(visited);
}