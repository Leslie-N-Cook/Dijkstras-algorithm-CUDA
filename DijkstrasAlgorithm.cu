/*
Leslie Cook
Parker Hagmaier
11/28/2022
High performance GPU programing 
Dr.Colmenares

PROGRAM DESCRIPTION:
In this program we created a parallel version of the famous Dijkstra's Shortest
Path Algorithm
Our requirments were that we use shared and global memory and that 
our program be compatable of an input size of 4k,8k, and 16k
The goal of the algorithm is to find the shortest path from a source node
(IN this case our source node always begins at 0)
and we must find the shortest path from the source to every other node in the graph
A path is defined as a series of nodes and edges where no node or edge can repeat
Our graph is defined as an adjacency matrix where our graph is fully connected 
and contains no self loops although this is not necisary and our program should function
with both self loops and a graph that is not fully connected but in that case there
is the very off chance a node may not be reachable in which case the lengh would be
infinite which we have defined in this program as MAX_INT

INSTRUCTIONS TO RUN:
load the TACC Maverick batch script into maverick and sbatch Main
The only thing that has to be changed is the desired size of which you have 
four options size1 (default) 1024 size2 4096 and size3 8192 and size4 16384
To verify the corectness of the code a message will display at the end indicating
The sucess of the program as well as outputing the array of distances in 
alternating order between our parallel version and the sequntial version

*/


/*
HEADERS:
limits is used for INT_MAX which is our theoretical infinity
stdio.h and stdio are standard for C programs 
We used it to display our output and to use rand and srand to generate a graph
math is used in order order to get the square root of 
size in order to determine the number of nddes our matrix will have
since each graph is a square nXn matrix where n is the number of nodes

DEFINITIONS:
the sizes from size1 -> size4 are the various sizes to chose for the 
size of the input data (size of the adjacency matrix/graph)
one is for arrays of size one
lower and upper are for the limits on the matrix weight size
since we are dealing with large data we didn't want to risk the total 
weight reaching our theoretical infinity
blocks1-> blocks4 are the block sizes one can use our program will work with any
of these sizes regardless of the size of the matrix
threads is the number of threads we will have in our blocks
*/
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timer.h"
//#define size1 (1024) 
//#define size1 (4096)
//#define size1 (8192)
#define size1 (16384)
#define one (1)
#define lower (1)
#define upper (100)
#define blocks1 (1)
#define blocks2 (4)
#define blocks3 (8)
#define blocks4 (16)
#define threads (1024)


//PROTOTYPES
// FUNCTION DESCRIPTIONS WILL BE EXPLAINED BY THEIR DEFINITIONS
__global__ void FindMin(int *nodes,int *bools, int *distance, int *local_index, int *local_min);
__global__ void absoluteMin(int *edge, int *indx, int *local_min, int *local_index, int *bools);
__global__ void dijk(int *nodes, int *arr, int *bools, int *distance, int *edge, int *indx);
void check(int *arr1, int *arr2, int nodes);
void create(int *arr, int *arr2, int nodes);
void initialize(int *bools, int *dist, int *boolsD, int *distD, int nodes, int start);
int regHelper(int *distance, int *bools, int nodes);
void regDijk(int start, int *arr, int *distance,int *bools, int nodes);


//MAIN FUNCTION TO RUN PROGRAM OUR "DRIVER"
int main(){
printf("\n\nTotal number of nodes in the graph: %d\n", size1);
    /*
    CREATING variables for sequential code
    */

    //OUR STARTING NODE
    int start = 0;


    int nodes = sqrt(size1);
    int arr[size1];
    int bools[nodes];
    int dist[nodes];
      /*
    END OF CREATING variables for sequential code
    */


    /*
    CREATING variables for our Parallel Program
    */

 
   /*
    CREATING sizes we will pass to cudaMalloc so our variables 
    will have memory inside the GPU

    isize = size of array (in bytes) of size nodes
    localSize (for finding the local min) it will be of size blocksize3 which is 8
    (the above can change but we found it works well with 8)
    biggerIsize is for the size of the adjacency matrix whihc is of size size (nodes * nodes)
    oneSize is for arrays of size one 
    */
   
  const int isize = nodes*sizeof(int);
  const int localSize = blocks3 * sizeof(int);
  const int biggerIsize = size1 * sizeof(int);
  const int oneSize = one * sizeof(int);
  
  //ARRAY for our cuda Dijkstra program it will have same values as sequential
  int oneD[size1];
  //initializing array to have our gpu answers copied into
  int answers[nodes];

  // bools2 will hold true or false (1,0) for if a node has been visited  
  int bools2[nodes];
  // distance will hold the current distance of each node (defaults to INT_MAX)  
  int distance[nodes];
  //will hold the index of the smallest node for each iteration
  int index[one] = {start};
  
  //edege will hold the current weight of the smallest node for each iteration
  int edge[one] = {INT_MAX};
  //holds the number of nodes
  int nodesArr[one] = {nodes};

//create the graph (adjacency matrix) for both sequntial and parallel arrays
//they will be the same value 
 create(arr, oneD,nodes);

 //initialize our bools, distances starts, and nodes to default values
 //bools will default to False
 //dist default to INT_MAX (infinity)
 //nodes will take on the size of nodes
 //start tells it to make distance at our starting node 0 
 initialize(bools,dist,bools2,distance,nodes, start);

    //initializing arrays that we will copy in information that we declared 
    //above in order to pass it to our parallel program

  int *arrD; //will hold values of our graph
  int *distD; //will hold distances of each node
  int *boolsD; //will hold true false if node has been visited
  int *indexD; //index of closest node on each iteration
  int *edgeD; //distance to closest node on each iteration
  int *localIndx; //storing the closeset index  for each block
  int *localmin; //storing the smallest distance for each block
  int *nodesD; //will store the number of nodes

     /*
    CUDAMalloc
    Here we reserve memory int the GPU for our parameteres
    What these variables are and their sizes have already been explained
    */

  cudaMalloc((void**)&localIndx, localSize);
  cudaMalloc((void**)&localmin, localSize);
  cudaMalloc((void**)&arrD, biggerIsize);
  cudaMalloc((void**)&distD, isize);
  cudaMalloc((void**)&boolsD, isize);
  cudaMalloc((void**)&indexD, oneSize);
  cudaMalloc((void**)&edgeD, oneSize);
  cudaMalloc((void**)&nodesD, oneSize);

     /*
    cudaMemcpy
    Here we transfer over information to our pointers to pass 
    to the parallel program 
    arrD will have same info as oneD (our adjacency matrix)
    distD will have same info as distance
    indexD will have same info as index ect...
    */
  cudaMemcpy( arrD, oneD, biggerIsize, cudaMemcpyHostToDevice );
  cudaMemcpy( distD, distance, isize, cudaMemcpyHostToDevice );
  cudaMemcpy( boolsD, bools2, isize, cudaMemcpyHostToDevice );
  cudaMemcpy( indexD, index, oneSize, cudaMemcpyHostToDevice );
  cudaMemcpy( edgeD, edge, oneSize, cudaMemcpyHostToDevice );
  cudaMemcpy( nodesD, nodesArr, oneSize, cudaMemcpyHostToDevice );

    /*
    dim3
    HERe we create the various grid sizes to use for our program 
    and we have a constant block size of max threads 1024
    you can use different grid sizes depending on how 
    much input data you are using currently we are using dimGrid1
    dimBlock holds our 1024 by 1 block of threads
    */

  //dim3 blockHelper(1,1); 
  dim3 dimGrid1(1, 1);
  //dim3 dimGrid2(4, 1);
  //dim3 dimGrid3(8, 1);
  //dim3 dimGrid4(16, 1);
  dim3 dimBlock(threads, 1);
   
   /*
   CALLING OUR KERNALS
    Since Dijkstra's is a greedy algorithm and we must ensure that 
    no thread tries to run before a new smallest node has been found
    our kernal calls must be performed inside a for loop fo size node
    we pass each of our three functions dimGrid1, dimBlock
    the parameters are explained by each funcion defintion 
    */
  cudaEvent_t START, STOP;             
  float elapsed_exec;                  
  cudaEventCreate(&START);
  cudaEventCreate(&STOP);
  printf("\nRunning parallel <3\n");
  cudaEventRecord(START);
    for (int i=0;i<nodes;i++){
        FindMin<<<dimGrid1, dimBlock>>>(nodesD, boolsD,distD,localmin,localIndx);
        absoluteMin<<<dimGrid1, dimBlock>>>(edgeD, indexD,localmin,localIndx, boolsD);
        dijk<<<dimGrid1, dimBlock>>>(nodesD,arrD,boolsD,distD,edgeD,indexD);
        cudaDeviceSynchronize();
    }
  cudaEventRecord(STOP);
  cudaEventSynchronize(STOP);
  cudaEventElapsedTime(&elapsed_exec, START, STOP);        
  printf("\nGPU time (ms): %7.9f\n\n\n", elapsed_exec);
    //HERE WE COPY the results of distanceD to answers to display/check accuracy
  cudaMemcpy(answers, distD, isize, cudaMemcpyDeviceToHost);
  /*
  cudaFree:
  here we free all the data we reserved inside the gpu
  */
  cudaFree(arrD);
  cudaFree(distD);
  cudaFree(boolsD);
  cudaFree(indexD);
  cudaFree(edgeD);
  cudaFree(localIndx);
  cudaFree(localmin);
  cudaFree(nodesD);

    //calling sequential dijkstra in order to compare speed results to 
    //our parallel cuda version
  regDijk(start,arr,dist,bools,nodes);

  //check verifies our results are the same
  check(answers,dist, nodes);

    /*
  PRINT THE RESULTING distance arrays
  add the sum to verify just in case our check function
  was in question or if the resulting arrays wish to be viewed
  */
  int seqR1 = 0;
  int seqR2 = 0;
  for (int i=0;i<nodes;i++){
    printf("%d\n",answers[i]);
    seqR1 += answers[i];
    printf("%d\n",dist[i]);
    seqR2 += dist[i];
  }

  if (seqR1 == seqR2){
    printf("FINALLLLLLLLY\n");
  }
  else{
    printf("STIL A FAILURE\n");
  }

  printf("Parallel is: %d\n", seqR1);
  printf("Sequential is: %d\n", seqR2);


    return 0;
}
//END OF MAIN FUNCTION


  /*
  FUNCTION DEFINTIONS
  */


  /*
  CREATE
  description:
    given two pointers to arrays create identical oneD adjacency 
    arrays that will act as our weight values for each array
    no self loops so when i==y the result is 0
    generate random numbers using rand with a srand seed of 2 to verify results

    params:
    int *arr: pointer to a array
    int *arr2: pointer to second array to be intilized
    int nodes: number of nodes in our graph
  */
void create(int *arr, int *arr2, int nodes){
  srand(2);
  int count = 0;
  for (int i=0;i<nodes;i++){
    for (int y =0; y<nodes;y++){
      if (i !=y){
      int num = (rand() % (upper - lower + 1)) + lower;
      arr[count] = num;
      arr2[count] = num;
      count +=1;
      }
      else{
        arr[count] = 0;
        arr2[count] = 0;
        count +=1;
      } 
    }
  }
}

/*
  CHECK
  description:
    verifies the results of the parallel code and the 
    sequntial code are the same 
    When a value does not match it will print out wrong
    if all values are read wihtout a wrong value it will display
    'results matched'

    params:
    int *arr1: pointer to a array (either the sequauntial dist or parallel)
    int *arr2: pointer to a array (either the sequauntial dist or parallel)
    int nodes: number of nodes in our graph
  */
void check(int *arr1, int *arr2, int nodes){
    int right = 1;
  for (int i=0;i<nodes;i++){
    if (arr1[i] != arr2[i]){
        right = 0;
    }
  }
  if (right == 0){
  printf("WRONG\n");
  }
  else{
  printf("Results matched\n");
  }
}

/*
  initialize
  description:
    initialize both cuda parallel and cpu sequntial variables to 
    default values. All distances besides the distance of the starting
    node will be set to our theretical infinity (INT_MAX)
    both bool arrays which are really just ones and zeroes 
    will be set to 0 indicating false and therefore unvisited

    params:
    int *bools: array of boolean values for seequntial code
    int *dist: array of distances for sequntial code
    int boolsD: array of boolean values for parallel code
    int *distD: array of distances for parallel code 
    int nodes: number of nodes in the graph
    int start: the starting node
  */
void initialize(int *bools, int *dist, int *boolsD, int *distD, int nodes,int start){
    for (int i=0;i<nodes;i++){
        bools[i] = 0;
        boolsD[i] = 0;
        dist[i] = INT_MAX;
        distD[i] = INT_MAX;
    }
    dist[start] = 0;
    distD[start] = 0;
}

/*
  regHelper
  description:
    A helper function for the sequential Dijkstra function
    This function will find the closest unvisited node in the graph
    shortest is set to infinity and and any node that is univisted
    and less then the current shortest node becomes the shortest node 
    and everything else is checked against that distance
    we return the index of the shortest node

    params:
    int *distance: array holding the distance from source to each node
    int *bools: array that holds information about whether a node has been visited
    int nodes: the number of nodes we have in our graph 
    used to determine length of the for loop (how many nodes we have to check)
  */
int regHelper(int *distance, int *bools, int nodes) {
  int shortest = INT_MAX;
  int indx;
  for (int i = 0; i < nodes; i++) {
    if (distance[i] < shortest && bools[i] == 0) {
      shortest = distance[i];
      indx = i;
    }
  }
  return indx;
}

/*
  regDijk
  description:
    This function 'relaxes' the edges of the graph
    Whatever nodes that have not been seen but connect 
    from a visited vertex can have their weights updated in order to 
    reflect their current shortest path which also helps determine 
    which node will be chosen next by the helper function
    int requires a call from helper to get the next shortest node 
    and then for whatever nodes it can it will update the edge weight
    to the smallest possible value

    params:
    int start: the starting node
    int *arr: the array holding the 1D adjacency matrix
    int *distance: array holding the distance from source to each node
    int *bools: array that holds information about whether a node has been visited
    int nodes: the number of nodes we have in our graph 
    used to determine length of the for loop (how many nodes we have to check)
  */
void regDijk(int start, int *arr, int *distance,int *bools, int nodes) {
  double begin, finish, elapsed;
  printf("\nRunning sequential <3\n");
  GET_TIME(begin);
  int indx;
  distance[start] = 0;
  for (int i = 0; i < nodes; i++) {
    indx = regHelper(distance, bools,nodes);
    bools[indx] = 1;
    for (int k = 0; k < nodes; k++) {
       if (arr[nodes * indx + k] > 0 && bools[k] == 0 &&
          distance[k] > distance[indx] + arr[nodes * indx + k]) {
        distance[k] = distance[indx] + arr[nodes * indx + k];
      }
    }
  }
   GET_TIME(finish);
   elapsed = finish - begin;
   printf("\nCPU time (ms): %1.9e \n\n\n", elapsed*1000000000); 
}




/*
FindMin:
Description:
    FindMin will find the local min for each block. This function helps showcase
    the use of shared memory as sLocal_min and Slocal_index are both shared memory
    arrays
    This function will find the shortest distance for each block
    the for loop helps make our code be more scalebale and vary the block size
    to ensure that the end of the for loop is always at most of size nodes 
    we check to see if end is somehow more than the number of nodes
    sync threads is then performed to make sure that we have all the local indexes
    saved and then we will do a for loop 
    performed by a single thread to ensure that we get the all our block mins
    since we were having trouble eliminating this for loop in parallel

    params:
    nodes: number of nodes in graph
    bools: holds if node has been visited or not
    distance: holds distance for each node in the graph
    local_min: will hold the block min
    local_indx: will hold the indx of the closest node
 */
__global__ void FindMin(int *nodes, int *bools, int *distance, int *local_min, int *local_index) {
    int num_per_thread = (int) ceil(nodes[0] * 1.0/(blockDim.x * gridDim.x));

    __shared__ int Slocal_min[threads];
    __shared__ int Slocal_indx[threads];

    int start = (blockIdx.x * blockDim.x + threadIdx.x) * num_per_thread;
    int end = start + num_per_thread;
    if (end > nodes[0]) {
        end = nodes[0];
    }

    int min = INT_MAX;
    int index = -1;
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = start; i < end; i++){
        if (distance[i] < min && bools[i] == 0){
            min = distance[i];
            index = i;
        }
        __syncthreads();
    }
    Slocal_min[threadIdx.x] = min;
    Slocal_indx[threadIdx.x] = index;
    __syncthreads();

    if (threadIdx.x == 0) {
        int block_min = INT_MAX;
        int block_min_index = -1;
        for (int i = 0; i < blockDim.x; i++)
        {
            if (Slocal_min[i] < block_min && bools[i] == 0)
            {
                block_min = Slocal_min[i];
                block_min_index = Slocal_indx[i];
            }
        }
        local_min[blockIdx.x] = block_min;
        local_index[blockIdx.x] = block_min_index;
    }
}

/*
absoluteMin:
Description:
    this functin only utilizes one thread and it loops through 
    all of the local mins (min of each block)
    to find a true global minimum
    performing this in parallel was difficult and we only had sucess 
    when essentially using the GPU to perorm a sequential operation
    since the comparisons in parallel could lead to wrong values
    the function is essentially a very simple program to find
    the smallest value in an array

    params:
    edge: array that will hold the global minumum of this iteration
    indx: will hold the index of closest array for this iteration
    bools: holds if node has been visited or not
    distance: holds distance for each node in the graph
    local_min: will hold the block min
    local_indx: will hold the indx of the closest node
 */
__global__ void absoluteMin(int *edge, int *indx, int *local_min, int *local_indx, int *bools) {
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        int tempVal = INT_MAX;
        int tempIdx = -1;
        for (int i = 0; i < gridDim.x; i++)
        {
            if(tempVal > local_min[i])
            {
                tempVal = local_min[i];
                tempIdx = local_indx[i];
            }
        }
        edge[0] = tempVal;
        indx[0] = tempIdx;
        bools[indx[0]] = 1;
    }
}

/*
dijk:
Description:
    this functin relaxes the edges after a new smallest node 
    has been visited. This allows us to run the program on the next iteration
    with updated weights to find the next smallest node and it also 
    will find the current shortest distance to each node from all the possible nodes 
    we have visited so far. 
    we once again utilize start and end to ensure that a thread may have to 
    perform multiple calculations depending on how large the data set is and 
    how many blocks are used

    params:
    nodes: how many nodes are in the graph
    arr: stores the adjacency matrix with all the weights
    bools: stores if a node has been visited or not
    distance: stores the current distance to every node in the graph
    edge: the global minimum weight for the particular iteraton 
    indx: stores the gloabl indx of the minimum node
 */
__global__ void dijk(int *nodes, int *arr, int *bools, int *distance, int *edge, int *indx) {
    int num_per_thread = (int) ceil((nodes[0] * 1.0)/(blockDim.x * gridDim.x));

    int start = (blockIdx.x * blockDim.x + threadIdx.x) * num_per_thread;
    int end = start + num_per_thread;
    if (end > nodes[0]) {
        end = nodes[0];
    }

    for(int i = start; i < end; i++)
    {
        if (edge[0] + arr[indx[0] * nodes[0] + i] < distance[i] && bools[i] == 0)
        {
            distance[i] = edge[0] + arr[indx[0] * nodes[0] + i];
        }
    }
    __syncthreads();
}