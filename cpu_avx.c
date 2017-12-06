/*
	CPU BenchMarking with AVX
	Author: Shruti Gupta and Sagar Mane
*/

//Importing require packages
#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

//define max operation 
unsigned long maxOperation = 1e9;
int num_thread;
//Function declaration
void * iops(void *);
void * flops(void *);

int main(int argc, char * argv[])
{
    /*
	Operation 1: IOPS
	Operation 2: flops
    */
    int operation = atoi(argv[1]);
   
    //Number of threads
    num_thread = atoi(argv[2]);
     
    //Initialize start and end time
    struct timeval startTime;
    struct timeval endTime;
    
    //Initialize no. of threads array
	pthread_t thread[num_thread];
    
    //Get start time of the operations
    gettimeofday(&startTime, NULL);

    int i;
	for(i = 0; i < num_thread; i++)
	{
		switch(operation)
		{
		//Thread for integer operations
		case 1:pthread_create(&thread[i], NULL, iops, NULL); 
			break;
		//Thread for floating operations
		case 2:pthread_create(&thread[i], NULL, flops, NULL); 
			break;
		//Default for error
		default: printf("Error\n");
		}
		 
	}
    for (i = 0; i < num_thread; i++) 
{
        pthread_join(thread[i], NULL);
}

//Get end time of operation execution
    gettimeofday(&endTime, NULL);
    
    float runTime;
	unsigned long long Start, End;
	//find start time in ms
	Start = (unsigned long long)startTime.tv_sec * 1000 
		+ startTime.tv_usec/1000;
	//find end time in ms
	End = (unsigned long long)endTime.tv_sec * 1000 
		+ endTime.tv_usec/1000;
	//Time in ms
	runTime = (float)(End - Start) ;
	FILE *f = fopen("cpuAvx.txt", "a+");
	
	//runTime = runTime/1e6;
       
	if (operation == 2) {
        double flops = (((double)maxOperation * 8 / (runTime/1000)))/ 1e9   ; //GFLOPS
		fprintf(f, "%s,%d,%10f,%10f\n",argv[1],num_thread,runTime,flops);
    printf("No of threads: %d\n", num_thread);
    printf("Execution time: %10f ms\n", runTime);
    printf("GFlOPS: %10f\n",flops);
    }
    else if (operation == 1){
        double iops = (((double)maxOperation * 8/ (runTime/1000))) / 1e9 ; //IOPS
		fprintf(f, "%s,%d,%10f,%10f\n",argv[1],num_thread,runTime,iops);
    printf("No of threads: %d\n", num_thread);
    printf("Execution time: %10f ms\n", runTime);
    printf("GIOPS: %10f\n",iops);
    }
     fclose(f);
		 
    return 0;
}



void * flops(void * arg)
{
   __m256 evens = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
  __m256 odds = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);

   int i;
int j;
int k;
   for (i = 0; i < maxOperation/num_thread; i++) {

  /* Compute the difference between the two vectors */
  __m256 result = _mm256_add_ps(evens, odds);

   
  
}  
    return NULL;
}

void * iops(void * arg)
{
    
__m256i a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
__m256i b = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
   
   int i;
int j;
int k;
   for (i = 0; i < maxOperation/num_thread; i++) {

   __m256i result = _mm256_add_epi32(a,b);


    

}    
    return NULL;
}
