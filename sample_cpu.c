/*
	CPU BenchMarking
	Author: Shruti Gupta and Sagar Mane
*/

//Importing require packages
#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>

//define max operation 10^8
unsigned long maxOperation = 109999999;

//Function declaration
void * iops(void *);
void * flops(void *);

int main(int argc, char * argv[])
{
    /*
	Operation 1: iops
	operation 2: flops
    */
    int operation = atoi(argv[1]);
    int num_thread = 8;
    
    //Initialize start and end time variable
    struct timeval startTime;
    struct timeval endTime;
    
    //Initialize no. of threads array
	pthread_t thread[num_thread];
    
    //get starting time of operation
    gettimeofday(&startTime, NULL);

    int i;
	for(i = 0; i < num_thread; i++)
	{
		switch(operation)
		{
                // THread for integer operations
		case 1:pthread_create(&thread[i], NULL, iops, NULL); 
		break;
		// Thread for floating operations
		case 2:pthread_create(&thread[i], NULL, flops, NULL); 
		break;
		//Default option for error
		default: printf("Error\n");
		}
		 
	}
    for (i = 0; i < num_thread; i++) 
{
        pthread_join(thread[i], NULL);
    }

    // get end time of operation execution
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
	FILE *f = fopen("cpuSample.txt", "a+");
	
	//runTime = runTime/1e6;
       
	if (operation == 2) {
        double flops = ((double)maxOperation * 3 * 8 / (runTime/1000))/ 1e9  ; //GFLOPS
		fprintf(f, "%s,%d,%10f,%10f\n",argv[1],num_thread,runTime,flops);
    printf("No of threads: %d\n", num_thread);
    printf("Execution time: %10f ms\n", runTime);
    printf("GFlOPS: %10f\n",flops);
    }
    else if (operation == 1){
        double iops = (((double)maxOperation * 3 * 8/ (runTime/1000))) / 1e9 ; //IOPS
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
    int i ;
    float a = 1.0;
	float b = 2.0;
	float c = 3.0;
	float d = 4.0;
	float e ;
	float f ;
	float g ;
    
    for (i = 0; i < maxOperation; i++) {
      e = a + b;
      f = c + d;
      g = e * f;


    }
    
    return NULL;
}

void * iops(void * arg)
{
    int i;
    int a = 1;
	int b = 2;
	int c = 3;
	int d = 4;
	int e;
	int f;
	int g;
    
    for (i = 0; i < maxOperation; i++) {
       
          e = a + b;
      f = c + d;
      g = e * f;

    }
    
    return NULL;
}
