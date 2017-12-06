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

//define max operation 
unsigned long maxOperation = 1e9;
int num_thread;
//Function declaration
void * iops(void *);
void * flops(void *);

int main(int argc, char * argv[])
{
    /*
	operation 1: iops
	operation 2: flops
    */
    int operation = atoi(argv[1]);
    
    //Number of thread	
    num_thread = atoi(argv[2]);
    
    //Initialize start and end time variable	
    struct timeval startTime;
    struct timeval endTime;

    //Initialize no. of threads
    pthread_t thread[num_thread];
     
    //Get starting time
    gettimeofday(&startTime, NULL);

    int i;
	for(i = 0; i < num_thread; i++)
	{
		switch(operation)
		{
		//Thread for performing integer operations
		case 1:pthread_create(&thread[i], NULL, iops, NULL); 
			break;
		//Thread for performing floating operations
		case 2:pthread_create(&thread[i], NULL, flops, NULL); 
			break;
		//Default for any error
		default: printf("Error\n");
		}
		 
	}
    for (i = 0; i < num_thread; i++) 
{
        pthread_join(thread[i], NULL);
    }

    //Get end time of completion of operations
    gettimeofday(&endTime, NULL);
    
    float runTime;
	unsigned long long Start, End;
	//Calculating start time in ms
	Start = (unsigned long long)startTime.tv_sec * 1000 
		+ startTime.tv_usec/1000;
	//Calculating end time in ms
	End = (unsigned long long)endTime.tv_sec * 1000 + 
		endTime.tv_usec/1000;
	//Time in ms
	runTime = (float)(End - Start) ;
	FILE *f = fopen("cpu.txt", "a+");
	
	//runTime = runTime/1e6;
       
	if (operation == 2) {
        double flops = ((double)maxOperation * 3 / (runTime/1000))/ 1e9   ; //GFLOPS
		fprintf(f, "%s,%d,%10f,%10f\n",argv[1],num_thread,runTime,flops);
    printf("No of threads: %d\n", num_thread);
    printf("Execution time: %10f ms\n", runTime);
    printf("GFlOPS: %10f\n",flops);

    }
    else if (operation == 1){
        double iops = (((double)maxOperation * 3/ (runTime/1000))) / 1e9 ; //IOPS
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
    
    int j;
int k;
   for (i = 0; i < maxOperation/num_thread; i++) {
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
    
   int j;
int k;
   for (i = 0; i < maxOperation/num_thread; i++) {
       
          e = a + b;
      f = c + d;
      g = e * f;

}
    return NULL;
}
