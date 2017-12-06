/*

Memory Bench Marking

Author: Shruti Gupta and Sagar Mane

*/

//Importing require packages
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>

//Declaring file length
long memLength;

//Function Declaration before used
void * accessSeq(void *);
void * accessRand(void *);
void * writeSeq(void *);
void * writeRand(void *);
void * ReadWrite(void *);
int BlockSize(char *);
long fileSize(char *);

int main(int argc, char *argv[])
{
	// Initialization
	int operation = atoi(argv[1]);
	char *block = argv[2];
	int num_Thread = atoi(argv[3]);
	
	//Set Block size 1B = 1  1KB = 1024  1MB == 1024*1024  10MB==10*1024*1024
	int blockSize;
	int BlockSize(char *block);
	blockSize = BlockSize(block);
	
	//Set File Size
	long fileSize(char *block);
	memLength = fileSize(block);
	memLength = memLength/num_Thread;
	//Initialize no. of threads
	pthread_t thread[num_Thread];
	
	//Initialization variable for calculating time.
	struct timeval startTime ;
	struct timeval endTime;
	unsigned long long Start;
	unsigned long long End;
	float totalTime;
	
	//Set Start Time
	gettimeofday(&startTime, NULL);
	
	/*
	Creating thread operation based on number of operation
	1 = Sequentially Write
	2 = Randomly Write
	3 = Sequentially Access
	4 = Randomly Access
	*/
	int i;
	for(i = 0; i < num_Thread; i++)
	{
		switch(operation)
		{
		//Thread write sequential
		case 1:pthread_create(&thread[i], NULL, writeSeq, (void*)(long)blockSize); 
		break;
		//Thread write random
		case 2:pthread_create(&thread[i], NULL, writeRand, (void*)(long)blockSize); 
		break;
		//thread read sequential
		case 3:pthread_create(&thread[i], NULL, accessSeq, (void*)(long)blockSize); 
		break;
		//Thread Read random
		case 4:pthread_create(&thread[i], NULL, accessRand, (void*)(long)blockSize); 
		break;
		//THread rean and write
		case 5:pthread_create(&thread[i], NULL, ReadWrite, (void*)(long)blockSize); 
		break;
		//Default for error
		default: printf("Error\n");
		}
		 
	}
	/*
	Joint All thread for parallel running thread
	*/
	for(i = 0; i < num_Thread; i++)
	{
		pthread_join(thread[i], NULL);
	}
	
	//Set End time of Operations
	gettimeofday(&endTime, NULL);
	//Calculating running time by difference between EndTime and StartTime  
	float runTime;
	//start time for us
	Start = (unsigned long long)startTime.tv_sec * 1000000 
		+ startTime.tv_usec;
	//End time for us
	End = (unsigned long long)endTime.tv_sec * 1000000 
		+ endTime.tv_usec;
	runTime = (float)(End - Start) / 1000;
	
	//Calculating ThroughPut using 1MB
	float throughPut = (float)num_Thread * memLength / runTime * 1000 / 1048576;
	
	//Calculating Latency using 1 bit
	float latency = (float)(runTime*1000)/(memLength*num_Thread*8);

	FILE *f = fopen("memory.txt", "a+");
	fprintf(f, "%d,%s,%d,%10f,%10f,%10f\n",operation,block,num_Thread,runTime,throughPut,latency);
        fclose(f);
	printf("Run time: %10f ms\n", runTime);
	printf("ThroughPut: %10f MB/sec\n",throughPut);
	printf("Latency: %10f us/bit\n",latency);
	return 0;
		
}
/*
	Function to return block size
	//Set Block size 1B = 1  1KB = 1024  1MB == 1024*1024  10MB==10*1024*1024
*/
int BlockSize(char *block)
{
	int blockSize;
	if(strcmp(block, "8B") == 0)
	{
		blockSize = 8;
	}
	else if(strcmp(block, "8KB") == 0)
	{
		blockSize = 8*1024;
	}
	else if(strcmp(block, "8MB") == 0)
	{
		blockSize = 1024*1024*8;
	}
	else if(strcmp(block, "80MB") == 0)
	{
		blockSize = 1024*1024*80;
	}
	return blockSize;
}
long fileSize(char *block)
{
	long fileSize;
	if(strcmp(block, "8B") == 0)
	{
		fileSize = 1024*1024*1000;
	}
	else 
	{
		fileSize = 1024*1024*1000;
	}
	
	return fileSize;
}

void *writeSeq(void *blockSize)
{
	//define block size
	int size = (int)(long)blockSize;
	int i;
	//Allocate memory for buffer
	char *buffer = (char*)malloc(sizeof(char) * memLength);
	for(i = 0; i < memLength/size; i++)
	{	
	
	
	//memset used for fill the block of memory with particular value
	memset(&buffer[i*size], 's', size);
	}
	free(buffer);
	
}

void *writeRand(void *blockSize)
{
	//define block size
	int size = (int)(long)blockSize;
	int i;
	int j;
	//Allocate memory
	char *buffer = (char*)malloc(sizeof(char) * memLength);
	for(i = 0; i < memLength/size; i++)
	{	
		//Allocate Memory
	j = rand() % (memLength/size);
	
	//memset used for fill the block of memory with particular value
	memset(&buffer[j*size], 's', size);
	}
   free(buffer);
}
/*

	Function to Access Sequentially (Operation No. 3)
	
*/
void *accessSeq(void *blockSize)
{
	int i;
	//define block size
	int size = (int)(long)blockSize;
	//Memory Allocation
	char *temp1 = (char*)malloc(sizeof(char) * memLength);
	char *temp = (char*)malloc(sizeof(char) * size);
	//copy memory block by block to new allocated memories
	for(i = 0; i < memLength/size; i++)
	{	
		// Cop Memory to temp
		memcpy(temp, temp1 + i * size, size);
	}
	//Free Memory
	free(temp1);
	free(temp);	
}
/*

	Function to Accesss randomly (Operation No. 4) 
	
*/
void *accessRand(void *blockSize)
{
	int i;
	int ran;
	//define block size
	int size = (int)(long)blockSize;
	//Memory Allocation
	char *temp1 = (char*)malloc(sizeof(char) * memLength);
	char *temp = (char*)malloc(sizeof(char) * size);
	srand((unsigned)time(NULL));
	for(i = 0; i < memLength/size; i++)
	{
		ran = rand()%(memLength/size);
		//Memory copy to temp
		memcpy(temp, temp1 + ran * size, size);
	}
	//Free Memory
	free(temp1);
	free(temp);
}

/*

	Function to Access Sequentially (Operation No. 3)
	
*/
void *ReadWrite(void *blockSize)
{
	int i;
	//define block size
	int size = (int)(long)blockSize;
	//Memory Allocation
	char *temp1 = (char*)malloc(sizeof(char) * memLength);
	char *temp = (char*)malloc(sizeof(char) * size);
	char *NewMem = (char*)malloc(sizeof(char) * memLength);
	//copy memory block by block to new allocated memories
	for(i = 0; i < memLength/size; i++)
	{	
		// Copy Memory to temp
		memcpy(temp, temp1 + i * size, size);
		memset(NewMem+i * size, *temp, size);
	}
	//Free Memory
	free(NewMem);
	free(temp1);
	free(temp);	
}

