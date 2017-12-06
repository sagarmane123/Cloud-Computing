/*

Disk Bench Marking

Author: Shruti Gupta and Sagar Mane


*/

//Importing require packages

#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>

//Declaring file length
long fileLength;

//Function Declaration before used
void * WriteSeq(void *);
void * WriteRand(void *);
void * ReadSeq(void *);
void * ReadRand(void *);
void * ReadWrite(void *);
int block_Size(char *);
long file_Size(char *);
void startThread(int, int);


int main(int argc, char *argv[])
{
	// Initialization
	int operation = atoi(argv[1]);
	char *block = argv[2];
	int num_Thread = atoi(argv[3]);
	
	//Set Block size 1B = 1  1KB = 1024  1MB == 1024*1024  10MB==10*1024*1024
	int blockSize;
	int block_Size(char *block);
	blockSize = block_Size(block);
	//printf("%d %d\n", operation, blockSize);
	//Set File Size
	long file_Size(char *block);
	fileLength = file_Size(block);
	fileLength = fileLength/num_Thread;
	//printf("%d %d\n", fileLength, blockSize);
	
	//Initialization variable for calculating time.
	struct timeval startTime;
        struct timeval endTime;
	
	float totalTime;
	
	//Set Start Time
	gettimeofday(&startTime, NULL);
	clock_t t;
        t = clock();
	
	//Initialize no. of threads
	pthread_t thread[num_Thread];
	/*
	Creating thread operation based on number of operation
	1 = Sequentially Write
	2 = Randomly Write
	3 = Sequentially Read
	4 = Randomly Read
	*/
	int i;
	for(i = 0; i < num_Thread; i++)
	{
		switch(operation)
		{
		// Write Sequentially
		case 1:pthread_create(&thread[i], NULL, WriteSeq, (void*)(long)blockSize); 
			break;
		//Write Randomly
		case 2:pthread_create(&thread[i], NULL, WriteRand, (void*)(long)blockSize); 
			break;
		//Read Sequintially
		case 3:pthread_create(&thread[i], NULL, ReadSeq, (void*)(long)blockSize); 
			break;
		//Read Randomly
		case 4:pthread_create(&thread[i], NULL, ReadRand, (void*)(long)blockSize); 
			break;
		//Reand and Write file
		case 5:pthread_create(&thread[i], NULL, ReadWrite, (void*)(long)blockSize); 
			break;
		// Default for error
		default: printf("Error in thread\n");
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
	t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
	//Calculating running time by difference between EndTime and StartTime  
	float runTime;
	unsigned long long Start, End;
	//find start time in sec
	Start = (unsigned long long)startTime.tv_sec  +
	 startTime.tv_usec/1000000;
	//find end time in sec
	End = (unsigned long long)endTime.tv_sec  +
	 endTime.tv_usec/1000000;
	runTime = (float)(End - Start);
	//printf("start\n %f",Start);
	//printf("End\n %f",End);
	//printf("runTime\n %f",runTime);
	//Calculating ThroughPut using 1MB
	
	float throughPut = (float)num_Thread * fileLength / (time_taken) / 1048576;;

	//Calculating Latency using 1 bit
	float latency = (float)(time_taken*1000000)/(fileLength*num_Thread);
	FILE *f = fopen("disk.txt", "a+");
	fprintf(f, "%d,%s,%d,%10f,%10f,%10f\n",operation,block,num_Thread,time_taken*1000,throughPut,latency);
        fclose(f);
	printf("Run time: %10f ms\n",time_taken*1000);
	printf("Throughput: %10f Mb/Sec\n",throughPut);
	printf("Latency: %10f ms/bit\n",latency);
	return 0;
		
}
/*
	Function to return block size
	//Set Block size 1B = 1  1KB = 1024  1MB == 1024*1024  10MB==10*1024*1024
*/
int block_Size(char *block)
{
	int blockSize;
	if(strcmp(block, "8B") == 0)
	{
		blockSize = 8;
	}
	else if(strcmp(block, "8KB") == 0)
	{
		blockSize = 1024*8;
	}
	else if(strcmp(block, "8MB") == 0)
	{
		blockSize = 1048576*8;
	}
	else if(strcmp(block, "80MB") == 0)
	{
		blockSize = 1048576*80;
	}
	return blockSize;
}

long file_Size(char *block)
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

void startThread(int num_Thread, int Operation){
	
}
/*
	Function to write Sequentially (Operation No. 1)
*/
void *WriteSeq(void *block_Sizes)
{
	/*
	Variable declaration for Read file randomly
	@i used for storing file length
	@size store block size
	*/
	int i;
	int size = (int)(long)block_Sizes;
	int file;
	//open file for write
	file = open("test.txt", O_CREAT|O_TRUNC|O_WRONLY, 0666);
	//Check for file Exit or not
	if(file < 0)
	{
		exit(0);
	}
	//Allocate Memory
	char *buffer = (char*)malloc(sizeof(char) * size);
	//memset used for fill the block of memory with particular value
	memset(buffer, 's', size);
	i = fileLength;
	while(i>0)
	{
		//Write file
		write(file, buffer, size);
		//Reduce file length
		i = i - size;
	}
	//Free Memory
	free(buffer);
	close(file);

}

/*
	Function to write randomly (Operation No. 2) 
*/
void *WriteRand(void *block_Sizes)
{
	/*
	Variable declaration for Read file randomly
	@i used for storing file length
	@r used for storing randomly read file position
	@size store block size
	*/
	int i, r;
	int size = (int)(long)block_Sizes;
	srand((unsigned)time(NULL));
	int file;
	//open file for write
	file = open("test.txt", O_CREAT|O_TRUNC|O_WRONLY, 0666);
	//Check for file Exit or not
	if(file < 0)
	{
		exit(0);
	}
	//Allocate Memory
	char *buffer = (char*)malloc(sizeof(char) * size);
	//memset used for fill the block of memory with particular value
	memset(buffer, 's', size);
	i = fileLength;
	while(i>0)
	{
		r = rand()%(fileLength / size);
		/*
		lseek used to change location of read pointer
		@file : The file descriptor of the pointer that is going to be moved
		@r*size : The offset of the pointer
		@SEEK_SET : SEEK_CUR specifies that the offset provided is relative to the current file position
		*/
		lseek(file, r * size, SEEK_SET);
		//Write file
		write(file, buffer, size);
		//Reduce size of file
		i = i - size;
	}
	//Free Memory
	free(buffer);
	close(file);

}

/*
	Function to Read Sequentially (Operation No. 3)
*/
void *ReadSeq(void *block_Sizes)
{
	/*
	Variable declaration for Read file randomly
	@i used for storing file length
	@size store block size
	*/
	int i;
	int size = (int)(long)block_Sizes;
	int file;
	//open file for read
	file = open("test.txt", O_RDONLY, 0666);
	//Check for file Exit or not
	if(file < 0)
	{
		exit(0);
	}
	
	i = fileLength;
	while(i>0)
	{	
		// Allocate Memory
		char *buffer = (char*)malloc(sizeof(char) * size);
		//Read file data
		read(file, buffer, size);
		//Reduce file size
		i = i - size;
		//Free memory
		free(buffer);
	}
	
	close(file);

}

/*
	Function to Read Randomly
*/
void *ReadRand(void *block_Sizes)
{
	/*
	Variable declaration for Read file randomly
	@i used for storing file length
	@r used for storing randomly read file position
	@size store block size
	*/
	int i, r;
	int size = (int)(long)block_Sizes;
	srand((unsigned)time(NULL));
	int file;
	//open file for read
	file = open("test.txt", O_RDONLY, 0666);
	//Check for file Exit or not
	if(file < 0)
	{
		exit(0);
	}	
	i = fileLength;
	while(i>0)
	{
		r = rand()%(fileLength / size);
		// Allocate memory
		char *buffer = (char*)malloc(sizeof(char) * size);
		/*
		lseek used to change location of read pointer
		@file : The file descriptor of the pointer that is going to be moved
		@r*size : The offset of the pointer
		@SEEK_SET : SEEK_CUR specifies that the offset provided is relative to the current file position
		*/
		lseek(file, r * size, SEEK_SET);
		//read file data
		read(file, buffer, size);
		
		//reduce size of file
		i = i - size;
		//free allocated memory
		free(buffer);
	}
	
	close(file);

}


/*
	Function to Read+Write Sequentially (Operation No. 5)
*/
void *ReadWrite(void *block_Sizes)
{
	/*
	Variable declaration for Read file randomly
	@i used for storing file length
	@size store block size
	*/
	int i;
	int size = (int)(long)block_Sizes;
	int file;
	//open file for read
	file = open("test.txt", O_RDONLY, 0666);
	//Check for file Exit or not
	if(file < 0)
	{
		exit(0);
	}
	
	int fileNew;
	//open new file for write
	fileNew = open("testNew.txt", O_CREAT|O_TRUNC|O_WRONLY, 0666);
	//Check for file Exit or not
	if(fileNew < 0)
	{
		exit(0);
	}
	
	i = fileLength;
	while(i>0)
	{	
		// Allocate Memory
		char *buffer = (char*)malloc(sizeof(char) * size);
		//Read file data
		read(file, buffer, size);
		//Reduce file size
		//write file
		write(fileNew, buffer, size);
		i = i - size;
		//Free memory
		free(buffer);
	}
	//close file
	close(file);

}
