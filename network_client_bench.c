#include <errno.h>
#include <pthread.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>


//Defining Port Number
#define PORT 1245
int ThreadNumber;

//Funtion Initilization
void * clientTCP ( void * );
void * clientUDP ( void * );

//Struct to store parameters
typedef struct Info
{
	int packet_size;
	int thread_id;
} PackageData;


int main(int argc, char *argv[])
{

	int i;
	/*
	TYPE of connection
	1: TCP
	2: UDP
	*/
	int conection = atoi(argv[1]);
	//number of threads
	ThreadNumber = atoi(argv[2]);
	//Thread array Initialization
	pthread_t threadID[ThreadNumber];
	
	//parameter structure passed in pthread
	PackageData d[8];
	d[0].packet_size = 65536;
	d[1].packet_size = 65536;
	d[2].packet_size = 65536;
	d[3].packet_size = 65536;
	d[4].packet_size = 65536;
	d[5].packet_size = 65536;
	d[6].packet_size = 65536;
	d[7].packet_size = 65536;
	//variables to count the time
	struct timeval startTime, endTime;
	unsigned long long start, end;
	
	//Getting start time
	gettimeofday(&startTime, NULL);
	
	/*
		
	Creating thread operation based on number of operation
	1 = TCP
	2 = UDP
	
	*/
	for (i = 0; i < ThreadNumber; i++)
	{
		d[i].thread_id = i;
		switch(conection)
		{
		//thread for tcp
		case 1:pthread_create(&threadID[i], NULL, clientTCP, &d[i]);
			 break;
		//thread for udp
		case 2:pthread_create(&threadID[i], NULL, clientUDP, &d[i]); 
			break;
		//default for error
		default: printf("Error...\n");
		}
	}

	for (i = 0; i < ThreadNumber; i++)
	{
		pthread_join(threadID[i], NULL);
	}
	
	//end time
	gettimeofday(&endTime, NULL);
	//calculating start time in us
	start = (unsigned long long)startTime.tv_sec * 1000000 
		+ startTime.tv_usec;
	//calculating end time in us
	end = (unsigned long long)endTime.tv_sec * 1000000 
		+ endTime.tv_usec;
	//Calculating running time in ms
	float running_time = (float)(end - start) / 1000;
	float through_put = (((float) 65536 * 1024 * 1024 * 400 * 8) / (running_time * 1000)) / 1048576;
	float latency = (float)(running_time * 1000)/(1024 * 1024 * 400);
	//Creating tile for storing result
	FILE *f = fopen("network.txt", "a+");
	fprintf(f, "%d,%d,%0.3f,%10f,%10f\n",conection,ThreadNumber,running_time,through_put,latency);
    fclose(f);
	printf("Running time %.3f ms\n", running_time);
	printf("Throughput is %10f Mbits/sec\n", through_put);
	printf("Latency is %10f ms/bit\n", latency);
	
	return 0;
}


/*
	TCP Client
*/
void *clientTCP(void *arg)
{
	//Socket descriptor
	int sockfd;
	struct sockaddr_in serverAdress;
	char *buffer;
	int BuffSize, threadID;
	PackageData *pInfo;
	pInfo = (PackageData *)arg;
	BuffSize = pInfo->packet_size;
	threadID = pInfo->thread_id;
	
	/*
		Socket Creation
		AF_INET (IPv4 protocol) : communication domain
		type: communication type : SOCK_STREAM: TCP(reliable, connection oriented)

	*/
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd == 0)
    {
        perror("socket Creation Error...");
        exit(EXIT_FAILURE);
    }
	//printf("Client socket connected for %d\n", threadID);
	//set TCP packet size according to the packet size
	//setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, (const char*)&BuffSize, sizeof(int));
	
	memset(&serverAdress, 0, sizeof(serverAdress));
	serverAdress.sin_family = AF_INET;
	serverAdress.sin_addr.s_addr = htonl(INADDR_ANY);
	serverAdress.sin_port = htons(PORT + threadID);
	
	//Connect client to server
	if (connect(sockfd, (struct sockaddr *)&serverAdress, sizeof(serverAdress)) < 0)
    {
        printf("\nConnection Failed \n");
        exit(EXIT_FAILURE);
    }
	//printf("Client connected to server for %d\n", threadID);
	//Sending message from client to server 
	int i;
	for(i = 0; i < (1024*1024*400/ThreadNumber)/65536; i++)
	{
		buffer = (char*)malloc(65536);
		memset(buffer, 'a', 65536);
		send(sockfd, buffer, 65536, 0);
	}
	//close socket
	close(sockfd);
}

/*
	UDP Client
*/
void *clientUDP(void *arg)
{
	//Socket descriptor
	int sockfd,size;
	struct sockaddr_in serverAdress;
	char *buffer;
	socklen_t addrlen;
	int BuffSize, threadID;
	PackageData *pInfo;
	pInfo = (PackageData *)arg;
	BuffSize = pInfo->packet_size;
	threadID = pInfo->thread_id;
	
	/*
		Socket Creation
		AF_INET (IPv4 protocol) : communication domain
		type: communication type : SOCK_DGRAM: UDP(unreliable, connectionless)

	*/
	sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	if (sockfd == 0)
    {
        perror("socket connection error");
        exit(EXIT_FAILURE);
    }
	
	// Allow to used port many times
	setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, (const char*)&BuffSize, sizeof(int));

	memset(&serverAdress, 0, sizeof(serverAdress));
	serverAdress.sin_family = AF_INET;
	serverAdress.sin_addr.s_addr = htonl(INADDR_ANY);
	serverAdress.sin_port = htons(PORT + threadID);

	addrlen = sizeof(serverAdress);

	int i;
	
	//start send messages to server
	for(i = 0; i < (1024*1024*400/ThreadNumber)/65536; i++)
	{
		
		buffer = (char*)malloc(65536);
		memset(buffer, 'a', 65536);
		size = 65536;
		int sendsize;
		while(size > 0)
		{
			//Sending packet to server: packet size is 64kb
			sendsize = sendto(sockfd, buffer, BuffSize, 0, (struct sockaddr *)&serverAdress, addrlen);
			size -= BuffSize;
		}
	}	
	//Close socket
	close(sockfd);
}


