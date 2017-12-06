#include <pthread.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

//buffer size to be received (64KB)
#define SIZE_TO_RECEIVE 65536

//port number
#define PORT 1245

//Initializing the functions
void * TCP(void *);
void * UDP(void *);


int main(int argc, char *argv[])
{

	int id;
	/*
	TYPE of connection
	1: TCP
	2: UDP
	*/
	int conection = atoi(argv[1]);
	//number of threads
	int num_Thread = atoi(argv[2]);
	pthread_t threads[num_Thread];

	for (id = 0; id < num_Thread; id++)
	{
		switch(conection)
		{
		//thread for tcp
		case 1:pthread_create(&threads[id], NULL, TCP, (void*)(long)id); 
		break;
		//thread for udp
		case 2:pthread_create(&threads[id], NULL, UDP, (void*)(long)id); 
		break;
		//default for error
		default: printf("Error...\n");
		}
	}

	int i;
	for (i = 0; i < num_Thread; i++)
	{
		pthread_join(threads[i], NULL);
	}

	return 0;
}

/*
 TCP Server
*/
void *TCP(void *threadID)
{
	//Thread id
	int threadid = (int)(long)threadID;
	//Socket descriptor for server and client
	int Server_sockfd, Client_sockfd;
	int size;
	struct sockaddr_in address;
	char *buffer;
	socklen_t addrlen;
	int opt = 1;

	/*
		Socket Creation
		AF_INET (IPv4 protocol) : communication domain
		type: communication type : SOCK_STREAM: TCP(reliable, connection oriented)

	*/
	Server_sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (Server_sockfd == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }


	 if (setsockopt(Server_sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                                                  &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons(PORT + threadid);

	//binds the socket to the address and port number specified in addr
    if (bind(Server_sockfd, (struct sockaddr *)&address, sizeof(address))<0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

	/*
	Puts the server socket in a passive mode, where it waits for the client to approach the server to make a connection
	8 is maximum length to which the queue of pending connections for sockfd may grow
	*/
	 if (listen(Server_sockfd, 5) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }
	addrlen = sizeof(struct sockaddr);
	//when a client require to connect, server accept it
	Client_sockfd = accept(Server_sockfd, (struct sockaddr *)&Client_sockfd, &addrlen);

	 if (Client_sockfd < 0)
    {
        perror("accept");
        exit(EXIT_FAILURE);
    }
	buffer = (char*)malloc(SIZE_TO_RECEIVE);
	size = SIZE_TO_RECEIVE;

	//if size is 0, all messages have been sent
	while(size > 0)
	{
		//receive a message from a connected socket
		size = recv(Client_sockfd, buffer, SIZE_TO_RECEIVE, 0);

	}
	//Close socket connection.
	free(buffer);
	close(Client_sockfd);
	close(Server_sockfd);
}

/*
 UDP Server
*/
void *UDP(void *threadID)
{
	//Socket descriptor
	int sockfd;
	int size;
	struct sockaddr_in serverAddress, clientAddress;
	char *buffer;
	socklen_t addrlen;
int opt = 1;

	int threadid = (int)(long)threadID;
	/*
		Socket Creation
		AF_INET (IPv4 protocol) : communication domain
		type: communication type : SOCK_DGRAM: UDP(unreliable, connectionless)

	*/
	sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	if (sockfd == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

 if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                                                  &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

	memset(&serverAddress, 0 ,sizeof(serverAddress));
	serverAddress.sin_family = AF_INET;
	serverAddress.sin_addr.s_addr = INADDR_ANY;
	serverAddress.sin_port = htons(PORT + threadid);

	addrlen = sizeof(struct sockaddr);

	//binds the socket to the address and port number specified in addr
    if (bind(sockfd, (struct sockaddr *)&serverAddress, sizeof(serverAddress))<0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

	//Set Buffer size for receiving messages from client
	buffer = (char*)malloc(65636);
	memset(buffer, 0, 65636);

	//Messages receive from client
	size = recvfrom(sockfd, buffer, 65636, 0, (struct sockaddr *)&clientAddress, &addrlen);

	//close socket
	free(buffer);
	close(sockfd);
}

