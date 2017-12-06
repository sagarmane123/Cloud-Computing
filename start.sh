#!/bin/sh


echo CPU Benchmarking Start!

gcc -O -o cpu_bench cpu_bench.c -lpthread

mkdir cpu
for opt in  1 2
	do 
		for thread in 1 2 4 8
			do
				
				echo "operation type: "$opt",thread number: "$thread
				echo $thread >> cpu"_"$opt",thread number: "$thread.txt	
				./cpu_bench $opt $thread >>	\
				cpu"_"$opt",thread number: "$thread.txt
				mv cpu"_"$opt",thread number: "$thread.txt cpu
			done
	done
mv cpu.txt cpu
chmod 777 cpu

echo CPU Benchmarking Finish!


echo CPU AVX Benchmarking Start!

gcc -O -mavx2 -o cpu_avx cpu_avx.c -lpthread

mkdir cpuAvx
for opt in  1 2
	do 
		for thread in 1 2 4 8
			do
				
				echo "operation type: "$opt",thread number: "$thread
				echo $thread >> cpu"_"$opt",thread number: "$thread.txt	
				./cpu_avx $opt $thread >>	\
				cpu"_"$opt",thread number: "$thread.txt
				mv cpu"_"$opt",thread number: "$thread.txt cpuAvx
			done
	done
mv cpuAvx.txt cpuAvx
chmod 777 cpuAvx

echo CPU AVX Benchmarking Finish!


echo 10 Min CPU Benchmarking Start!

gcc -O -o sample_cpu sample_cpu.c -lpthread

mkdir cpuSample
for opt in 1 2
do
	for (( i = 0; i < 600; i++ ))
	do
		echo "Samples are taken for "$opt $i
		./sample_cpu $opt >> samples"_"$opt.txt
		
	done
	mv samples"_"$opt.txt cpuSample
done
mv cpuSample.txt cpuSample
chmod 777 cpuSample

echo 10 Min CPU Benchmarking Finish!

echo Disk Benchmarking Start!

gcc -O -o disk_bench disk_bench.c -lpthread

mkdir disk
for opt in 1 2 3 4 5
	do
		for blocksize in 8B 8KB 8MB 80MB
		do
			for thread in 1 2 4 8
			do
				
				echo "operation type: "$opt", block size:"$blocksize", thread number: "$thread
				echo $thread >> disk"_"$opt", block size:"$blocksize", thread number: "$thread.txt	
				./disk_bench $opt $blocksize $thread >>	\
				disk"_"$opt", block size:"$blocksize", thread number: "$thread.txt
				mv disk"_"$opt", block size:"$blocksize", thread number: "$thread.txt disk
			done 
			
		done
	done
mv disk.txt disk
chmod 777 disk

echo Disk Benchmarking Finish!

echo Memory Benchmarking start!

gcc -O -o memory_bench memory_bench.c -lpthread

mkdir memory
for opt in 1 2 3 4 5
	do
	for blocksize in 8B 8KB 8MB 80MB
		do
			for thread in 1 2 4 8
			do
				
				echo "operation type: "$opt", block size:"$blocksize", thread number: "$thread
				echo $thread >> memory"_"$opt", block size:"$blocksize", thread number: "$thread.txt	
				./memory_bench $opt $blocksize $thread >>	\
				memory"_"$opt", block size:"$blocksize", thread number: "$thread.txt
				mv memory"_"$opt", block size:"$blocksize", thread number: "$thread.txt memory
			done 
			
		done
	done
mv memory.txt memory
chmod 777 memory
echo Memory Benchmarking finish!

echo Network Benchmarking start!

gcc -o network_server_bench network_server_bench.c -lpthread

gcc -o network_client_bench network_client_bench.c -lpthread

mkdir network
for connect in 1 2
do
	for thread in 1 2 4 8
		do	
			
	
			echo "connect type: "$connect", thread number: "$thread
			echo $thread thread >> network"_ConnectionType: "$connect", THread: "$thread.txt 		
			./network_server_bench $connect $thread &	
			sleep 2
			./network_client_bench $connect $thread >> network"_ConnectionType: "$connect", THread: "$thread.txt 
                        mv network"_ConnectionType: "$connect", THread: "$thread.txt  network
		done
		
done
mv network.txt network
echo Network testing finish!
