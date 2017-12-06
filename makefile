gcc -O -o cpu_bench cpu_bench.c -lpthread

gcc -O -mavx2 -o cpu_avx cpu_avx.c -lpthread

gcc -O -o sample_cpu sample_cpu.c -lpthread

gcc -O -o disk_bench disk_bench.c -lpthread

gcc -O -o memory_bench memory_bench.c -lpthread

gcc -o network_server_bench network_server_bench.c -lpthread

gcc -o network_client_bench network_client_bench.c -lpthread
