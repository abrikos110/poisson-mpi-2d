g++ -std=c++11 hw.cpp -o hw_seq -O3 -Wall -Wextra -Wpedantic
g++ -std=c++11 hw.cpp -o hw_par -O3 -Wall -Wextra -Wpedantic -fopenmp

mpic++ -std=c++11 hw_single.cpp -o hw_single -O3 -Wall -Wextra -Wpedantic -fopenmp -DOMPI_SKIP_MPICXX
