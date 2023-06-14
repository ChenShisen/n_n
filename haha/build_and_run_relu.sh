source /opt/rh/devtoolset-7/enable
g++ -msse -msse2 -O3 -std=c++11 -march=native relu.cpp -o relu
./relu
