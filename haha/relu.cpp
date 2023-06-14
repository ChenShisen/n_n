#include <cstdio>
#include <xmmintrin.h>
#include <chrono>

void relu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}

void relu_sse(float* input, float* output, int size) {
    __m128 zero = _mm_setzero_ps();
    #pragma omp parallel for
    for (int i = 0; i < size; i += 4) {
        __m128 in_value = _mm_loadu_ps(input + i);
        __m128 mask = _mm_cmpgt_ps(in_value, zero);
        __m128 out_value = _mm_and_ps(in_value, mask);
        _mm_storeu_ps(output + i, out_value);
    }
}

int main() {
    int size = 10000000;
    float* input = new float[size];
    float* output = new float[size];
    for (int i = 0; i < size; i++) {
        input[i] = i - (size / 2);
    }

    auto start_time = std::chrono::steady_clock::now();
    relu(input, output, size);
    auto end_time = std::chrono::steady_clock::now();
    auto time_diff = end_time - start_time;
    std::printf("�~N~_�~K CPU 计�~W�~@~W�~W��~Z%.3lf �~R\n", std::chrono::duration<double>(time_diff).count());

    start_time = std::chrono::steady_clock::now();
    relu_sse(input, output, size);
    end_time = std::chrono::steady_clock::now();
    time_diff = end_time - start_time;
    std::printf("SSE �~L~G令�~X�~L~V计�~W�~@~W�~W��~Z%.3lf �~R\n", std::chrono::duration<double>(time_diff).count());

    delete[] input;
    delete[] output;
    return 0;
}
