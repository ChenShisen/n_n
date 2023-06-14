#include <cstdio>
#include <immintrin.h>
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

void relu_sse_bf16(unsigned short* input, unsigned short* output, int size) {
    __m128i zero = _mm_setzero_si128();
    __m128i mask_16 = _mm_set1_epi16(0x7fff);
    __m128i mask_32 = _mm_set1_epi32(0x7fff7fff);
    __m128i input_32, in_value;
    __m128 out_value;
    #pragma omp parallel for
    for (int i = 0; i < size; i += 8) {
        input_32 = _mm_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(input + i)));
        in_value = _mm_castps_si128(_mm_cvtph_ps(input_32));
        __m128 mask = _mm_cmpgt_ps(_mm_castsi128_ps(in_value), _mm_castsi128_ps(zero));
        __mmask8 mask8 = _mm_movemask_ps(mask);
        out_value = _mm_mask_mov_ps(_mm_castsi128_ps(in_value), mask8, _mm_castsi128_ps(zero));
        _mm_storeu_si128((__m128i*)(output + i), _mm_packus_epi32(_mm_castps_si128(out_value), zero));
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
    std::printf("CPU 计算耗时：%.3lf 秒\n", std::chrono::duration<double>(time_diff).count());
    
    start_time = std::chrono::steady_clock::now();
    relu_sse(input, output, size);
    end_time = std::chrono::steady_clock::now();
    time_diff = end_time - start_time;
    std::printf("SSE 指令优化单精度浮点数计算耗时：%.3lf 秒\n", std::chrono::duration<double>(time_diff).count());

    unsigned short* input_bf16 = new unsigned short[size];
    unsigned short* output_bf16 = new unsigned short[size];
    for (int i = 0; i < size; i++) {
        input_bf16[i] = i - (size / 2);
    }

    start_time = std::chrono::steady_clock::now();
    relu_sse_bf16(input_bf16, output_bf16, size);
    end_time = std::chrono::steady_clock::now();
    time_diff = end_time - start_time;
    std::printf("SSE 指令优化 BF16 计算耗时：%.3lf 秒\n", std::chrono::duration<double>(time_diff).count());

    delete[] input;
    delete[] output;
    delete[] input_bf16;
    delete[] output_bf16;
    return 0;
}
