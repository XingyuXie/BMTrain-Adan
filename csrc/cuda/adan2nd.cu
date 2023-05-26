#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
// Blocks: <n // 1024>, Threads: <min(n, 1024)>
__global__ void adan2nd_fp32_accum(
    int32_t n,
    const half *g,            // (n)
    const half *neg_pre_g,    // (n)
    // const half *hessian_est,  // (n), could be nullptr
    float *exp_avg,           // (n)
    float *exp_avg_diff,      // (n)
    float *exp_avg_sq,        // (n)
    float *param,             // (n)
    half *param_h,            // (n)
    float beta1,
    float beta2,
    // float beta3,
    float eps,
    float lr,
    float rho,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float bias_correction3
) {
    int32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= n) return;

    float local_g = __half2float(g[global_id]);  // real_g * scale
    float update = local_g + __half2float(neg_pre_g[global_id]);  // real_g * scale

    // Update exp_avg and exp_avg_diff
    exp_avg[global_id] = beta1 * exp_avg[global_id] + (1 - beta1) * local_g;
    exp_avg_diff[global_id] = beta2 * exp_avg_diff[global_id] + (1 - beta2) * update;

    // // Update exp_avg_sq
    // if (hessian_est != nullptr) {
    //     // real_est * scale
    //     exp_avg_sq[global_id] = beta3 * exp_avg_sq[global_id] + (1 - beta3) * __half2float(hessian_est[global_id]);
    // }

    // Update parameters
    float denom = exp_avg_sq[global_id] / bias_correction3 * rho + eps * scale;
    float step_size_diff = lr * beta2 / bias_correction2;
    float step_size = lr / bias_correction1;

    
    param[global_id] -= max(min(step_size * exp_avg[global_id] / denom
                        + step_size_diff * exp_avg_diff[global_id] / denom, lr), -lr);
    param[global_id] /= (1 + lr * weight_decay);

    param_h[global_id] = __float2half(param[global_id]);
}

}

void adan2nd_launcher(
    const torch::Tensor &param_fp32,
    const torch::Tensor &param_fp16,
    const torch::Tensor &g_fp16,
    const torch::Tensor &neg_pre_g_fp16,
    // const std::optional<torch::Tensor> &hessian_est_fp16,
    const torch::Tensor &exp_avg_fp32,
    const torch::Tensor &exp_avg_diff_fp32,
    const torch::Tensor &exp_avg_sq_fp32,
    float beta1, float beta2, //float beta3, 
    float eps, float lr, float rho,
    float scale, 
    float weight_decay, 
    float bias_correction1, 
    float bias_correction2, 
    float bias_correction3
) {
    int32_t n = param_fp32.numel();
    if (n <= 0) return;

    auto g_ptr = reinterpret_cast<half*>(g_fp16.data_ptr<at::Half>());
    auto neg_pre_g_ptr = reinterpret_cast<half*>(neg_pre_g_fp16.data_ptr<at::Half>());
    auto exp_avg_ptr = exp_avg_fp32.data_ptr<float>();
    auto exp_avg_diff_ptr = exp_avg_diff_fp32.data_ptr<float>();
    auto exp_avg_sq_ptr = exp_avg_sq_fp32.data_ptr<float>();
    auto param_ptr = param_fp32.data_ptr<float>();
    auto param_h_ptr = reinterpret_cast<half*>(param_fp16.data_ptr<at::Half>());

    // half* hessian_est_ptr = nullptr;
    // if (hessian_est_fp16.has_value()) {
    //     hessian_est_ptr = reinterpret_cast<half*>(hessian_est_fp16.value().data_ptr<at::Half>());
    // }

    int32_t threads = 1024;
    dim3 block_size = dim3(threads, 1, 1);
    dim3 grid_size = dim3((n + threads - 1) / threads, 1, 1);

    auto stream = at::cuda::getCurrentCUDAStream();

    adan2nd_fp32_accum<<<grid_size, block_size, 0, stream.stream()>>>(
        n, g_ptr, neg_pre_g_ptr, exp_avg_ptr, exp_avg_diff_ptr, 
        exp_avg_sq_ptr, param_ptr, param_h_ptr,
        beta1, beta2, eps, lr, rho,
        scale, weight_decay,
        bias_correction1, bias_correction2, bias_correction3
    );
}
