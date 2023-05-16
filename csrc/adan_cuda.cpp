#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Define constants
const int MAX_THREADS = 1024;

// Function declarations
void has_nan_inf_launcher(const torch::Tensor &g_fp16, torch::Tensor mid, torch::Tensor out);
void adan_launcher(
    const torch::Tensor &param_fp32,
    const torch::Tensor &param_fp16,
    const torch::Tensor &g_fp16,
    const torch::Tensor &neg_pre_g_fp16,
    const torch::Tensor &exp_avg_fp32,
    const torch::Tensor &exp_avg_diff_fp32,
    const torch::Tensor &exp_avg_sq_fp32,
    float beta1, float beta2, float beta3, 
    float eps, float lr, 
    float scale, 
    float weight_decay, 
    float bias_correction1, 
    float bias_correction2, 
    float bias_correction3_sqrt
);

// Macro definitions for checking tensor properties
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Main functions
void F_adan(
    const torch::Tensor &param_fp32,
    const torch::Tensor &param_fp16,
    const torch::Tensor &g_fp16,
    const torch::Tensor &neg_pre_g_fp16,
    const torch::Tensor &exp_avg_fp32,
    const torch::Tensor &exp_avg_diff_fp32,
    const torch::Tensor &exp_avg_sq_fp32,
    float beta1, float beta2, float beta3,
    float bias_correction1,
    float bias_correction2,
    float bias_correction3_sqrt,
    float lr, float decay, float eps, float grad_scale)
{
    // Check input tensors
    CHECK_INPUT(param_fp32);
    CHECK_INPUT(param_fp16);
    CHECK_INPUT(g_fp16);
    CHECK_INPUT(neg_pre_g_fp16);
    CHECK_INPUT(exp_avg_fp32);
    CHECK_INPUT(exp_avg_diff_fp32);
    CHECK_INPUT(exp_avg_sq_fp32);

    // Check tensor sizes
    int64_t num_elem = param_fp32.numel();
    AT_ASSERTM(param_fp16.numel() == num_elem,
                "number of elements in param_fp16 and p tensors should be equal");
    AT_ASSERTM(exp_avg_fp32.numel() == num_elem,
                "number of elements in exp_avg and p tensors should be equal");
    AT_ASSERTM(exp_avg_sq_fp32.numel() == num_elem,
                "number of elements in exp_avg_sq and p tensors should be equal");
    AT_ASSERTM(exp_avg_diff_fp32.numel() == num_elem,
                "number of elements in exp_avg_diff and p tensors should be equal");
    AT_ASSERTM(neg_pre_g_fp16.numel() == num_elem,
                "number of elements in pre_g and p tensors should be equal");
    AT_ASSERTM(g_fp16.numel() == num_elem,
                "number of elements in g and p tensors should be equal");
    AT_ASSERTM(param_fp32.dtype() == torch::kFloat, "param_fp32 must be a float tensor");
    AT_ASSERTM(param_fp16.dtype() == torch::kHalf, "param_fp16 must be a half tensor");
    AT_ASSERTM(g_fp16.dtype() == torch::kHalf, "g_fp16 must be a half tensor");
    AT_ASSERTM(neg_pre_g_fp16.dtype() == torch::kHalf, "pre_g_fp16 must be a half tensor");
    AT_ASSERTM(exp_avg_fp32.dtype() == torch::kFloat, "exp_avg_fp32 must be a float tensor");
    AT_ASSERTM(exp_avg_diff_fp32.dtype() == torch::kFloat, "exp_avg_diff_fp32 must be a float tensor");
    AT_ASSERTM(exp_avg_sq_fp32.dtype() == torch::kFloat, "exp_avg_sq_fp32 must be a float tensor");
    
    // Call the launcher function
    adan_launcher(
        param_fp32,
        param_fp16,
        g_fp16,
        neg_pre_g_fp16,
        exp_avg_fp32,
        exp_avg_diff_fp32,
        exp_avg_sq_fp32,
        beta1, beta2, beta3, 
        eps, lr, 
        grad_scale, 
        decay, 
        bias_correction1, 
        bias_correction2, 
        bias_correction3_sqrt
    );
}

void F_has_inf_nan(const torch::Tensor &g_fp16, torch::Tensor &out) {
    // Check input tensors
    CHECK_INPUT(g_fp16);
    CHECK_INPUT(out);
    AT_ASSERTM(g_fp16.dtype() == torch::kHalf, "g_fp16 must be a half tensor");
    AT_ASSERTM(out.dtype() == torch::kUInt8, "out must be a uint8 tensor");

    // Prepare temporary tensor
    torch::Tensor mid = out.new_zeros({MAX_THREADS});

    // Call the launcher function
    has_nan_inf_launcher(g_fp16, mid, out);
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_adan", &F_adan, "adan function");
    m.def("f_has_inf_nan", &F_has_inf_nan, "has inf or nan");
}