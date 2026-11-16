#include "../kernel.h"
#include "../kernel.cuh"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{.FWD_MODE = SparseAttnFwdMode::Decode, .MODEL_TYPE = ModelType::V41, .EXTRA_MODEL_TYPE = ModelType::V41_FP4, .H_Q = 64, .ENABLE_Q_NORM = false, .STAGE_Q = false, .QKV_FP8 = true}>(const ParamT<SparseAttnFwdMode::Decode>& params);

}
