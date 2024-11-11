#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef SD_BUILD_SHARED_LIB
#define SD_API
#else
#ifdef SD_BUILD_DLL
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "ggml.h"

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG,
    N_RNG_TYPES
};

static const char* rng_types_argument_str[] = {
    "std_default",
    "cuda",
};

SD_API rng_type_t sd_argument_to_rng_type(const char* str);
SD_API const char* sd_rng_type_to_argument(rng_type_t rng_type);

enum sample_method_t {
    EULER_A,
    EULER,
    HEUN,
    DPM2,
    DPMPP2S_A,
    DPMPP2M,
    DPMPP2Mv2,
    IPNDM,
    IPNDM_V,
    LCM,
    N_SAMPLE_METHODS
};

static const char* sample_methods_argument_str[] = {
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
};

SD_API sample_method_t sd_argument_to_sample_method(const char* str);
SD_API const char* sd_sample_method_to_argument(sample_method_t sample_method);

enum schedule_t {
    DEFAULT,
    DISCRETE,
    KARRAS,
    EXPONENTIAL,
    AYS,
    GITS,
    N_SCHEDULES
};

static const char* schedulers_argument_str[] = {
    "default",
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
};

SD_API schedule_t sd_argument_to_schedule(const char* str);
SD_API const char* sd_schedule_to_argument(schedule_t schedule);

enum sd_log_level_t {
    SD_LOG_DEBUG,
    SD_LOG_INFO,
    SD_LOG_WARN,
    SD_LOG_ERROR
};

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);

SD_API void sd_log_set(sd_log_cb_t sd_log_cb, void* data);
SD_API void sd_progress_set(sd_progress_cb_t cb, void* data);
SD_API int32_t get_num_physical_cores();
SD_API const char* sd_get_system_info();

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

typedef struct sd_ctx_t sd_ctx_t;

SD_API sd_ctx_t* new_sd_ctx(const char* model_path,
                            const char* clip_l_path,
                            const char* clip_g_path,
                            const char* t5xxl_path,
                            const char* diffusion_model_path,
                            const char* vae_path,
                            const char* taesd_path,
                            const char* control_net_path_c_str,
                            const char* lora_model_dir,
                            const char* embed_dir_c_str,
                            const char* stacked_id_embed_dir_c_str,
                            bool vae_decode_only,
                            bool vae_tiling,
                            bool free_params_immediately,
                            int n_threads,
                            enum ggml_type wtype,
                            enum rng_type_t rng_type,
                            enum schedule_t s,
                            bool keep_clip_on_cpu,
                            bool keep_control_net_cpu,
                            bool keep_vae_on_cpu,
                            int main_gpu = 0);

SD_API void sd_ctx_free(sd_ctx_t* sd_ctx);

SD_API sd_image_t* txt2img(sd_ctx_t* sd_ctx,
                           const char* prompt,
                           const char* negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           float guidance,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           int64_t seed,
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_strength,
                           bool normalize_input,
                           const char* input_id_images_path);

SD_API sd_image_t* img2img(sd_ctx_t* sd_ctx,
                           sd_image_t init_image,
                           const char* prompt,
                           const char* negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           float guidance,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           float strength,
                           int64_t seed,
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_strength,
                           bool normalize_input,
                           const char* input_id_images_path);

typedef struct upscaler_ctx_t upscaler_ctx_t;

SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                        int n_threads,
                                        enum ggml_type wtype);
SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

SD_API sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor);

SD_API bool convert(const char* input_path, const char* vae_path, const char* output_path, enum ggml_type output_type);

SD_API uint8_t* preprocess_canny(uint8_t* img,
                                 int width,
                                 int height,
                                 float high_threshold,
                                 float low_threshold,
                                 float weak,
                                 float strong,
                                 bool inverse);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__
