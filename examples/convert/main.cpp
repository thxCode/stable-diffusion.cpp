#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "ggml.h"
#include "json.hpp"
#include "model.h"
#include "stable-diffusion.h"
#include "util.h"

using json = nlohmann::json;

static int unknown(const char* flag) {
    fprintf(stderr, "Unknown argument: %s\n", flag);
    return 1;
}

static int missing(const char* flag) {
    throw std::invalid_argument("Missing argument: " + std::string(flag));
    return 1;
}

static int invalid(const char* flag) {
    throw std::invalid_argument("Invalid argument: " + std::string(flag));
    return 1;
}

struct convert_params {
    std::string model_path;
    std::string diffusion_model_file_path;
    std::string vae_model_file_path;
    std::string clip_l_model_file_path;
    std::string clip_g_model_file_path;
    std::string t5xxl_model_file_path;
    std::string output_file_path;
    ggml_type vae_output_type    = GGML_TYPE_COUNT;
    ggml_type clip_l_output_type = GGML_TYPE_COUNT;
    ggml_type clip_g_output_type = GGML_TYPE_COUNT;
    ggml_type t5xxl_output_type  = GGML_TYPE_COUNT;
    ggml_type output_type        = GGML_TYPE_F16;
};

static void convert_params_print_usage(int, char** argv, const convert_params& params) {
    printf("usage: %s MODEL [arguments]\n", argv[0]);
    printf("\n");
    printf("arguments:\n");
    printf("  -h, --help                         show this help message and exit\n");
    printf("  --diffusion-model                  path to diffusion model file, implicit ignoring vae model\n");
    printf("  --vae-model                        path to vae model file\n");
    printf("  --clip-l-model                     path to clip-l model file\n");
    printf("  --clip-g-model                     path to clip-g model file\n");
    printf("  --t5xxl-model                      path to t5xxl model file\n");
    printf("  --outfile                          path to write to\n");
    printf("  --vae-outtype                      output format of vae model, reuse --outtype if not specified\n");
    printf("  --clip-l-outtype                   output format of clip_l model, reuse --outtype if not specified\n");
    printf("  --clip-g-outtype                   output format of clip_g model, reuse --outtype if not specified\n");
    printf("  --t5xxl-outtype                    output format of t5xxl model, reuse --outtype if not specified\n");
    printf("  --outtype                          output format, select from fp32;fp16;q8_0;q5_1;q5_0;q4_1;q4_0;q4_k;q3_k;q2_k\n");
}

static ggml_type convert_str_to_ggml_type(const std::string& str) {
    if (str == "fp32") {
        return GGML_TYPE_F32;
    } else if (str == "fp16") {
        return GGML_TYPE_F16;
    } else if (str == "q8_0") {
        return GGML_TYPE_Q8_0;
    } else if (str == "q5_1") {
        return GGML_TYPE_Q5_1;
    } else if (str == "q5_0") {
        return GGML_TYPE_Q5_0;
    } else if (str == "q4_1") {
        return GGML_TYPE_Q4_1;
    } else if (str == "q4_0") {
        return GGML_TYPE_Q4_0;
    } else if (str == "q4_k") {
        return GGML_TYPE_Q4_K;
    } else if (str == "q3_k") {
        return GGML_TYPE_Q3_K;
    } else if (str == "q2_k") {
        return GGML_TYPE_Q2_K;
    }
    return GGML_TYPE_COUNT;
}

static bool convert_params_parse(int argc, char** argv, convert_params& params) {
    try {
        for (int i = 1; i < argc;) {
            const char* flag = argv[i++];

            if (*flag != '-') {
                if (i == 2) {
                    params.model_path = std::string(flag);
                }
                continue;
            }

            if (!strcmp(flag, "-h") || !strcmp(flag, "--help")) {
                convert_params_print_usage(argc, argv, params);
                exit(0);
            }

            if (!strcmp(flag, "--diffusion-model")) {
                if (i == argc) {
                    missing("--diffusion-model");
                }
                params.diffusion_model_file_path = std::string(argv[i++]);
                continue;
            }

            if (!strcmp(flag, "--vae-model")) {
                if (i == argc) {
                    missing("--vae-model");
                }
                params.vae_model_file_path = std::string(argv[i++]);
                continue;
            }

            if (!strcmp(flag, "--clip-l-model")) {
                if (i == argc) {
                    missing("--clip-l-model");
                }
                params.clip_l_model_file_path = std::string(argv[i++]);
                continue;
            }

            if (!strcmp(flag, "--clip-g-model")) {
                if (i == argc) {
                    missing("--clip-g-model");
                }
                params.clip_g_model_file_path = std::string(argv[i++]);
                continue;
            }

            if (!strcmp(flag, "--t5xxl-model")) {
                if (i == argc) {
                    missing("--t5xxl-model");
                }
                params.t5xxl_model_file_path = std::string(argv[i++]);
                continue;
            }

            if (!strcmp(flag, "--outfile")) {
                if (i == argc) {
                    missing("--outfile");
                }
                params.output_file_path = std::string(argv[i++]);
                continue;
            }

            if (!strcmp(flag, "--vae-outtype")) {
                if (i == argc) {
                    missing("--vae-outtype");
                }
                const char* outtype    = argv[i++];
                params.vae_output_type = convert_str_to_ggml_type(outtype);
                if (params.vae_output_type >= GGML_TYPE_COUNT) {
                    invalid("--vae-outtype");
                }
                continue;
            }

            if (!strcmp(flag, "--clip-l-outtype")) {
                if (i == argc) {
                    missing("--clip-l-outtype");
                }
                const char* outtype       = argv[i++];
                params.clip_l_output_type = convert_str_to_ggml_type(outtype);
                if (params.clip_l_output_type >= GGML_TYPE_COUNT) {
                    invalid("--clip-l-outtype");
                }
                continue;
            }

            if (!strcmp(flag, "--clip-g-outtype")) {
                if (i == argc) {
                    missing("--clip-g-outtype");
                }
                const char* outtype       = argv[i++];
                params.clip_g_output_type = convert_str_to_ggml_type(outtype);
                if (params.clip_g_output_type >= GGML_TYPE_COUNT) {
                    invalid("--clip-g-outtype");
                }
                continue;
            }

            if (!strcmp(flag, "--t5xxl-outtype")) {
                if (i == argc) {
                    missing("--t5xxl-outtype");
                }
                const char* outtype      = argv[i++];
                params.t5xxl_output_type = convert_str_to_ggml_type(outtype);
                if (params.t5xxl_output_type >= GGML_TYPE_COUNT) {
                    invalid("--t5xxl-outtype");
                }
                continue;
            }

            if (!strcmp(flag, "--outtype")) {
                if (i == argc) {
                    missing("--outtype");
                }
                const char* outtype = argv[i++];
                params.output_type  = convert_str_to_ggml_type(outtype);
                if (params.output_type >= GGML_TYPE_COUNT) {
                    invalid("--outtype");
                }
                continue;
            }

            unknown(flag);
        }
    } catch (const std::invalid_argument& ex) {
        fprintf(stderr, "%s\n", ex.what());
        return false;
    }

    if (params.model_path.empty()) {
        fprintf(stderr, "error: the following arguments are required: MODEL\n");
        convert_params_print_usage(argc, argv, params);
        exit(1);
    }

    if (params.output_file_path.empty()) {
        auto pos  = params.model_path.find_last_of('/');
        auto name = (pos != std::string::npos) ? params.model_path.substr(pos + 1) : "output";
        auto type = std::string(ggml_type_name(params.output_type));
        std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return std::toupper(c); });
        params.output_file_path = name + "-" + type + ".gguf";
    }

    return true;
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    if (!log) {
        return;
    }

    const char* level_str;
    switch (level) {
        case SD_LOG_DEBUG:
            level_str = "D";
            break;
        case SD_LOG_INFO:
            level_str = "I";
            break;
        case SD_LOG_WARN:
            level_str = "W";
            break;
        case SD_LOG_ERROR:
            level_str = "E";
            break;
        default: /* Potential future-proofing */
            level_str = "?";
            break;
    }

    fprintf(stderr, "%s ", level_str);
    fputs(log, stderr);
    fflush(stderr);
}

json load_json(const std::string& path) {
    std::ifstream jfile(path);
    if (!jfile.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    json j;
    jfile >> j;
    jfile.close();

    return j;
}

int convert_sd3(const convert_params& params, const SDVersion ver) {
    ModelLoader loader;
    bool loaded = false;

    if (params.clip_l_model_file_path.empty()) {
        loaded = loader.init_from_safetensors_file(params.model_path, "text_encoder/model", params.clip_l_output_type, "te.");
    } else {
        loaded = loader.init_from_file(params.clip_l_model_file_path, "te.");
    }
    if (!loaded) {
        LOG_ERROR("Failed to load text encoder model");
        return 1;
    }

    if (params.clip_g_model_file_path.empty()) {
        loaded = loader.init_from_safetensors_file(params.model_path, "text_encoder_2/model", params.clip_g_output_type, "te1.");
    } else {
        loaded = loader.init_from_file(params.clip_g_model_file_path, "te1.");
    }
    if (!loaded) {
        LOG_ERROR("Failed to load text encoder 2 model");
        return 1;
    }

    if (params.t5xxl_model_file_path.empty()) {
        loaded = loader.init_from_safetensors_file(params.model_path, "text_encoder_3/model", params.t5xxl_output_type, "te2.");
    } else {
        loaded = loader.init_from_file(params.t5xxl_model_file_path, "te2.");
    }
    if (!loaded) {
        LOG_ERROR("Failed to load text encoder 3 model");
        return 1;
    }

    bool ignore_vae = false;
    if (!params.diffusion_model_file_path.empty()) {
        ignore_vae = true;
    }

    if (!ignore_vae || !params.vae_model_file_path.empty()) {
        if (params.vae_model_file_path.empty()) {
            loaded = loader.init_from_safetensors_file(params.model_path, "vae/diffusion_pytorch_model", params.vae_output_type, "vae.");
        } else {
            loaded = loader.init_from_file(params.vae_model_file_path, "vae.");
        }
        if (!loaded) {
            LOG_ERROR("Failed to load vae model");
            return 1;
        }
    }

    if (params.diffusion_model_file_path.empty()) {
        loaded = loader.init_from_safetensors_file(params.model_path, "transformer/diffusion_pytorch_model", params.output_type, "transformer.");
    } else {
        loaded = loader.init_from_file(params.diffusion_model_file_path);
    }
    if (!loaded) {
        LOG_ERROR("Failed to load transformer model");
        return 1;
    }

    return !loader.save_to_gguf_file(params.output_file_path,
                                     params.output_type,
                                     params.vae_output_type,
                                     params.clip_l_output_type,
                                     params.clip_g_output_type,
                                     params.t5xxl_output_type);
}

int convert_flux(const convert_params& params, const SDVersion ver) {
    ModelLoader loader;
    bool loaded = false;

    if (params.clip_l_model_file_path.empty()) {
        loaded = loader.init_from_safetensors_file(params.model_path, "text_encoder/model", params.clip_l_output_type, "te.");
    } else {
        loaded = loader.init_from_file(params.clip_l_model_file_path, "te.");
    }
    if (!loaded) {
        LOG_ERROR("Failed to load text encoder model");
        return 1;
    }

    if (params.t5xxl_model_file_path.empty()) {
        loaded = loader.init_from_safetensors_file(params.model_path, "text_encoder_2/model", params.t5xxl_output_type, "te1.");
    } else {
        loaded = loader.init_from_file(params.t5xxl_model_file_path, "te1.");
    }
    if (!loaded) {
        LOG_ERROR("Failed to load text encoder 2 model");
        return 1;
    }

    bool ignore_vae = false;
    if (!params.diffusion_model_file_path.empty()) {
        ignore_vae = true;
    }

    if (!ignore_vae || !params.vae_model_file_path.empty()) {
        if (params.vae_model_file_path.empty()) {
            loaded = loader.init_from_safetensors_file(params.model_path, "vae/diffusion_pytorch_model", params.vae_output_type, "vae.");
        } else {
            loaded = loader.init_from_file(params.vae_model_file_path, "vae.");
        }
        if (!loaded) {
            LOG_ERROR("Failed to load vae model");
            return 1;
        }
    }

    if (params.diffusion_model_file_path.empty()) {
        if (ver == VERSION_FLUX_DEV) {
            loaded = loader.init_from_safetensors_file(params.model_path, "flux1-dev", params.output_type, "transformer.");
        } else {
            loaded = loader.init_from_safetensors_file(params.model_path, "flux1-schnell", params.output_type, "transformer.");
        }
    } else {
        loaded = loader.init_from_file(params.diffusion_model_file_path, "model.diffusion_model.");
    }
    if (!loaded) {
        LOG_ERROR("Failed to load transformer model");
        return 1;
    }

    return !loader.save_to_gguf_file(params.output_file_path,
                                     params.output_type,
                                     params.vae_output_type,
                                     params.clip_l_output_type,
                                     params.clip_g_output_type,
                                     params.t5xxl_output_type);
}

int convert_sdxl(const convert_params& params, const SDVersion ver) {
    ModelLoader loader;
    bool loaded = false;

    if (params.clip_l_model_file_path.empty()) {
        if (is_directory(path_join(params.model_path, "text_encoder"))) {
            loaded = loader.init_from_safetensors_file(params.model_path, "text_encoder/model", params.clip_l_output_type, "te.");
        } else {
            loaded = true;
        }
    } else {
        loaded = loader.init_from_file(params.clip_l_model_file_path, "te.");
    }
    if (!loaded) {
        LOG_ERROR("Failed to load text encoder model");
        return 1;
    }

    if (params.clip_g_model_file_path.empty()) {
        loaded = loader.init_from_safetensors_file(params.model_path, "text_encoder_2/model", params.clip_g_output_type, "te1.");
    } else {
        loaded = loader.init_from_file(params.clip_g_model_file_path, "te1.");
    }
    if (!loaded) {
        LOG_ERROR("Failed to load text encoder 2 model");
        return 1;
    }

    bool ignore_vae = false;
    if (!params.diffusion_model_file_path.empty()) {
        ignore_vae = true;
    }

    if (!ignore_vae || !params.vae_model_file_path.empty()) {
        if (params.vae_model_file_path.empty()) {
            loaded = loader.init_from_safetensors_file(params.model_path, "vae/diffusion_pytorch_model", params.vae_output_type, "vae.");
        } else {
            loaded = loader.init_from_file(params.vae_model_file_path, "vae.");
        }
        if (!loaded) {
            LOG_ERROR("Failed to load vae model");
            return 1;
        }
    }

    if (params.diffusion_model_file_path.empty()) {
        loaded = loader.init_from_safetensors_file(params.model_path, "unet/diffusion_pytorch_model", params.output_type, "unet.");
    } else {
        loaded = loader.init_from_file(params.diffusion_model_file_path);
    }
    if (!loaded) {
        LOG_ERROR("Failed to load unet model");
        return 1;
    }

    return !loader.save_to_gguf_file(params.output_file_path,
                                     params.output_type,
                                     params.vae_output_type,
                                     params.clip_l_output_type,
                                     params.clip_g_output_type,
                                     params.t5xxl_output_type);
}

int convert_sd(const convert_params& params, const SDVersion ver) {
    ModelLoader loader;
    bool loaded = false;

    if (params.clip_l_model_file_path.empty()) {
        loaded = loader.init_from_safetensors_file(params.model_path, "text_encoder/model", params.clip_l_output_type, "te.");
    } else {
        loaded = loader.init_from_file(params.clip_l_model_file_path, "te.");
    }
    if (!loaded) {
        LOG_ERROR("Failed to load text encoder model");
        return 1;
    }

    bool ignore_vae = false;
    if (!params.diffusion_model_file_path.empty()) {
        ignore_vae = true;
    }

    if (!ignore_vae || !params.vae_model_file_path.empty()) {
        if (params.vae_model_file_path.empty()) {
            loaded = loader.init_from_safetensors_file(params.model_path, "vae/diffusion_pytorch_model", params.vae_output_type, "vae.");
        } else {
            loaded = loader.init_from_file(params.vae_model_file_path, "vae.");
        }
        if (!loaded) {
            LOG_ERROR("Failed to load vae model");
            return 1;
        }
    }

    if (params.diffusion_model_file_path.empty()) {
        loaded = loader.init_from_safetensors_file(params.model_path, "unet/diffusion_pytorch_model", params.output_type, "unet.");
    } else {
        loaded = loader.init_from_file(params.diffusion_model_file_path);
    }
    if (!loaded) {
        LOG_ERROR("Failed to load unet model");
        return 1;
    }

    return !loader.save_to_gguf_file(params.output_file_path,
                                     params.output_type,
                                     params.vae_output_type,
                                     params.clip_l_output_type,
                                     params.clip_g_output_type,
                                     params.t5xxl_output_type);
}

int convert_file(const convert_params& params) {
    ModelLoader loader;
    bool loaded = false;

    loaded = loader.init_from_file(params.model_path, "");
    if (!loaded) {
        LOG_ERROR("Failed to load file");
        return 1;
    }

    return !loader.save_to_gguf_file(params.output_file_path, params.output_type);
}

int main(int argc, char** argv) {
    convert_params params;
    if (!convert_params_parse(argc, argv, params)) {
        convert_params_print_usage(argc, argv, params);
        return 1;
    }

    sd_set_log_callback(sd_log_cb, nullptr);

    if (!is_directory(params.model_path)) {
        return convert_file(params);
    }

    auto model_index_path = path_join(params.model_path, "model_index.json");
    if (!file_exists(model_index_path)) {
        LOG_ERROR("Model index.json is not found: %s", model_index_path.c_str());
        return 1;
    }

    SDVersion ver    = VERSION_COUNT;
    auto model_index = load_json(model_index_path);
    auto class_name  = model_index.at("_class_name").get<std::string>();
    if (class_name == "StableDiffusion3Pipeline") {
        auto transformer_config_path = path_join(params.model_path, "transformer/config.json");
        if (!file_exists(transformer_config_path)) {
            LOG_ERROR("Transformer config.json is not found: %s", transformer_config_path.c_str());
            return 1;
        }
        auto transformer_config = load_json(transformer_config_path);
        auto num_layers         = transformer_config.at("num_layers").get<int>();
        if (num_layers == 38) {
            ver = VERSION_SD3_5_8B;
        } else {
            auto pos_embed_max_size = transformer_config.at("pos_embed_max_size").get<int>();
            if (pos_embed_max_size == 384) {
                ver = VERSION_SD3_5_2B;
            } else {
                ver = VERSION_SD3_2B;
            }
        }
    } else if (class_name == "FluxPipeline") {
        auto transformer_config_path = path_join(params.model_path, "transformer/config.json");
        if (!file_exists(transformer_config_path)) {
            LOG_ERROR("Transformer config.json is not found: %s", transformer_config_path.c_str());
            return 1;
        }
        auto transformer_config = load_json(transformer_config_path);
        ver                     = VERSION_FLUX_SCHNELL;
        if (transformer_config.contains("guidance_embeds")) {
            auto guidance_embeds = transformer_config.at("guidance_embeds").get<bool>();
            if (guidance_embeds) {
                ver             = VERSION_FLUX_DEV;
                auto num_layers = transformer_config.at("num_layers").get<int>();
                if (num_layers == 8) {
                    ver = VERSION_FLUX_LITE;
                }
            }
        }
    } else if (class_name == "StableDiffusionXLPipeline") {
        ver = VERSION_SDXL;
    } else if (class_name == "StableDiffusionXLImg2ImgPipeline") {
        ver = VERSION_SDXL_REFINER;
    } else if (class_name == "StableDiffusionPipeline") {
        auto text_encoder_config_path = path_join(params.model_path, "text_encoder/config.json");
        if (!file_exists(text_encoder_config_path)) {
            LOG_ERROR("Text encoder config.json is not found: %s", text_encoder_config_path.c_str());
            return 1;
        }
        auto text_encoder_config = load_json(text_encoder_config_path);
        auto hidden_size         = text_encoder_config.at("hidden_size").get<int>();
        if (hidden_size == 1024) {
            ver = VERSION_SD2;
        } else {
            ver = VERSION_SD1;
        }
    }

    if (ver == VERSION_COUNT) {
        LOG_ERROR("Unknown model version");
        return 1;
    }

    switch (ver) {
        case VERSION_SD3_2B:
        case VERSION_SD3_5_2B:
        case VERSION_SD3_5_8B:
            return convert_sd3(params, ver);
        case VERSION_FLUX_DEV:
        case VERSION_FLUX_SCHNELL:
        case VERSION_FLUX_LITE:
            return convert_flux(params, ver);
        case VERSION_SDXL:
        case VERSION_SDXL_REFINER:
            return convert_sdxl(params, ver);
        case VERSION_SD2:
        case VERSION_SD1:
            return convert_sd(params, ver);
        default:
            LOG_ERROR("Unsupported model version");
            return 1;
    }
}