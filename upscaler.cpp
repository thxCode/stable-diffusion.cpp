#include "esrgan.hpp"
#include "ggml_extend.hpp"
#include "model.h"
#include "stable-diffusion.h"

struct UpscalerGGML {
    ggml_backend_t backend    = NULL;  // general backend
    ggml_type model_data_type = GGML_TYPE_F16;
    std::shared_ptr<ESRGAN> esrgan_upscaler;
    std::string esrgan_path;
    int n_threads;

    UpscalerGGML(int n_threads)
        : n_threads(n_threads) {
    }

    bool load_from_file(const std::string& esrgan_path, const std::vector<std::string>& rpc_servers, const float* tensor_split) {
        ggml_log_set(ggml_log_callback_default, nullptr);

        std::vector<ggml_backend_dev_t> devices;

        if (!rpc_servers.empty()) {
            ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
            if (!rpc_reg) {
                LOG_ERROR("failed to find RPC backend");
                return false;
            }

            typedef ggml_backend_dev_t (*ggml_backend_rpc_add_device_t)(const char* endpoint);
            ggml_backend_rpc_add_device_t ggml_backend_rpc_add_device_fn = (ggml_backend_rpc_add_device_t)ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_device");
            if (!ggml_backend_rpc_add_device_fn) {
                LOG_ERROR("failed to find RPC device add function");
                return false;
            }

            for (const std::string& server : rpc_servers) {
                ggml_backend_dev_t dev = ggml_backend_rpc_add_device_fn(server.c_str());
                if (dev) {
                    devices.push_back(dev);
                } else {
                    LOG_ERROR("failed to add RPC device for server '%s'", server.c_str());
                    return false;
                }
            }
        }

        // use all available devices
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // skip CPU backends since they are handled separately
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU:
                    devices.push_back(dev);
                    break;
            }
        }

        for (auto* dev : devices) {
            size_t free, total;  // NOLINT
            ggml_backend_dev_memory(dev, &free, &total);
            LOG_INFO("using device %s (%s) - %zu MiB free", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), free / 1024 / 1024);
        }

        // build GPU devices buffer list
        std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>> gpu_devices;
        {
            const bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + devices.size(), [](float x) { return x == 0.0f; });
            // add GPU buffer types
            for (size_t i = 0; i < devices.size(); ++i) {
                if (!all_zero && tensor_split[i] <= 0.0f) {
                    continue;
                }
                ggml_backend_device* dev = devices[i];
                gpu_devices.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
            }
        }

        // initialize the backend
        if (gpu_devices.empty()) {
            // no GPU devices available
            backend = ggml_backend_cpu_init();
        } else {
            // use the first GPU device: device 0
            backend = ggml_backend_dev_init(gpu_devices[0].first, nullptr);
        }

        ModelLoader model_loader;
        if (!model_loader.init_from_file(esrgan_path)) {
            LOG_ERROR("init model loader from file failed: '%s'", esrgan_path.c_str());
        }
        model_loader.set_wtype_override(model_data_type);

        LOG_INFO("Upscaler weight type: %s", ggml_type_name(model_data_type));
        esrgan_upscaler = std::make_shared<ESRGAN>(backend, model_loader.tensor_storages_types);
        if (!esrgan_upscaler->load_from_file(esrgan_path)) {
            return false;
        }
        return true;
    }

    sd_image_t upscale(sd_image_t input_image, uint32_t upscale_factor) {
        // upscale_factor, unused for RealESRGAN_x4plus_anime_6B.pth
        sd_image_t upscaled_image = {0, 0, 0, NULL};
        int output_width          = (int)input_image.width * esrgan_upscaler->scale;
        int output_height         = (int)input_image.height * esrgan_upscaler->scale;
        LOG_INFO("upscaling from (%i x %i) to (%i x %i)",
                 input_image.width, input_image.height, output_width, output_height);

        struct ggml_init_params params;
        params.mem_size = output_width * output_height * 3 * sizeof(float) * 2;
        params.mem_size += 2 * ggml_tensor_overhead();
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        // draft context
        struct ggml_context* upscale_ctx = ggml_init(params);
        if (!upscale_ctx) {
            LOG_ERROR("ggml_init() failed");
            return upscaled_image;
        }
        LOG_DEBUG("upscale work buffer size: %.2f MB", params.mem_size / 1024.f / 1024.f);
        ggml_tensor* input_image_tensor = ggml_new_tensor_4d(upscale_ctx, GGML_TYPE_F32, input_image.width, input_image.height, 3, 1);
        sd_image_to_tensor(input_image.data, input_image_tensor);

        ggml_tensor* upscaled = ggml_new_tensor_4d(upscale_ctx, GGML_TYPE_F32, output_width, output_height, 3, 1);
        auto on_tiling        = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
            esrgan_upscaler->compute(n_threads, in, &out);
        };
        int64_t t0 = ggml_time_ms();
        sd_tiling(input_image_tensor, upscaled, esrgan_upscaler->scale, esrgan_upscaler->tile_size, 0.25f, on_tiling);
        esrgan_upscaler->free_compute_buffer();
        ggml_tensor_clamp(upscaled, 0.f, 1.f);
        uint8_t* upscaled_data = sd_tensor_to_image(upscaled);
        ggml_free(upscale_ctx);
        int64_t t3 = ggml_time_ms();
        LOG_INFO("input_image_tensor upscaled, taking %.2fs", (t3 - t0) / 1000.0f);
        upscaled_image = {
            (uint32_t)output_width,
            (uint32_t)output_height,
            3,
            upscaled_data,
        };
        return upscaled_image;
    }
};

struct upscaler_ctx_t {
    UpscalerGGML* upscaler = NULL;
};

upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path_c_str,
                                 int n_threads,
                                 const char* rpc_servers,
                                 const float* tensor_splits) {
    upscaler_ctx_t* upscaler_ctx = (upscaler_ctx_t*)malloc(sizeof(upscaler_ctx_t));
    if (upscaler_ctx == NULL) {
        return NULL;
    }
    std::string esrgan_path(esrgan_path_c_str);
    std::vector<std::string> rpc_servers_vec;
    if (rpc_servers != nullptr && rpc_servers[0] != '\0') {
        // split the servers set them into model->rpc_servers
        std::string servers(rpc_servers);
        size_t pos = 0;
        while ((pos = servers.find(',')) != std::string::npos) {
            std::string server = servers.substr(0, pos);
            rpc_servers_vec.push_back(server);
            servers.erase(0, pos + 1);
        }
        rpc_servers_vec.push_back(servers);
    }

    upscaler_ctx->upscaler = new UpscalerGGML(n_threads);
    if (upscaler_ctx->upscaler == NULL) {
        return NULL;
    }

    if (!upscaler_ctx->upscaler->load_from_file(esrgan_path, rpc_servers_vec, tensor_splits)) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = NULL;
        free(upscaler_ctx);
        return NULL;
    }
    return upscaler_ctx;
}

sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor) {
    return upscaler_ctx->upscaler->upscale(input_image, upscale_factor);
}

void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx->upscaler != NULL) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = NULL;
    }
    free(upscaler_ctx);
}
