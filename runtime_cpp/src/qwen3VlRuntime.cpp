#include "qwen3VlRuntime.h"

#include "common/logger.h"
#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"

#include <NvInfer.h>
#include <chrono>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace qwen3_vl
{
namespace
{

using PluginHandle = std::unique_ptr<void, trt_edgellm::DlDeleter>;

void checkCuda(cudaError_t status, char const* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void requireFile(std::filesystem::path const& path, char const* label)
{
    if (!std::filesystem::is_regular_file(path))
    {
        throw std::runtime_error(std::string(label) + " not found: " + path.string());
    }
}

PluginHandle loadPlugin(std::string const& path)
{
    if (path.empty())
    {
        PluginHandle handle = trt_edgellm::loadEdgellmPluginLib();
        if (!handle)
        {
            throw std::runtime_error(
                "failed to load the Edge-LLM plugin; pass pluginLibrary or set EDGELLM_PLUGIN_PATH");
        }
        return handle;
    }

    requireFile(path, "Edge-LLM plugin library");
    PluginHandle handle(dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE));
    if (!handle)
    {
        char const* error = dlerror();
        throw std::runtime_error("failed to load Edge-LLM plugin " + path + ": "
            + (error != nullptr ? std::string(error) : std::string("unknown dlopen error")));
    }

    using InitPlugins = bool (*)(void*, char const*);
    auto initPlugins = reinterpret_cast<InitPlugins>(dlsym(handle.get(), "initEdgellmPlugins"));
    if (initPlugins == nullptr || !initPlugins(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), ""))
    {
        throw std::runtime_error("failed to initialize Edge-LLM TensorRT plugins from " + path);
    }
    return handle;
}

trt_edgellm::rt::Message textMessage(std::string role, std::string text)
{
    trt_edgellm::rt::Message message;
    message.role = std::move(role);
    message.contents.push_back({"text", std::move(text)});
    return message;
}

trt_edgellm::rt::LLMGenerationRequest makeRequest(GenerationRequest const& input)
{
    if (input.prompt.empty())
    {
        throw std::runtime_error("prompt must not be empty");
    }
    if (input.maxNewTokens <= 0)
    {
        throw std::runtime_error("maxNewTokens must be positive");
    }
    if (input.temperature < 0.0F)
    {
        throw std::runtime_error("temperature must be non-negative");
    }
    if (input.topP <= 0.0F || input.topP > 1.0F)
    {
        throw std::runtime_error("topP must be in (0, 1]");
    }
    if (input.topK < 0)
    {
        throw std::runtime_error("topK must be non-negative");
    }

    trt_edgellm::rt::LLMGenerationRequest request{};
    request.temperature = input.temperature;
    request.topP = input.topP;
    request.topK = input.topK;
    request.maxGenerateLength = input.maxNewTokens;
    request.applyChatTemplate = true;
    request.addGenerationPrompt = true;
    request.enableThinking = input.enableThinking;

    trt_edgellm::rt::LLMGenerationRequest::Request item;
    if (!input.systemPrompt.empty())
    {
        item.messages.push_back(textMessage("system", input.systemPrompt));
    }

    trt_edgellm::rt::Message userMessage;
    userMessage.role = "user";
    for (ImageInput const& imageInput : input.images)
    {
        requireFile(imageInput.path, "input image");
        auto image = trt_edgellm::rt::imageUtils::loadImageFromFile(imageInput.path);
        if (image.buffer == nullptr)
        {
            throw std::runtime_error("failed to decode input image: " + imageInput.path);
        }
        image.doResize = imageInput.resize;
        item.imageBuffers.push_back(std::move(image));
        userMessage.contents.push_back({"image", imageInput.path});
    }
    userMessage.contents.push_back({"text", input.prompt});
    item.messages.push_back(std::move(userMessage));
    item.stopStrings = input.stopStrings;
    request.requests.push_back(std::move(item));
    return request;
}

} // namespace

class Qwen3VlRuntime::Impl
{
public:
    explicit Impl(RuntimeConfig config)
        : mConfig(std::move(config))
    {
        validateConfig();
        trt_edgellm::gLogger.setLevel(mConfig.verboseLogging ? nvinfer1::ILogger::Severity::kVERBOSE
                                                            : nvinfer1::ILogger::Severity::kINFO);
        mPluginHandle = loadPlugin(mConfig.pluginLibrary);
        checkCuda(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

        try
        {
            std::unordered_map<std::string, std::string> const loraWeights;
            mRuntime = std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
                mConfig.llmEngineDir, mConfig.multimodalEngineDir, loraWeights, mStream);
            if (mConfig.enableCudaGraph)
            {
                mCudaGraphCaptured = mRuntime->captureDecodingCUDAGraph(mStream);
            }
        }
        catch (...)
        {
            cudaStreamDestroy(mStream);
            mStream = nullptr;
            throw;
        }
    }

    ~Impl() noexcept
    {
        mRuntime.reset();
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
            mStream = nullptr;
        }
        mPluginHandle.reset();
    }

    GenerationResponse generate(GenerationRequest const& input) noexcept
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        GenerationResponse result;
        auto start = std::chrono::steady_clock::now();

        try
        {
            if (!input.images.empty() && mConfig.multimodalEngineDir.empty())
            {
                throw std::runtime_error("multimodalEngineDir is required for image requests");
            }
            if (!mWarmupComplete)
            {
                runWarmup(input);
                mWarmupComplete = true;
                start = std::chrono::steady_clock::now();
            }

            auto request = makeRequest(input);
            trt_edgellm::rt::LLMGenerationResponse response;
            if (!mRuntime->handleRequest(request, response, mStream))
            {
                throw std::runtime_error("LLMInferenceRuntime::handleRequest returned false");
            }
            checkCuda(cudaStreamSynchronize(mStream), "cudaStreamSynchronize");

            if (response.outputTexts.size() != 1U || response.outputIds.size() != 1U)
            {
                throw std::runtime_error("runtime returned an unexpected response batch size");
            }

            result.ok = true;
            result.text = std::move(response.outputTexts.front());
            result.tokenIds = std::move(response.outputIds.front());
            if (!response.finishReasons.empty())
            {
                result.finishReason = trt_edgellm::rt::finishReasonName(response.finishReasons.front());
            }
            if (!response.inputTokenCounts.empty())
            {
                result.inputTokens = response.inputTokenCounts.front();
            }
        }
        catch (std::exception const& error)
        {
            result.error = error.what();
        }
        catch (...)
        {
            result.error = "unknown runtime failure";
        }

        auto const end = std::chrono::steady_clock::now();
        result.latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
        if (result.ok && result.latencyMs > 0.0)
        {
            result.tokensPerSecond = static_cast<double>(result.tokenIds.size()) * 1000.0 / result.latencyMs;
        }
        return result;
    }

    bool cudaGraphCaptured() const noexcept
    {
        return mCudaGraphCaptured;
    }

private:
    void validateConfig() const
    {
        std::filesystem::path const llmDir{mConfig.llmEngineDir};
        if (!std::filesystem::is_directory(llmDir))
        {
            throw std::runtime_error("LLM engine directory not found: " + llmDir.string());
        }
        requireFile(llmDir / "llm.engine", "LLM engine");
        requireFile(llmDir / "config.json", "LLM config");
        requireFile(llmDir / "embedding.safetensors", "embedding table");
        requireFile(llmDir / "tokenizer.json", "tokenizer");

        if (mConfig.multimodalEngineDir.empty())
        {
            throw std::runtime_error("multimodal engine directory is required for Qwen3-VL");
        }
        std::filesystem::path const multimodalDir{mConfig.multimodalEngineDir};
        bool const nestedVisual = std::filesystem::is_regular_file(multimodalDir / "visual" / "visual.engine");
        bool const directVisual = std::filesystem::is_regular_file(multimodalDir / "visual.engine");
        if (!nestedVisual && !directVisual)
        {
            throw std::runtime_error(
                "visual.engine not found below multimodal engine directory: " + multimodalDir.string());
        }
        if (mConfig.warmupRuns < 0)
        {
            throw std::runtime_error("warmupRuns must be non-negative");
        }
    }

    void runWarmup(GenerationRequest const& input)
    {
        for (int32_t run = 0; run < mConfig.warmupRuns; ++run)
        {
            auto request = makeRequest(input);
            trt_edgellm::rt::LLMGenerationResponse response;
            if (!mRuntime->handleRequest(request, response, mStream))
            {
                throw std::runtime_error("warmup request failed");
            }
        }
        checkCuda(cudaStreamSynchronize(mStream), "warmup cudaStreamSynchronize");
    }

    RuntimeConfig mConfig;
    PluginHandle mPluginHandle;
    cudaStream_t mStream{nullptr};
    std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> mRuntime;
    std::mutex mMutex;
    bool mCudaGraphCaptured{false};
    bool mWarmupComplete{false};
};

std::unique_ptr<Qwen3VlRuntime> Qwen3VlRuntime::create(RuntimeConfig const& config, std::string& error) noexcept
{
    try
    {
        auto impl = std::make_unique<Impl>(config);
        error.clear();
        return std::unique_ptr<Qwen3VlRuntime>(new Qwen3VlRuntime(std::move(impl)));
    }
    catch (std::exception const& exception)
    {
        error = exception.what();
    }
    catch (...)
    {
        error = "unknown runtime initialization failure";
    }
    return nullptr;
}

Qwen3VlRuntime::Qwen3VlRuntime(std::unique_ptr<Impl> impl) noexcept
    : mImpl(std::move(impl))
{
}

Qwen3VlRuntime::~Qwen3VlRuntime() noexcept = default;

GenerationResponse Qwen3VlRuntime::generate(GenerationRequest const& request) noexcept
{
    if (!mImpl)
    {
        GenerationResponse response;
        response.error = "runtime is not initialized";
        return response;
    }
    return mImpl->generate(request);
}

bool Qwen3VlRuntime::cudaGraphCaptured() const noexcept
{
    return mImpl && mImpl->cudaGraphCaptured();
}

} // namespace qwen3_vl
