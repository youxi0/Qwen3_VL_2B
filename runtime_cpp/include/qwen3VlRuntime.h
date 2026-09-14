#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qwen3_vl
{

struct RuntimeConfig
{
    std::string llmEngineDir;
    std::string multimodalEngineDir;
    std::string pluginLibrary;
    bool enableCudaGraph{true};
    bool verboseLogging{false};
    int32_t warmupRuns{0};
};

struct ImageInput
{
    std::string path;
    bool resize{true};
};

struct GenerationRequest
{
    std::string prompt;
    std::string systemPrompt;
    std::vector<ImageInput> images;
    std::vector<std::string> stopStrings;
    int64_t maxNewTokens{32};
    float temperature{0.0F};
    float topP{1.0F};
    int64_t topK{1};
    bool enableThinking{false};
};

struct GenerationResponse
{
    bool ok{false};
    std::string error;
    std::string text;
    std::string finishReason;
    std::vector<int32_t> tokenIds;
    int32_t inputTokens{0};
    double latencyMs{0.0};
    double tokensPerSecond{0.0};
};

//! Project-owned, single-device Qwen3-VL runtime.
//!
//! One instance owns one non-blocking CUDA stream and keeps the model resources
//! resident. Calls to generate() are serialized on that instance.
class Qwen3VlRuntime
{
public:
    //! Create and initialize a runtime without propagating dependency exceptions.
    static std::unique_ptr<Qwen3VlRuntime> create(RuntimeConfig const& config, std::string& error) noexcept;

    ~Qwen3VlRuntime() noexcept;

    Qwen3VlRuntime(Qwen3VlRuntime const&) = delete;
    Qwen3VlRuntime& operator=(Qwen3VlRuntime const&) = delete;

    //! Run one batch-one request and return generated text and timing data.
    GenerationResponse generate(GenerationRequest const& request) noexcept;

    //! Return whether decode CUDA Graph capture succeeded during initialization.
    bool cudaGraphCaptured() const noexcept;

private:
    class Impl;

    explicit Qwen3VlRuntime(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> mImpl;
};

} // namespace qwen3_vl
