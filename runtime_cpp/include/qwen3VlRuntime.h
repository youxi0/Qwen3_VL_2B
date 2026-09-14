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
    double timeToFirstTokenMs{0.0};
    double timePerOutputTokenMs{0.0};
    double decodeTokensPerSecond{0.0};
    double visionLatencyMs{0.0};
    double prefillLatencyMs{0.0};
    double decodeLatencyMs{0.0};
};

//! 本项目专用的单设备 Qwen3-VL 运行时。
//!
//! 每个实例独占一个非阻塞 CUDA 流，并让模型资源常驻内存。
//! 同一实例上的 generate() 调用会串行执行。
class Qwen3VlRuntime
{
public:
    //! 创建并初始化运行时，依赖库异常不会向调用方传播。
    static std::unique_ptr<Qwen3VlRuntime> create(RuntimeConfig const& config, std::string& error) noexcept;

    ~Qwen3VlRuntime() noexcept;

    Qwen3VlRuntime(Qwen3VlRuntime const&) = delete;
    Qwen3VlRuntime& operator=(Qwen3VlRuntime const&) = delete;

    //! 执行一个批大小为 1 的请求，并返回生成文本及耗时数据。
    GenerationResponse generate(GenerationRequest const& request) noexcept;

    //! 返回初始化期间是否成功捕获解码 CUDA Graph。
    bool cudaGraphCaptured() const noexcept;

private:
    class Impl;

    explicit Qwen3VlRuntime(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> mImpl;
};

} // 命名空间 qwen3_vl
