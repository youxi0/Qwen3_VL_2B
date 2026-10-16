#include "qwen3VlRuntime.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

struct Arguments
{
    qwen3_vl::RuntimeConfig runtime;
    qwen3_vl::GenerationRequest request;
    int32_t repeat{1};
    std::string jsonOutput;
};

void printUsage(char const* program)
{
    std::cout << "Usage: " << program
              << " --engine-dir DIR --multimodal-engine-dir DIR --prompt TEXT [options]\n\n"
                 "Options:\n"
                 "  --plugin PATH             Edge-LLM plugin shared library\n"
                 "  --image PATH              Add an image; may be repeated\n"
                 "  --system TEXT             Optional system prompt\n"
                 "  --max-new-tokens N        Default: 32\n"
                 "  --temperature T           Default: 0 (greedy)\n"
                 "  --top-p P                 Default: 1\n"
                 "  --top-k K                 Default: 1\n"
                 "  --stop TEXT               Stop string; may be repeated\n"
                 "  --enable-thinking         Enable model thinking mode\n"
                 "  --no-image-resize         Disable runtime resizing for all images\n"
                 "  --no-cuda-graph           Disable decode CUDA Graph capture\n"
                 "  --warmup N                Full-request warmup count; default: 0\n"
                 "  --repeat N                Run the same request N times; default: 1\n"
                 "  --json-output PATH        Write machine-readable metrics to JSON\n"
                 "  --verbose                 Enable verbose Edge-LLM logging\n"
                 "  -h, --help                Show this help\n";
}

std::string takeValue(int& index, int argc, char** argv, std::string const& option)
{
    if (index + 1 >= argc)
    {
        throw std::runtime_error("missing value for " + option);
    }
    return argv[++index];
}

int64_t parseInteger(std::string const& value, std::string const& option)
{
    size_t consumed = 0;
    int64_t result = 0;
    try
    {
        result = std::stoll(value, &consumed);
    }
    catch (std::exception const&)
    {
        throw std::runtime_error("invalid integer for " + option + ": " + value);
    }
    if (consumed != value.size())
    {
        throw std::runtime_error("invalid integer for " + option + ": " + value);
    }
    return result;
}

float parseFloat(std::string const& value, std::string const& option)
{
    size_t consumed = 0;
    float result = 0.0F;
    try
    {
        result = std::stof(value, &consumed);
    }
    catch (std::exception const&)
    {
        throw std::runtime_error("invalid number for " + option + ": " + value);
    }
    if (consumed != value.size() || !std::isfinite(result))
    {
        throw std::runtime_error("invalid number for " + option + ": " + value);
    }
    return result;
}

int32_t parseInt32(std::string const& value, std::string const& option)
{
    int64_t const parsed = parseInteger(value, option);
    if (parsed < std::numeric_limits<int32_t>::min() || parsed > std::numeric_limits<int32_t>::max())
    {
        throw std::runtime_error("integer outside int32 range for " + option + ": " + value);
    }
    return static_cast<int32_t>(parsed);
}

Arguments parseArguments(int argc, char** argv)
{
    Arguments args;
    bool resizeImages = true;
    std::vector<std::string> imagePaths;

    for (int index = 1; index < argc; ++index)
    {
        std::string const option = argv[index];
        if (option == "--engine-dir")
        {
            args.runtime.llmEngineDir = takeValue(index, argc, argv, option);
        }
        else if (option == "--multimodal-engine-dir")
        {
            args.runtime.multimodalEngineDir = takeValue(index, argc, argv, option);
        }
        else if (option == "--plugin")
        {
            args.runtime.pluginLibrary = takeValue(index, argc, argv, option);
        }
        else if (option == "--prompt")
        {
            args.request.prompt = takeValue(index, argc, argv, option);
        }
        else if (option == "--system")
        {
            args.request.systemPrompt = takeValue(index, argc, argv, option);
        }
        else if (option == "--image")
        {
            imagePaths.push_back(takeValue(index, argc, argv, option));
        }
        else if (option == "--stop")
        {
            args.request.stopStrings.push_back(takeValue(index, argc, argv, option));
        }
        else if (option == "--max-new-tokens")
        {
            args.request.maxNewTokens = parseInteger(takeValue(index, argc, argv, option), option);
        }
        else if (option == "--temperature")
        {
            args.request.temperature = parseFloat(takeValue(index, argc, argv, option), option);
        }
        else if (option == "--top-p")
        {
            args.request.topP = parseFloat(takeValue(index, argc, argv, option), option);
        }
        else if (option == "--top-k")
        {
            args.request.topK = parseInteger(takeValue(index, argc, argv, option), option);
        }
        else if (option == "--warmup")
        {
            args.runtime.warmupRuns = parseInt32(takeValue(index, argc, argv, option), option);
        }
        else if (option == "--repeat")
        {
            args.repeat = parseInt32(takeValue(index, argc, argv, option), option);
        }
        else if (option == "--json-output")
        {
            args.jsonOutput = takeValue(index, argc, argv, option);
        }
        else if (option == "--enable-thinking")
        {
            args.request.enableThinking = true;
        }
        else if (option == "--no-image-resize")
        {
            resizeImages = false;
        }
        else if (option == "--no-cuda-graph")
        {
            args.runtime.enableCudaGraph = false;
        }
        else if (option == "--verbose")
        {
            args.runtime.verboseLogging = true;
        }
        else if (option == "-h" || option == "--help")
        {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else
        {
            throw std::runtime_error("unknown option: " + option);
        }
    }

    if (args.runtime.llmEngineDir.empty())
    {
        throw std::runtime_error("--engine-dir is required");
    }
    if (args.request.prompt.empty())
    {
        throw std::runtime_error("--prompt is required");
    }
    if (args.runtime.multimodalEngineDir.empty())
    {
        throw std::runtime_error("--multimodal-engine-dir is required for Qwen3-VL");
    }
    if (args.repeat <= 0)
    {
        throw std::runtime_error("--repeat must be positive");
    }

    for (std::string& path : imagePaths)
    {
        args.request.images.push_back({std::move(path), resizeImages});
    }
    return args;
}

} // 匿名命名空间

int main(int argc, char** argv)
{
    Arguments args;
    try
    {
        args = parseArguments(argc, argv);
    }
    catch (std::exception const& error)
    {
        std::cerr << "Argument error: " << error.what() << "\n\n";
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string initializationError;
    auto runtime = qwen3_vl::Qwen3VlRuntime::create(args.runtime, initializationError);
    if (!runtime)
    {
        std::cerr << "Runtime initialization failed: " << initializationError << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Runtime ready; CUDA Graph: " << (runtime->cudaGraphCaptured() ? "captured" : "disabled/unavailable")
              << '\n';

    nlohmann::json runs = nlohmann::json::array();
    for (int32_t run = 0; run < args.repeat; ++run)
    {
        qwen3_vl::GenerationResponse response = runtime->generate(args.request);
        if (!response.ok)
        {
            std::cerr << "Inference failed on run " << run + 1 << ": " << response.error << '\n';
            return EXIT_FAILURE;
        }

        std::cout << "\n[run " << run + 1 << "] " << response.text << "\n"
                  << "input_tokens=" << response.inputTokens << ", generated_tokens=" << response.tokenIds.size()
                  << ", finish=" << response.finishReason << ", latency_ms=" << std::fixed << std::setprecision(2)
                  << response.latencyMs << ", tokens_per_second=" << response.tokensPerSecond
                  << ", ttft_ms=" << response.timeToFirstTokenMs << ", tpot_ms=" << response.timePerOutputTokenMs
                  << ", decode_tokens_per_second=" << response.decodeTokensPerSecond
                  << ", vision_latency_ms=" << response.visionLatencyMs
                  << ", prefill_latency_ms=" << response.prefillLatencyMs
                  << ", decode_latency_ms=" << response.decodeLatencyMs << '\n';

        runs.push_back({{"run", run + 1}, {"text", response.text}, {"token_ids", response.tokenIds},
            {"input_tokens", response.inputTokens}, {"generated_tokens", response.tokenIds.size()},
            {"finish_reason", response.finishReason}, {"latency_ms", response.latencyMs},
            {"tokens_per_second", response.tokensPerSecond}, {"ttft_ms", response.timeToFirstTokenMs},
            {"tpot_ms", response.timePerOutputTokenMs},
            {"decode_tokens_per_second", response.decodeTokensPerSecond},
            {"vision_latency_ms", response.visionLatencyMs}, {"prefill_latency_ms", response.prefillLatencyMs},
            {"decode_latency_ms", response.decodeLatencyMs}});
    }

    if (!args.jsonOutput.empty())
    {
        std::filesystem::path const outputPath{args.jsonOutput};
        if (outputPath.has_parent_path())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        std::ofstream output(outputPath);
        if (!output)
        {
            std::cerr << "Failed to open JSON output: " << outputPath << '\n';
            return EXIT_FAILURE;
        }
        nlohmann::json const document{{"cuda_graph_captured", runtime->cudaGraphCaptured()}, {"runs", runs}};
        output << document.dump(2) << '\n';
    }
    return EXIT_SUCCESS;
}
