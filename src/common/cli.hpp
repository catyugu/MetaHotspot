#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace mhs::cli {

    // 命令行解析结果的结构化表达。所有标志都是命名、顺序无关的；
    // 不接受位置参数（这是有意为之，见 README 与 CONTEXT.md）。
    struct Options {
        std::string input; // 必填：输入 XML
        std::string output_vtu = "./output.vtu";
        std::string output_xml = "./output.xml";
        std::optional<std::string> fluid_overlay; // 显式覆盖；仅在传入 --fluid-overlay 时填充，未传入则不执行流体相关逻辑
        std::string log_file = "metahotspot.log";
        bool console_log = true; // --no-console-log 翻转
        std::string program_name; // argv[0] 的 basename
    };

    enum class ParseStatus {
        Ok,
        HelpRequested,
        Error,
    };

    struct ParseResult {
        ParseStatus status = ParseStatus::Error;
        std::optional<Options> options; // 仅在 status == Ok 时填充
        std::string message; // 用法文本 / 版本文本 / 错误文本
    };

    // 解析 argv。argv[0] 是程序名（始终视为已提供）。
    // 所有标志顺序无关；位置参数被拒绝（错误返回 Error）。
    ParseResult parse(int argc, char** argv);

    std::string usage_text(std::string_view program_name);

    // 启发式：<input>.xml -> <input>_additional.xml。
    // 纯函数，便于测试；当前主流程不再自动调用，仅在用户显式需要时使用。
    std::string infer_fluid_overlay_path(std::string_view input);

} // namespace mhs::cli