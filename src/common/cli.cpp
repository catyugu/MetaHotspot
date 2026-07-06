#include "cli.hpp"

#include <sstream>
#include <string>
#include <string_view>

namespace mhs::cli {

    namespace {

        // 取 argv[0] 的 basename 部分（Windows 与 POSIX 通用）。
        std::string basename_of(std::string_view path)
        {
            auto pos = path.find_last_of("/\\");
            if (pos == std::string_view::npos) {
                return std::string(path);
            }
            return std::string(path.substr(pos + 1));
        }

        // 简单的字符串相等比较（避免 <string_view> 上 operator== 在某些 stdlib
        // 上的细微不一致；这里走 std::string 比较，行为清晰）。
        bool eq(std::string_view a, const char* b) { return a == b; }

        // 期望当前标志还需要一个值；若缺失则填入错误并返回 false。
        bool require_value(int& i, int argc, char** argv, std::string& out, std::string& err)
        {
            if (i + 1 >= argc) {
                err = std::string("missing value for ") + argv[i];
                return false;
            }
            out = argv[++i];
            return true;
        }

        // 拼装错误：前缀 + 当前标志名 + 详细信息，便于用户定位。
        std::string flag_error(char* current, std::string_view detail)
        {
            std::string msg = "invalid argument '";
            msg += current ? current : "";
            msg += "': ";
            msg += detail;
            return msg;
        }

    } // namespace

    std::string infer_fluid_overlay_path(std::string_view input)
    {
        auto dot = input.rfind('.');
        if (dot != std::string_view::npos) {
            return std::string(input.substr(0, dot)) + "_additional" + std::string(input.substr(dot));
        }
        return std::string(input) + "_additional";
    }

    std::string usage_text(std::string_view program_name)
    {
        std::ostringstream os;
        os << "Usage: " << program_name << " [OPTIONS]\n"
           << "\n"
           << "  --input <file>          Input XML describing the simulation (required).\n"
           << "  --output-vtu <file>     Output VTU path (default: ./output.vtu).\n"
           << "  --output-xml <file>     Output XML path (default: ./output.xml).\n"
           << "  --fluid-overlay <file>  Explicit fluid-overlay XML; only when this flag is given will fluid-related logic run.\n"
           << "  --log-file <file>       Log file path (default: metahotspot.log).\n"
           << "  --no-console-log        Disable console logging.\n"
           << "  --help                  Print this help and exit 0.\n"
           << "\n"
           << "All flags are order-independent. Positional arguments are not accepted.\n";
        return os.str();
    }

    ParseResult parse(int argc, char** argv)
    {
        ParseResult result;

        if (argc <= 0 || argv == nullptr) {
            result.status = ParseStatus::Error;
            result.message = "no argv provided";
            return result;
        }

        Options opts;
        opts.program_name = basename_of(argv[0]);

        // 帮助 / 版本与"已经看过 input"是互斥的；先扫一遍处理短路的 helpn。
        for (int i = 1; i < argc; ++i) {
            if (eq(argv[i], "--help")) {
                result.status = ParseStatus::HelpRequested;
                result.message = usage_text(opts.program_name);
                return result;
            }
        }

        std::string err;
        bool input_set = false;

        for (int i = 1; i < argc; ++i) {
            const char* cur = argv[i];

            if (eq(cur, "--input")) {
                std::string v;
                if (!require_value(i, argc, argv, v, err)) {
                    result.status = ParseStatus::Error;
                    result.message = flag_error(argv[i], err);
                    return result;
                }
                if (input_set) {
                    result.status = ParseStatus::Error;
                    result.message = "--input specified more than once";
                    return result;
                }
                opts.input = std::move(v);
                input_set = true;
            }
            else if (eq(cur, "--output-vtu")) {
                std::string v;
                if (!require_value(i, argc, argv, v, err)) {
                    result.status = ParseStatus::Error;
                    result.message = flag_error(argv[i], err);
                    return result;
                }
                opts.output_vtu = std::move(v);
            }
            else if (eq(cur, "--output-xml")) {
                std::string v;
                if (!require_value(i, argc, argv, v, err)) {
                    result.status = ParseStatus::Error;
                    result.message = flag_error(argv[i], err);
                    return result;
                }
                opts.output_xml = std::move(v);
            }
            else if (eq(cur, "--fluid-overlay")) {
                std::string v;
                if (!require_value(i, argc, argv, v, err)) {
                    result.status = ParseStatus::Error;
                    result.message = flag_error(argv[i], err);
                    return result;
                }
                opts.fluid_overlay = std::move(v);
            }
            else if (eq(cur, "--log-file")) {
                std::string v;
                if (!require_value(i, argc, argv, v, err)) {
                    result.status = ParseStatus::Error;
                    result.message = flag_error(argv[i], err);
                    return result;
                }
                opts.log_file = std::move(v);
            }
            else if (eq(cur, "--no-console-log")) {
                opts.console_log = false;
            }
            else if (cur != nullptr && cur[0] == '-') {
                result.status = ParseStatus::Error;
                result.message = flag_error(argv[i], "unknown flag");
                return result;
            }
            else {
                // 位置参数被显式拒绝（"不要向后兼容"）。
                result.status = ParseStatus::Error;
                result.message = std::string("positional argument '") + (cur ? cur : "")
                    + "' is not accepted; use named flags (try --help)";
                return result;
            }
        }

        if (!input_set) {
            result.status = ParseStatus::Error;
            result.message = "required flag --input <file> is missing";
            return result;
        }

        // fluid overlay 路径：仅在显式传入 --fluid-overlay 时填充；
        // 未传入时保持 std::nullopt，主流程据此跳过所有流体相关逻辑。
        // 不再由 input 自动推导。

        result.status = ParseStatus::Ok;
        result.options = std::move(opts);
        return result;
    }

} // namespace mhs::cli