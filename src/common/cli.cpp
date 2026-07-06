#include "cli.hpp"

#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>

namespace mhs::cli {

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

    namespace {

        bool read_value(int& i, int argc, char** argv, const char* flag, std::string& out, std::string& err)
        {
            if (i + 1 >= argc) {
                err = std::string("invalid argument '") + flag + "': missing value";
                return false;
            }
            out = argv[++i];
            return true;
        }

    } // namespace

    ParseResult parse(int argc, char** argv)
    {
        ParseResult result;

        if (argc <= 0 || argv == nullptr || argv[0] == nullptr) {
            result.status = ParseStatus::Error;
            result.message = "no argv provided";
            return result;
        }

        // std::filesystem::path covers both POSIX '/' and Windows '\\' separators.
        result.program_name = std::filesystem::path(argv[0]).filename().string();

        Options opts;
        std::string err;

        for (int i = 1; i < argc; ++i) {
            const char* cur = argv[i];

            if (cur == nullptr) {
                continue;
            }
            std::string_view arg = cur;

            if (arg == "--help") {
                result.status = ParseStatus::HelpRequested;
                result.message = usage_text(result.program_name);
                return result;
            }
            if (arg == "--input") {
                std::string v;
                if (!read_value(i, argc, argv, cur, v, err)) {
                    result.status = ParseStatus::Error;
                    result.message = err;
                    return result;
                }
                if (!opts.input.empty()) {
                    result.status = ParseStatus::Error;
                    result.message = "--input specified more than once";
                    return result;
                }
                opts.input = std::move(v);
            }
            else if (arg == "--output-vtu") {
                std::string v;
                if (!read_value(i, argc, argv, cur, v, err)) {
                    result.status = ParseStatus::Error;
                    result.message = err;
                    return result;
                }
                opts.output_vtu = std::move(v);
            }
            else if (arg == "--output-xml") {
                std::string v;
                if (!read_value(i, argc, argv, cur, v, err)) {
                    result.status = ParseStatus::Error;
                    result.message = err;
                    return result;
                }
                opts.output_xml = std::move(v);
            }
            else if (arg == "--fluid-overlay") {
                std::string v;
                if (!read_value(i, argc, argv, cur, v, err)) {
                    result.status = ParseStatus::Error;
                    result.message = err;
                    return result;
                }
                opts.fluid_overlay = std::move(v);
            }
            else if (arg == "--log-file") {
                std::string v;
                if (!read_value(i, argc, argv, cur, v, err)) {
                    result.status = ParseStatus::Error;
                    result.message = err;
                    return result;
                }
                opts.log_file = std::move(v);
            }
            else if (arg == "--no-console-log") {
                opts.console_log = false;
            }
            else if (arg.size() > 1 && arg[0] == '-') {
                result.status = ParseStatus::Error;
                result.message = std::string("unknown flag '") + cur + "'";
                return result;
            }
            else {
                result.status = ParseStatus::Error;
                result.message = std::string("positional argument '") + cur
                    + "' is not accepted; use named flags (try --help)";
                return result;
            }
        }

        if (opts.input.empty()) {
            result.status = ParseStatus::Error;
            result.message = "required flag --input <file> is missing";
            return result;
        }

        result.status = ParseStatus::Ok;
        result.options = std::move(opts);
        return result;
    }

} // namespace mhs::cli
