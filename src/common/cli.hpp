#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace mhs::cli {

    // Named, order-independent flags; positional arguments are not accepted.
    struct Options {
        std::string input; // required
        std::string output_vtu = "./output.vtu";
        std::string output_xml = "./output.xml";
        std::optional<std::string> fluid_overlay; // absent => skip fluid logic
        std::string log_file = "metahotspot.log";
        bool console_log = true; // --no-console-log flips this
    };

    enum class ParseStatus {
        Ok,
        HelpRequested,
        Error,
    };

    struct ParseResult {
        ParseStatus status = ParseStatus::Error;
        // Populated only when status == Ok.
        std::optional<Options> options;
        // Basename of argv[0], set on every path so callers can render usage text.
        std::string program_name;
        // Help text on HelpRequested, error description on Error, empty on Ok.
        std::string message;
    };

    ParseResult parse(int argc, char** argv);

    std::string usage_text(std::string_view program_name);

} // namespace mhs::cli
