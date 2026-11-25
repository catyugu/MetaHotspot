"""Build, test, install and package MetaHotspot."""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def check_sdk(prefix, build, config, generator):
    headers = prefix / "include/metahotspot"
    for header in headers.rglob("*.hpp"):
        text = header.read_text(encoding="utf-8")
        for dependency in ("Eigen", "tinyxml2"):
            if dependency.lower() in text.lower():
                raise RuntimeError(f"{dependency} leaked into public header: {header}")
    linear_headers = {p.name for p in (headers / "linear").glob("*.hpp")}
    if linear_headers != {"solver.hpp"}:
        raise RuntimeError(
            f"Unexpected public linear backend headers: {linear_headers}"
        )
    for package in prefix.rglob("*MetaHotspot*.cmake"):
        text = package.read_text(encoding="utf-8")
        for dependency in ("Eigen", "tinyxml2"):
            if dependency.lower() in text.lower():
                raise RuntimeError(f"{dependency} leaked into installed package: {package}")
    for path in prefix.rglob("*"):
        if "tinyxml2" in path.name.lower():
            raise RuntimeError(f"Private XML dependency installed separately: {path}")
    consumer = build / "sdk-consumer"
    consumer.mkdir(exist_ok=True)
    (consumer / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(SDKConsumer LANGUAGES CXX)\n"
        "find_package(MetaHotspot CONFIG REQUIRED)\n"
        "add_executable(consumer main.cpp)\n"
        "target_link_libraries(consumer PRIVATE mhs::metahotspot mhs::erom_extraction)\n",
        encoding="utf-8",
    )
    (consumer / "main.cpp").write_text(
        '#include "solver/solver.hpp"\n'
        '#include "core/matrix.hpp"\n'
        '#include "linear/solver.hpp"\n'
        '#include "model/erom.hpp"\n'
        '#include "erom/extraction.hpp"\n'
        "int main(int argc, char** argv) {\n"
        "  if (argc == 3) {\n"
        "    auto asset = mhs::model::read_erom(argv[1]);\n"
        "    mhs::app::erom::write_erom(argv[2], *asset);\n"
        "  }\n"
        "  mhs::core::CSRMatrix<double> a(1, 1, {0, 1}, {0}, {2.0});\n"
        "  mhs::core::Matrix<double> dense(1, 1);\n"
        "  dense(0, 0) = a.coeff(0, 0);\n"
        "  mhs::core::Vector<double> b{6.0}, x{0.0};\n"
        "  auto solver = mhs::linear::create_solver();\n"
        "  solver->compute(a.view());\n"
        "  return solver->solve(b, x).success ? 0 : 1;\n"
        "}\n",
        encoding="utf-8",
    )
    run(
        "cmake",
        "-S",
        consumer,
        "-B",
        consumer / "build",
        "-G",
        generator,
        f"-DCMAKE_BUILD_TYPE={config}",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        "-DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE",
        "-DCMAKE_DISABLE_FIND_PACKAGE_tinyxml2=TRUE",
    )
    run("cmake", "--build", consumer / "build", "--config", config)


def check_erom_cli(executable, build):
    project = build / "erom-input.xml"
    output = build / "erom-output.xml"
    block = (
        "<Block><Name>die</Name><MaterialName>solid</MaterialName>"
        "<HeatSourceId>heater</HeatSourceId><VolumetricHeatSource>0</VolumetricHeatSource>"
        "<AllRects><Rect><Add_sub>true</Add_sub><WidthExpression>1</WidthExpression>"
        "<HeightExpression>1</HeightExpression><XExpression>{x}</XExpression>"
        "<YExpression>0</YExpression></Rect></AllRects></Block>"
    )
    xml = (
        '<Structure xmlns:i="http://www.w3.org/2001/XMLSchema-instance">'
        "<LengthUnit>M</LengthUnit><InitialTemperature>300</InitialTemperature>"
        "<Materials><Entry><Key>solid</Key><Value><ThermalConductivity>2</ThermalConductivity>"
        "<Density>1</Density><SpecificHeatCapacity>4</SpecificHeatCapacity>"
        "<FluidMaterial>false</FluidMaterial></Value></Entry></Materials>"
        "<HeatSources><HeatSource><Id>heater</Id><Name>shared</Name>"
        "<PowerDensity>0</PowerDensity></HeatSource></HeatSources>"
        "<Layers><Layer><ThicknessExpression>1</ThicknessExpression><Blocks>"
        + block.format(x=5)
        + block.format(x=6)
        + "</Blocks></Layer></Layers>"
        "<Boundaries><Boundary><Name>right</Name><FaceKeys><string>X|E|7|0|1|0|1</string></FaceKeys>"
        '<ThermalBoundary i:type="ThirdTypeThermalBoundary">'
        "<ConvectionCoefficient>5</ConvectionCoefficient>"
        "<EnvironmentTemperature>310</EnvironmentTemperature></ThermalBoundary></Boundary></Boundaries>"
        "<GeneratedMesh><XArray><double>5</double><double>6</double><double>7</double></XArray>"
        "<YArray><double>0</double><double>1</double></YArray>"
        "<ZArray><double>0</double><double>1</double></ZArray></GeneratedMesh>"
        '<EromExtraction version="1"><BoundaryGroups><BoundaryGroup index="0">'
        "<HtcMin>1</HtcMin><HtcMax>10</HtcMax></BoundaryGroup></BoundaryGroups></EromExtraction>"
        "<ObservePoints3D><ObservePoint3D><Name>probe</Name><X>5.5</X><Y>0.5</Y>"
        "<Z>0.5</Z></ObservePoint3D></ObservePoints3D></Structure>"
    )
    project.write_text(xml, encoding="utf-8")
    run(executable, "--input", project, "--output", output)
    root = ET.parse(output).getroot()
    if root.tag != "Erom" or root.attrib.get("version") != "1":
        raise RuntimeError("invalid EROM XML root")
    sources = root.findall("./Sources/Source")
    if len(sources) != 1 or sources[0].attrib.get("name") != "shared":
        raise RuntimeError("explicit shared zero-power source was not preserved")
    interface = root.find("./Interfaces/Interface")
    if interface is None or float(interface.attrib["coordinate"]) != 2.0:
        raise RuntimeError("EROM geometry was not translated")
    before = output.read_bytes()
    project.write_text(
        xml.replace("<HtcMin>1</HtcMin>", "<HtcMin>0</HtcMin>"), encoding="utf-8"
    )
    failure = subprocess.run(
        [str(executable), "--input", str(project), "--output", str(output)], check=False
    )
    if failure.returncode == 0 or output.read_bytes() != before:
        raise RuntimeError(
            "invalid extraction succeeded or overwrote the existing asset"
        )
    project.unlink()
    output.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--generator", default="Ninja")
    parser.add_argument("--cmake-arg", action="append", default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    build = Path(args.build_dir).resolve()
    prefix = build / "package"
    flags = ["-G", args.generator, f"-DCMAKE_BUILD_TYPE={args.config}", *args.cmake_arg]
    run("cmake", "-S", root, "-B", build, *flags)
    run("cmake", "--build", build, "--config", args.config, "--parallel", "4")
    run("ctest", "--test-dir", build, "-C", args.config, "--output-on-failure")
    if args.config == "Release":
        executable = build / "apps/metahotspot/metahotspot"
        if sys.platform == "win32":
            executable = executable.with_suffix(".exe")
        os.environ["MHS_EXECUTABLE"] = str(executable)
        os.environ["PYTHONIOENCODING"] = "utf-8"
        run(sys.executable, root / "run_cases.py")
        erom = build / "apps/metahotspot-erom/metahotspot-erom"
        if sys.platform == "win32":
            erom = erom.with_suffix(".exe")
        check_erom_cli(erom, build)
        run("cmake", "--install", build, "--config", args.config, "--prefix", prefix)
        run(prefix / "bin" / erom.name, "--help")
        check_sdk(prefix, build, args.config, args.generator)
        run(
            "cpack",
            "--config",
            build / "CPackConfig.cmake",
            "-C",
            args.config,
            "-B",
            build,
        )


if __name__ == "__main__":
    main()
