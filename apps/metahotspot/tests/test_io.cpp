#include "core/vector.hpp"
#include "model/model_definition.hpp"
#include "pta/io.hpp"
#include "pta_test_utils.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

// Build a small in-memory XML string with/without ObservePoints3D block.
static std::string make_xml_with_observe_points(const std::string& obs_block)
{
    std::string body = R"(<?xml version="1.0" encoding="utf-8"?>
<Structure>
    <StudyType>Transient</StudyType>
    <Dimension>Dimension3D</Dimension>
    <LengthUnit>Mm</LengthUnit>
    <AmbientTemperature>300</AmbientTemperature>
    <InitialTemperature>300</InitialTemperature>
    <TransientStudyDuration>10</TransientStudyDuration>
    <TransientStudyTimeStep>1</TransientStudyTimeStep>
    <TransientTimeUnit>S</TransientTimeUnit>
    <OtherThermalBondary i:type="SecondType"><a:HeatFlux>0</a:HeatFlux></OtherThermalBondary>
    <Results>
        <a:anyType i:type="Result3D">
            <Mesh>
                <b:XArray><a:double>0</a:double><a:double>1</a:double></b:XArray>
                <b:YArray><a:double>0</a:double><a:double>1</a:double></b:YArray>
                <b:ZArray><a:double>0</a:double><a:double>1</a:double></b:ZArray>
            </Mesh>
            <Values>
                <SizeX>2</SizeX><SizeY>2</SizeY><SizeZ>2</SizeZ>
                <Data><a:double>0</a:double></Data>
            </Values>
        </a:anyType>
    </Results>
)";
    if (!obs_block.empty()) {
        body += "    " + obs_block + "\n";
    }
    body += "</Structure>\n";
    return body;
}

static std::filesystem::path tmp_dir()
{
    const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
    auto dir = std::filesystem::current_path() / "mhs_test_io" / test->name();
    std::filesystem::create_directories(dir);
    return dir;
}

static std::filesystem::path write_tmp_xml(const std::string& name, const std::string& contents)
{
    auto path = tmp_dir() / name;
    {
        std::ofstream out(path);
        out << contents;
    }
    return path;
}

static std::string read_file(const std::filesystem::path& path)
{
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

TEST(IoTest, ReadXmlParsesObservationPoints3D)
{
    std::string obs = R"(<ObservePoints3D>
        <ObservePoint3D><Name>p1</Name><X>1.5</X><Y>2.5</Y><Z>3.5</Z></ObservePoint3D>
        <ObservePoint3D><Name>p2</Name><X>0.1</X><Y>0.2</Y><Z>0.3</Z></ObservePoint3D>
    </ObservePoints3D>)";
    auto path = write_tmp_xml("io_obs_present.xml", make_xml_with_observe_points(obs));

    auto io_structure = mhs::app::pta::read_xml(path.string());
    ASSERT_EQ(io_structure.model.point_probes.size(), 2u);
    EXPECT_EQ(io_structure.model.point_probes[0].name, "p1");
    // PTA adapter resolves observation coordinates to SI metres.
    EXPECT_EQ(io_structure.model.point_probes[0].x, 0.0015);
    EXPECT_EQ(io_structure.model.point_probes[0].y, 0.0025);
    EXPECT_EQ(io_structure.model.point_probes[0].z, 0.0035);
    EXPECT_EQ(io_structure.model.point_probes[1].name, "p2");
    EXPECT_EQ(io_structure.model.point_probes[1].x, 0.0001);
    std::filesystem::remove(path);
}

TEST(IoTest, ReadXmlWithoutObservationPointsReturnsEmpty)
{
    auto path = write_tmp_xml("io_obs_absent.xml", make_xml_with_observe_points(""));
    auto io_structure = mhs::app::pta::read_xml(path.string());
    EXPECT_TRUE(io_structure.model.point_probes.empty()) << "No <ObservePoints3D> → empty vector";
    std::filesystem::remove(path);
}

TEST(IoTest, ReadXmlPreservesAppendOrderForBuilderAssembly)
{
    const std::string xml = R"(<?xml version="1.0" encoding="utf-8"?>
<Structure>
    <StudyType>Steady</StudyType>
    <LengthUnit>Mm</LengthUnit>
    <InitialTemperature>300</InitialTemperature>
    <Materials>
        <Entry><Key>background</Key><Value><FluidMaterial>false</FluidMaterial></Value></Entry>
        <Entry><Key>foreground</Key><Value><FluidMaterial>false</FluidMaterial></Value></Entry>
    </Materials>
    <Layers>
        <Layer>
            <ThicknessExpression>10</ThicknessExpression>
            <XOffsetExpression>0</XOffsetExpression>
            <YOffsetExpression>0</YOffsetExpression>
            <Blocks>
                <Block>
                    <MaterialName>background</MaterialName>
                    <VolumetricHeatSource>1</VolumetricHeatSource>
                    <XOffsetExpression>0</XOffsetExpression>
                    <YOffsetExpression>0</YOffsetExpression>
                    <AllRects>
                        <Rect><Add_sub>true</Add_sub><XExpression>0</XExpression><YExpression>0</YExpression><WidthExpression>10</WidthExpression><HeightExpression>10</HeightExpression></Rect>
                        <Rect><Add_sub>false</Add_sub><XExpression>5</XExpression><YExpression>0</YExpression><WidthExpression>5</WidthExpression><HeightExpression>10</HeightExpression></Rect>
                    </AllRects>
                </Block>
                <Block>
                    <MaterialName>foreground</MaterialName>
                    <VolumetricHeatSource>2</VolumetricHeatSource>
                    <XOffsetExpression>0</XOffsetExpression>
                    <YOffsetExpression>0</YOffsetExpression>
                    <AllRects>
                        <Rect><Add_sub>true</Add_sub><XExpression>0</XExpression><YExpression>0</YExpression><WidthExpression>5</WidthExpression><HeightExpression>10</HeightExpression></Rect>
                    </AllRects>
                </Block>
            </Blocks>
        </Layer>
    </Layers>
    <Boundaries>
        <Boundary><ThermalBoundary i:type="FirstType"><a:Temperature>310</a:Temperature></ThermalBoundary></Boundary>
        <Boundary><ThermalBoundary i:type="ThirdType"><a:ConvectionCoefficient>42</a:ConvectionCoefficient><a:EnvironmentTemperature>280</a:EnvironmentTemperature></ThermalBoundary></Boundary>
    </Boundaries>
</Structure>)";
    const auto path = write_tmp_xml("io_builder_order.xml", xml);

    const auto definition = mhs::app::pta::read_xml(path.string());
    std::filesystem::remove(path);

    ASSERT_EQ(definition.model.fom(0).volumes.size(), 2u);
    EXPECT_EQ(definition.model.materials[definition.model.fom(0).volumes[0].material].name, "background");
    EXPECT_EQ(definition.model.materials[definition.model.fom(0).volumes[1].material].name, "foreground");
    ASSERT_EQ(definition.model.fom(0).volumes[0].geometry.size(), 2u);
    EXPECT_EQ(definition.model.fom(0).volumes[0].geometry[0].operation, mhs::model::GeometryOperation::Add);
    EXPECT_EQ(definition.model.fom(0).volumes[0].geometry[1].operation, mhs::model::GeometryOperation::Subtract);
    ASSERT_EQ(definition.model.fom(0).thermal_boundaries.size(), 2u);
    EXPECT_TRUE(std::holds_alternative<mhs::model::DirichletBoundary>(
        definition.model.fom(0).thermal_boundaries[0].condition));
    EXPECT_TRUE(std::holds_alternative<mhs::model::ConvectionBoundary>(
        definition.model.fom(0).thermal_boundaries[1].condition));
}

TEST(IoTest, WriteXmlEmitsResult0DTransient)
{
    std::string input_xml = R"(<?xml version="1.0" encoding="utf-8"?>
<Structure>
    <StudyType>Transient</StudyType>
    <Dimension>Dimension3D</Dimension>
    <LengthUnit>Mm</LengthUnit>
    <AmbientTemperature>300</AmbientTemperature>
    <InitialTemperature>300</InitialTemperature>
    <TransientStudyDuration>10</TransientStudyDuration>
    <TransientStudyTimeStep>1</TransientStudyTimeStep>
    <TransientTimeUnit>S</TransientTimeUnit>
    <OtherThermalBondary i:type="SecondType"><a:HeatFlux>0</a:HeatFlux></OtherThermalBondary>
    <Results>
        <a:anyType i:type="Result3D">
            <Mesh>
                <b:XArray><a:double>0</a:double><a:double>1</a:double></b:XArray>
                <b:YArray><a:double>0</a:double><a:double>1</a:double></b:YArray>
                <b:ZArray><a:double>0</a:double><a:double>1</a:double></b:ZArray>
            </Mesh>
            <Values>
                <SizeX>2</SizeX><SizeY>2</SizeY><SizeZ>2</SizeZ>
                <Data><a:double>0</a:double></Data>
            </Values>
        </a:anyType>
    </Results>
</Structure>
)";
    auto in_path = write_tmp_xml("io_write_input.xml", input_xml);
    auto out_path = tmp_dir() / "io_write_output.xml";

    test_pta::Model model;
    model.output.cartesian_shape[0] = 2;
    model.output.cartesian_shape[1] = 2;
    model.output.cartesian_shape[2] = 2;
    mhs::core::Vector<double> node_temperature = {300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0};

    mhs::core::Vector<mhs::solver::ProbeTrace> traces;
    mhs::solver::ProbeTrace t1;
    t1.name = "probe_a";
    const mhs::core::Vector<double> times {0.0, 1.0, 2.0};
    t1.values = {300.0, 310.0, 320.0};
    traces.push_back(t1);

    mhs::app::pta::write_xml(
        in_path.string(), out_path.string(), model.output.cartesian_shape, node_temperature, traces, times);

    std::string out_xml = read_file(out_path);
    EXPECT_NE(out_xml.find("Result0DTransient"), std::string::npos);
    EXPECT_NE(out_xml.find("<PointName>probe_a</PointName>"), std::string::npos);
    EXPECT_NE(out_xml.find("<a:double>300.000000</a:double>"), std::string::npos);
    EXPECT_NE(out_xml.find("<a:double>320.000000</a:double>"), std::string::npos);

    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);
}

TEST(IoTest, WriteXmlUpdatesSamplingDimensions)
{
    const auto in_path = write_tmp_xml("io_dimensions.xml", make_xml_with_observe_points(""));
    const auto out_path = tmp_dir() / "io_dimensions_output.xml";
    test_pta::Model model;
    model.output.cartesian_shape = {2, 3, 4};
    const mhs::core::Vector<double> temperatures(24, 300.0);

    mhs::app::pta::write_xml(in_path.string(), out_path.string(), model.output.cartesian_shape, temperatures, {});

    const auto xml = read_file(out_path);
    EXPECT_NE(xml.find("<SizeX>2</SizeX>"), std::string::npos);
    EXPECT_NE(xml.find("<SizeY>3</SizeY>"), std::string::npos);
    EXPECT_NE(xml.find("<SizeZ>4</SizeZ>"), std::string::npos);
    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);
}

TEST(IoTest, WriteXmlEmptyTracesLeavesNoProbeBlocks)
{
    std::string input_xml = R"(<?xml version="1.0" encoding="utf-8"?>
<Structure>
    <StudyType>Steady</StudyType>
    <Dimension>Dimension3D</Dimension>
    <LengthUnit>Mm</LengthUnit>
    <AmbientTemperature>300</AmbientTemperature>
    <InitialTemperature>300</InitialTemperature>
    <OtherThermalBondary i:type="SecondType"><a:HeatFlux>0</a:HeatFlux></OtherThermalBondary>
    <Results>
        <a:anyType i:type="Result3D">
            <Mesh>
                <b:XArray><a:double>0</a:double><a:double>1</a:double></b:XArray>
                <b:YArray><a:double>0</a:double><a:double>1</a:double></b:YArray>
                <b:ZArray><a:double>0</a:double><a:double>1</a:double></b:ZArray>
            </Mesh>
            <Values>
                <SizeX>2</SizeX><SizeY>2</SizeY><SizeZ>2</SizeZ>
                <Data><a:double>0</a:double></Data>
            </Values>
        </a:anyType>
    </Results>
</Structure>
)";
    auto in_path = write_tmp_xml("io_write_steady.xml", input_xml);
    auto out_path = tmp_dir() / "io_write_steady_output.xml";

    test_pta::Model model;
    model.output.cartesian_shape[0] = 2;
    model.output.cartesian_shape[1] = 2;
    model.output.cartesian_shape[2] = 2;
    mhs::core::Vector<double> node_temperature = {300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0};

    mhs::core::Vector<mhs::solver::ProbeTrace> traces;
    const mhs::core::Vector<double> times;
    mhs::app::pta::write_xml(
        in_path.string(), out_path.string(), model.output.cartesian_shape, node_temperature, traces, times);

    std::string out_xml = read_file(out_path);
    EXPECT_EQ(out_xml.find("Result0DTransient"), std::string::npos)
        << "Empty traces should not emit Result0DTransient blocks";

    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);
}

// Build a minimal in-memory XML that contains one material with the given
// ThermalConductivity text.
static std::string make_xml_with_thermal_conductivity(const std::string& conductivity_text)
{
    std::string body = R"(<?xml version="1.0" encoding="utf-8"?>
<Structure>
    <StudyType>Steady</StudyType>
    <Dimension>Dimension3D</Dimension>
    <LengthUnit>Mm</LengthUnit>
    <AmbientTemperature>300</AmbientTemperature>
    <InitialTemperature>300</InitialTemperature>
    <OtherThermalBondary i:type="SecondTypeThermalBoundary"><a:HeatFlux>0</a:HeatFlux></OtherThermalBondary>
    <Materials xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
        <a:KeyValueOfstringMaterialGyu7GfTz>
            <a:Key>mat</a:Key>
            <a:Value>
                <SpecificHeatCapacity>385</SpecificHeatCapacity>
                <ThermalConductivity>)";
    body += conductivity_text;
    body += R"(</ThermalConductivity>
                <Density>8920</Density>
                <FluidMaterial>false</FluidMaterial>
            </a:Value>
        </a:KeyValueOfstringMaterialGyu7GfTz>
    </Materials>
</Structure>
)";
    return body;
}

TEST(IoTest, ReadXmlThermalConductivitySingleExpressionTrimsWhitespace)
{
    auto path = write_tmp_xml("io_thermal_conductivity_trim.xml", make_xml_with_thermal_conductivity("  5  "));
    auto io_structure = mhs::app::pta::read_xml(path.string());
    ASSERT_EQ(io_structure.model.materials.size(), 1u);
    EXPECT_EQ(io_structure.model.materials[0].name, "mat");
    EXPECT_EQ(io_structure.model.materials[0].value.conductivity_x, "5");
    EXPECT_EQ(io_structure.model.materials[0].value.conductivity_y, "5");
    EXPECT_EQ(io_structure.model.materials[0].value.conductivity_z, "5");
    std::filesystem::remove(path);
}

TEST(IoTest, ReadXmlThermalConductivityThreeExpressionsWithTrim)
{
    auto path
        = write_tmp_xml("io_thermal_conductivity_3trim.xml", make_xml_with_thermal_conductivity("  1.5e2 , 2.5 , 0 "));
    auto io_structure = mhs::app::pta::read_xml(path.string());
    ASSERT_EQ(io_structure.model.materials.size(), 1u);
    EXPECT_EQ(io_structure.model.materials[0].name, "mat");
    EXPECT_EQ(io_structure.model.materials[0].value.conductivity_x, "1.5e2");
    EXPECT_EQ(io_structure.model.materials[0].value.conductivity_y, "2.5");
    EXPECT_EQ(io_structure.model.materials[0].value.conductivity_z, "0");
    std::filesystem::remove(path);
}

TEST(IoTest, RejectsXmlWithoutCartesianSamplingShape)
{
    const test_pta::Model model;
    EXPECT_THROW(mhs::app::pta::write_xml("unused.xml", "unused-output.xml", model.output.cartesian_shape, {}, {}),
        std::invalid_argument);
}
