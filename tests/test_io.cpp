#include "io/io.hpp"
#include "model/model_definition.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

using namespace mhs::io;

// Build a small in-memory XML string with/without ObservePoints3D block.
static std::string make_xml_with_observe_points(const std::string& obs_block)
{
    std::string body = R"(<?xml version="1.0" encoding="utf-8"?>
<Structure>
    <mhs::core::StudyType>Transient</mhs::core::StudyType>
    <mhs::core::Dimension>Dimension3D</mhs::core::Dimension>
    <mhs::core::LengthUnit>Mm</mhs::core::LengthUnit>
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
    auto dir = std::filesystem::temp_directory_path() / "mhs_test_io";
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

    mhs::model::ModelDefinition io_structure = mhs::io::read_xml(path.string());
    ASSERT_EQ(io_structure.observation_points.size(), 2u);
    EXPECT_EQ(io_structure.observation_points[0].name, "p1");
    // 坐标保留为 muparser 表达式字符串，由 build_model() 统一求值。
    EXPECT_EQ(io_structure.observation_points[0].x, "1.5");
    EXPECT_EQ(io_structure.observation_points[0].y, "2.5");
    EXPECT_EQ(io_structure.observation_points[0].z, "3.5");
    EXPECT_EQ(io_structure.observation_points[1].name, "p2");
    EXPECT_EQ(io_structure.observation_points[1].x, "0.1");
    std::filesystem::remove(path);
}

TEST(IoTest, ReadXmlWithoutObservationPointsReturnsEmpty)
{
    auto path = write_tmp_xml("io_obs_absent.xml", make_xml_with_observe_points(""));
    mhs::model::ModelDefinition io_structure = mhs::io::read_xml(path.string());
    EXPECT_TRUE(io_structure.observation_points.empty()) << "No <ObservePoints3D> → empty vector";
    std::filesystem::remove(path);
}

TEST(IoTest, WriteXmlEmitsResult0DTransient)
{
    std::string input_xml = R"(<?xml version="1.0" encoding="utf-8"?>
<Structure>
    <mhs::core::StudyType>Transient</mhs::core::StudyType>
    <mhs::core::Dimension>Dimension3D</mhs::core::Dimension>
    <mhs::core::LengthUnit>Mm</mhs::core::LengthUnit>
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

    mhs::core::Model model;
    model.mesh.nx = 1;
    model.mesh.ny = 1;
    model.mesh.nz = 1;
    std::vector<double> node_temperature = {300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0};

    std::vector<mhs::core::ProbeTrace> traces;
    mhs::core::ProbeTrace t1;
    t1.name = "probe_a";
    t1.times = {0.0, 1.0, 2.0};
    t1.values = {300.0, 310.0, 320.0};
    traces.push_back(t1);

    mhs::io::write_xml(in_path.string(), out_path.string(), model, node_temperature, traces);

    std::string out_xml = read_file(out_path);
    EXPECT_NE(out_xml.find("Result0DTransient"), std::string::npos);
    EXPECT_NE(out_xml.find("<PointName>probe_a</PointName>"), std::string::npos);
    EXPECT_NE(out_xml.find("<a:double>300.000000</a:double>"), std::string::npos);
    EXPECT_NE(out_xml.find("<a:double>320.000000</a:double>"), std::string::npos);

    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);
}

TEST(IoTest, WriteXmlEmptyTracesLeavesNoProbeBlocks)
{
    std::string input_xml = R"(<?xml version="1.0" encoding="utf-8"?>
<Structure>
    <mhs::core::StudyType>Steady</mhs::core::StudyType>
    <mhs::core::Dimension>Dimension3D</mhs::core::Dimension>
    <mhs::core::LengthUnit>Mm</mhs::core::LengthUnit>
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

    mhs::core::Model model;
    model.mesh.nx = 1;
    model.mesh.ny = 1;
    model.mesh.nz = 1;
    std::vector<double> node_temperature = {300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0};

    std::vector<mhs::core::ProbeTrace> traces;
    mhs::io::write_xml(in_path.string(), out_path.string(), model, node_temperature, traces);

    std::string out_xml = read_file(out_path);
    EXPECT_EQ(out_xml.find("Result0DTransient"), std::string::npos)
        << "Empty traces should not emit Result0DTransient blocks";

    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);
}

// Build a minimal in-memory XML that contains one material with the given
// DaoreXishu text. Used to exercise mhs::io::read_xml's DaoreXishu parser.
static std::string make_xml_with_daore_xishu(const std::string& daore_text)
{
    std::string body = R"(<?xml version="1.0" encoding="utf-8"?>
<Structure>
    <mhs::core::StudyType>Steady</mhs::core::StudyType>
    <mhs::core::Dimension>Dimension3D</mhs::core::Dimension>
    <mhs::core::LengthUnit>Mm</mhs::core::LengthUnit>
    <AmbientTemperature>300</AmbientTemperature>
    <InitialTemperature>300</InitialTemperature>
    <OtherThermalBondary i:type="SecondTypeThermalBoundary"><a:HeatFlux>0</a:HeatFlux></OtherThermalBondary>
    <Materials xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
        <a:KeyValueOfstringMaterialGyu7GfTz>
            <a:Key>mat</a:Key>
            <a:Value>
                <BiRerong>385</BiRerong>
                <DaoreXishu>)";
    body += daore_text;
    body += R"(</DaoreXishu>
                <Midu>8920</Midu>
            </a:Value>
        </a:KeyValueOfstringMaterialGyu7GfTz>
    </Materials>
</Structure>
)";
    return body;
}

TEST(IoTest, ReadXmlDaoreXishuSingleExpressionTrimsWhitespace)
{
    auto path = write_tmp_xml("io_daore_trim.xml", make_xml_with_daore_xishu("  5  "));
    mhs::model::ModelDefinition io_structure = mhs::io::read_xml(path.string());
    ASSERT_EQ(io_structure.materials.size(), 1u);
    EXPECT_EQ(io_structure.materials[0].name, "mat");
    EXPECT_EQ(io_structure.materials[0].value.conductivity_x, "5");
    EXPECT_EQ(io_structure.materials[0].value.conductivity_y, "5");
    EXPECT_EQ(io_structure.materials[0].value.conductivity_z, "5");
    std::filesystem::remove(path);
}

TEST(IoTest, ReadXmlDaoreXishuThreeExpressionsWithTrim)
{
    auto path = write_tmp_xml("io_daore_3trim.xml", make_xml_with_daore_xishu("  1.5e2 , 2.5 , 0 "));
    mhs::model::ModelDefinition io_structure = mhs::io::read_xml(path.string());
    ASSERT_EQ(io_structure.materials.size(), 1u);
    EXPECT_EQ(io_structure.materials[0].name, "mat");
    EXPECT_EQ(io_structure.materials[0].value.conductivity_x, "1.5e2");
    EXPECT_EQ(io_structure.materials[0].value.conductivity_y, "2.5");
    EXPECT_EQ(io_structure.materials[0].value.conductivity_z, "0");
    std::filesystem::remove(path);
}

// Helper: write overlay XML to a temp file and return its path.
static std::filesystem::path write_tmp_overlay(const std::string& content)
{
    auto path = std::filesystem::temp_directory_path() / "fluid_overlay_test.xml";
    std::ofstream ofs(path);
    ofs << content;
    return path;
}

TEST(IoTest, MergeFluidXmlAddsMaterialsAndBoundariesToDefinition)
{
    std::string xml = R"(<?xml version="1.0" encoding="UTF-8"?>
<FluidOverlay xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
    <FluidMaterial name="water">
        <DynamicViscosity>0.00089</DynamicViscosity>
    </FluidMaterial>
    <Boundary>
        <BoundaryCategory>Fluidic</BoundaryCategory>
        <Name>inlet</Name>
        <FaceKeys>
            <string>X|E|0|0.5|1.5|0.3|0.5</string>
        </FaceKeys>
        <Pressure>500</Pressure>
    </Boundary>
    <Boundary>
        <BoundaryCategory>Fluidic</BoundaryCategory>
        <Name>outlet</Name>
        <FaceKeys>
            <string>X|E|8|0.5|1.5|0.3|0.5</string>
        </FaceKeys>
        <Pressure>0</Pressure>
    </Boundary>
</FluidOverlay>)";

    mhs::model::ModelDefinition definition;
    definition.materials.push_back({"water", mhs::model::MaterialSpec {}});

    auto path = write_tmp_overlay(xml);
    const bool merged = mhs::io::merge_fluid_xml(path.string(), definition);
    std::filesystem::remove(path);

    ASSERT_TRUE(merged);
    ASSERT_EQ(definition.materials.size(), 1u);
    EXPECT_EQ(definition.materials[0].value.dynamic_viscosity, "0.00089");
    ASSERT_EQ(definition.fluid_boundaries.size(), 2u);
    EXPECT_EQ(definition.fluid_boundaries[0].kind, mhs::model::FluidBoundaryKind::Pressure);
    EXPECT_DOUBLE_EQ(definition.fluid_boundaries[0].value, 500.0);
    ASSERT_EQ(definition.fluid_boundaries[0].regions.size(), 1u);
    EXPECT_EQ(definition.fluid_boundaries[0].regions[0].axis, mhs::model::Axis::X);
    EXPECT_DOUBLE_EQ(definition.fluid_boundaries[0].regions[0].coordinate, 0.0);
    EXPECT_EQ(definition.fluid_boundaries[1].kind, mhs::model::FluidBoundaryKind::Pressure);
    EXPECT_DOUBLE_EQ(definition.fluid_boundaries[1].value, 0.0);
    ASSERT_EQ(definition.fluid_boundaries[1].regions.size(), 1u);
    EXPECT_EQ(definition.fluid_boundaries[1].regions[0].axis, mhs::model::Axis::X);
    EXPECT_DOUBLE_EQ(definition.fluid_boundaries[1].regions[0].coordinate, 8.0);
}

TEST(IoTest, MergeFluidXmlMissingElementReturnsFalse)
{
    std::string xml = "<?xml version=\"1.0\"?><Root/>";
    mhs::model::ModelDefinition definition;
    auto path = write_tmp_overlay(xml);
    const bool merged = mhs::io::merge_fluid_xml(path.string(), definition);
    std::filesystem::remove(path);

    EXPECT_FALSE(merged);
    EXPECT_TRUE(definition.materials.empty());
    EXPECT_TRUE(definition.fluid_boundaries.empty());
}

TEST(IoTest, MergeFluidXmlNonexistentFileReturnsFalse)
{
    mhs::model::ModelDefinition definition;
    EXPECT_FALSE(mhs::io::merge_fluid_xml("nonexistent_overlay.xml", definition));
}
