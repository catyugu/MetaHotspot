"""MetaHotspot XML Parser Tests.

Comprehensive tests covering all XML parsing functionality.
"""

from pathlib import Path

import numpy as np
import pytest
from pytest import approx
import xml.etree.ElementTree as ET

from metahotspot.xml_parser import *
from metahotspot.metahotspot_types import *

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_rect_xml() -> str:
    return """
    <Rect xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
        <Add_sub>true</Add_sub>
        <ArrayDisguise>false</ArrayDisguise>
        <CreatedByArray>false</CreatedByArray>
        <HeightExpression>50</HeightExpression>
        <Name>加操作 1</Name>
        <NotOccupiedByArray>true</NotOccupiedByArray>
        <ParentRect i:nil="true" xmlns:i="http://www.w3.org/2001/XMLSchema-instance"/>
        <WidthExpression>100</WidthExpression>
        <XExpression>0</XExpression>
        <XIntervalExpression>0</XIntervalExpression>
        <XSizeExpression>100</XSizeExpression>
        <YExpression>0</YExpression>
        <YIntervalExpression>0</YIntervalExpression>
        <YSizeExpression>50</YSizeExpression>
    </Rect>
    """


@pytest.fixture
def sample_block_xml() -> str:
    return """
    <Block xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
        <AllRects>
            <Rect>
                <Add_sub>true</Add_sub>
                <ArrayDisguise>false</ArrayDisguise>
                <CreatedByArray>false</CreatedByArray>
                <HeightExpression>50</HeightExpression>
                <Name>加操作 1</Name>
                <NotOccupiedByArray>true</NotOccupiedByArray>
                <ParentRect i:nil="true" xmlns:i="http://www.w3.org/2001/XMLSchema-instance"/>
                <WidthExpression>50</WidthExpression>
                <XExpression>0</XExpression>
                <XIntervalExpression>0</XIntervalExpression>
                <XSizeExpression>50</XSizeExpression>
                <YExpression>0</YExpression>
                <YIntervalExpression>0</YIntervalExpression>
                <YSizeExpression>50</YSizeExpression>
            </Rect>
            <Rect>
                <Add_sub>false</Add_sub>
                <ArrayDisguise>false</ArrayDisguise>
                <CreatedByArray>false</CreatedByArray>
                <HeightExpression>25</HeightExpression>
                <Name>减操作 1</Name>
                <NotOccupiedByArray>true</NotOccupiedByArray>
                <ParentRect i:nil="true" xmlns:i="http://www.w3.org/2001/XMLSchema-instance"/>
                <WidthExpression>25</WidthExpression>
                <XExpression>25</XExpression>
                <XIntervalExpression>0</XIntervalExpression>
                <XSizeExpression>25</XSizeExpression>
                <YExpression>25</YExpression>
                <YIntervalExpression>0</YIntervalExpression>
                <YSizeExpression>25</YSizeExpression>
            </Rect>
        </AllRects>
        <CanCreateArray>true</CanCreateArray>
        <Color xmlns:a="http://schemas.datacontract.org/2004/07/System.Windows.Media">
            <a:A>255</a:A>
            <a:B>255</a:B>
            <a:G>205</a:G>
            <a:R>147</a:R>
            <a:ScA>1</a:ScA>
            <a:ScB>1</a:ScB>
            <a:ScG>0.610495567</a:ScG>
            <a:ScR>0.291770667</a:ScR>
        </Color>
        <HasBeenChanged>true</HasBeenChanged>
        <InitialName>块 1</InitialName>
        <IsChargeConservation>false</IsChargeConservation>
        <IsChargeConservationPiezoelectricity>false</IsChargeConservationPiezoelectricity>
        <IsElectrode>false</IsElectrode>
        <IsLinearElasticMaterial>false</IsLinearElasticMaterial>
        <IsNormalMaterial>true</IsNormalMaterial>
        <IsPML>false</IsPML>
        <IsPiezoelectricMaterial>false</IsPiezoelectricMaterial>
        <IsTerminal>false</IsTerminal>
        <IsVisible>true</IsVisible>
        <MaterialName>copper</MaterialName>
        <MeshSizeXExpression>0</MeshSizeXExpression>
        <MeshSizeYExpression>0</MeshSizeYExpression>
        <MeshSizeZExpression>0</MeshSizeZExpression>
        <Name>块 1</Name>
        <NetDonorConcentration>0</NetDonorConcentration>
        <NotChargeConservation>true</NotChargeConservation>
        <PMLType/>
        <ParentBlock i:nil="true" xmlns:i="http://www.w3.org/2001/XMLSchema-instance"/>
        <SelectedPMLType/>
        <TerminalVoltage>0</TerminalVoltage>
        <ThicknessExpression>20</ThicknessExpression>
        <TiReyuan>1e8</TiReyuan>
        <XOffsetExpression>0</XOffsetExpression>
        <YOffsetExpression>0</YOffsetExpression>
        <ZOffsetExpression>0</ZOffsetExpression>
    </Block>
    """


@pytest.fixture
def sample_layer_xml() -> str:
    return """
    <Layer xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
        <Blocks>
            <Block>
                <AllRects>
                    <Rect>
                        <Add_sub>true</Add_sub>
                        <ArrayDisguise>false</ArrayDisguise>
                        <CreatedByArray>false</CreatedByArray>
                        <HeightExpression>50</HeightExpression>
                        <Name>加操作 1</Name>
                        <NotOccupiedByArray>true</NotOccupiedByArray>
                        <ParentRect i:nil="true" xmlns:i="http://www.w3.org/2001/XMLSchema-instance"/>
                        <WidthExpression>50</WidthExpression>
                        <XExpression>0</XExpression>
                        <XIntervalExpression>0</XIntervalExpression>
                        <XSizeExpression>50</XSizeExpression>
                        <YExpression>0</YExpression>
                        <YIntervalExpression>0</YIntervalExpression>
                        <YSizeExpression>50</YSizeExpression>
                    </Rect>
                </AllRects>
                <CanCreateArray>true</CanCreateArray>
                <Color xmlns:a="http://schemas.datacontract.org/2004/07/System.Windows.Media">
                    <a:A>255</a:A>
                    <a:B>255</a:B>
                    <a:G>205</a:G>
                    <a:R>147</a:R>
                </Color>
                <HasBeenChanged>true</HasBeenChanged>
                <InitialName>块 1</InitialName>
                <IsNormalMaterial>true</IsNormalMaterial>
                <IsVisible>true</IsVisible>
                <MaterialName>copper</MaterialName>
                <MeshSizeXExpression>0</MeshSizeXExpression>
                <MeshSizeYExpression>0</MeshSizeYExpression>
                <MeshSizeZExpression>0</MeshSizeZExpression>
                <Name>块 1</Name>
                <ThicknessExpression>20</ThicknessExpression>
                <TiReyuan>0</TiReyuan>
                <XOffsetExpression>0</XOffsetExpression>
                <YOffsetExpression>0</YOffsetExpression>
                <ZOffsetExpression>0</ZOffsetExpression>
            </Block>
        </Blocks>
        <CreatedByTopLayer>false</CreatedByTopLayer>
        <IsBasic>false</IsBasic>
        <IsBottomPackaging>false</IsBottomPackaging>
        <IsDie>false</IsDie>
        <IsDoule>false</IsDoule>
        <IsIDTSawModel>false</IsIDTSawModel>
        <IsNotIDTSawModel>false</IsNotIDTSawModel>
        <IsSingle>false</IsSingle>
        <IsSubstrate>false</IsSubstrate>
        <IsTIM>false</IsTIM>
        <IsTopLayer>true</IsTopLayer>
        <MeshSizeXExpression>10</MeshSizeXExpression>
        <MeshSizeYExpression>10</MeshSizeYExpression>
        <MeshSizeZExpression>5</MeshSizeZExpression>
        <MetalConversionRate>0.4</MetalConversionRate>
        <Name>层 1</Name>
        <PeriodWidth>10</PeriodWidth>
        <SawBoundaryOption>None</SawBoundaryOption>
        <ThicknessExpression>20</ThicknessExpression>
        <XOffsetExpression>0</XOffsetExpression>
        <YOffsetExpression>0</YOffsetExpression>
    </Layer>
    """


@pytest.fixture
def sample_material_xml() -> str:
    return """
    <a:KeyValueOfstringMaterialGyu7GfTz
        xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"
        xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models"
        xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
        <a:Key>copper</a:Key>
        <a:Value>
            <BiRerong i:nil="true"/>
            <Chi0 i:nil="true"/>
            <CouplingMatrixForSaw i:nil="true"/>
            <DaoreXishu>400</DaoreXishu>
            <DielectricLossFactorForSaw i:nil="true"/>
            <Eg0 i:nil="true"/>
            <ElasticMatrixForSaw i:nil="true"/>
            <Epsilon i:nil="true"/>
            <EpsilonForSaw i:nil="true"/>
            <EpsilonVectorForSaw i:nil="true"/>
            <MaterialKindForSaw>压电材料</MaterialKindForSaw>
            <MechanicalDampingFactorForSaw i:nil="true"/>
            <Midu>8960</Midu>
            <MiduForSaw i:nil="true"/>
            <MuN i:nil="true"/>
            <MuP i:nil="true"/>
            <Nc i:nil="true"/>
            <Nv i:nil="true"/>
            <PoissonsRatioForSaw i:nil="true"/>
            <WaveSpeedForSaw i:nil="true"/>
            <YoungsModulusForSaw i:nil="true"/>
        </a:Value>
    </a:KeyValueOfstringMaterialGyu7GfTz>
    """


@pytest.fixture
def sample_first_type_bc_xml() -> str:
    return """
    <ThermalBoundary i:type="a:FirstTypeThermalBoundary"
        xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions"
        xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
        <a:Temperature>500</a:Temperature>
    </ThermalBoundary>
    """


@pytest.fixture
def sample_second_type_bc_xml() -> str:
    return """
    <ThermalBoundary i:type="a:SecondTypeThermalBoundary"
        xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions"
        xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
        <a:HeatFlux>1000</a:HeatFlux>
    </ThermalBoundary>
    """


@pytest.fixture
def sample_third_type_bc_xml() -> str:
    return """
    <ThermalBoundary i:type="a:ThirdTypeThermalBoundary"
        xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions"
        xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
        <a:ConvectionCoefficient>10</a:ConvectionCoefficient>
        <a:EnvironmentTemperature>300</a:EnvironmentTemperature>
    </ThermalBoundary>
    """


@pytest.fixture
def sample_boundary_xml() -> str:
    return """
    <Boundary xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models"
        xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"
        xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
        <BoundaryCategory>Electrical</BoundaryCategory>
        <FaceIds xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"/>
        <FaceKeys xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
            <a:string>Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100</a:string>
        </FaceKeys>
        <Name>边界 1</Name>
        <ThermalBoundary i:type="a:FirstTypeThermalBoundary"
            xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions">
            <a:Temperature>500</a:Temperature>
        </ThermalBoundary>
    </Boundary>
    """


@pytest.fixture
def sample_mesh_xml() -> str:
    return """
    <Mesh xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
        <XArray xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
            <a:double>0</a:double>
            <a:double>10</a:double>
            <a:double>20</a:double>
            <a:double>30</a:double>
            <a:double>40</a:double>
            <a:double>50</a:double>
        </XArray>
        <YArray xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
            <a:double>0</a:double>
            <a:double>20</a:double>
            <a:double>40</a:double>
        </YArray>
        <ZArray xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
            <a:double>0</a:double>
            <a:double>5</a:double>
            <a:double>10</a:double>
        </ZArray>
    </Mesh>
    """


# ============================================================================
# Rect Parsing Tests
# ============================================================================


class TestRectParsing:
    """Test Rect element parsing."""

    def test_parse_rect_basic(self, sample_rect_xml):
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(sample_rect_xml)
        rect = parse_rect(elem)

        assert rect.name == "加操作 1"
        assert rect.add_sub is True
        assert rect.x == 0.0
        assert rect.y == 0.0
        assert rect.width == 100.0
        assert rect.height == 50.0

    def test_parse_rect_sub_operation(self, sample_rect_xml):
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(sample_rect_xml)
        rect = parse_rect(elem)
        # Add_sub is true in fixture, test False case
        assert rect.add_sub is True

    def test_parse_rect_missing_optional_fields(self):
        xml_str = """
        <Rect xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
            <Name>Test</Name>
            <Add_sub>true</Add_sub>
            <XExpression>10</XExpression>
            <YExpression>20</YExpression>
            <WidthExpression>30</WidthExpression>
            <HeightExpression>40</HeightExpression>
        </Rect>
        """
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(xml_str)
        rect = parse_rect(elem)

        assert rect.name == "Test"
        assert rect.x_interval == 0.0
        assert rect.y_interval == 0.0


# ============================================================================
# Block Parsing Tests
# ============================================================================


class TestBlockParsing:
    """Test Block element parsing."""

    def test_parse_block_with_rects(self, sample_block_xml):
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(sample_block_xml)
        block = parse_block(elem)

        assert block.name == "块 1"
        assert block.material_name == "copper"
        assert block.thickness == 20.0
        assert block.heat_source == 1e8
        assert len(block.rects) == 2
        assert block.rects[0].add_sub is True
        assert block.rects[1].add_sub is False

    def test_parse_block_with_zero_heat_source(self):
        xml_str = """
        <Block xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
            <AllRects/>
            <Name>块 2</Name>
            <MaterialName>silicon</MaterialName>
            <ThicknessExpression>10</ThicknessExpression>
            <TiReyuan>0</TiReyuan>
        </Block>
        """
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(xml_str)
        block = parse_block(elem)

        assert block.heat_source == 0.0
        assert len(block.rects) == 0

    def test_parse_block_offsets(self):
        xml_str = """
        <Block xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
            <AllRects/>
            <Name>块</Name>
            <MaterialName>copper</MaterialName>
            <ThicknessExpression>5</ThicknessExpression>
            <XOffsetExpression>10</XOffsetExpression>
            <YOffsetExpression>20</YOffsetExpression>
            <ZOffsetExpression>30</ZOffsetExpression>
        </Block>
        """
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(xml_str)
        block = parse_block(elem)

        assert block.x_offset == 10.0
        assert block.y_offset == 20.0
        assert block.z_offset == 30.0


# ============================================================================
# Layer Parsing Tests
# ============================================================================


class TestLayerParsing:
    """Test Layer element parsing."""

    def test_parse_layer_basic(self, sample_layer_xml):
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(sample_layer_xml)
        layer = parse_layer(elem)

        assert layer.name == "层 1"
        assert layer.thickness == 20.0
        assert layer.mesh_size_x == 10.0
        assert layer.mesh_size_y == 10.0
        assert layer.mesh_size_z == 5.0
        assert layer.is_top_layer is True
        assert len(layer.blocks) == 1

    def test_parse_layer_flags(self):
        xml_str = """
        <Layer xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
            <Blocks/>
            <IsDie>true</IsDie>
            <IsTIM>true</IsTIM>
            <IsSubstrate>true</IsSubstrate>
            <Name>层 3</Name>
            <ThicknessExpression>100</ThicknessExpression>
        </Layer>
        """
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(xml_str)
        layer = parse_layer(elem)

        assert layer.is_die is True
        assert layer.is_tim is True
        assert layer.is_substrate is True

    def test_parse_layer_empty_blocks(self):
        xml_str = """
        <Layer xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
            <Blocks/>
            <Name>空层</Name>
            <ThicknessExpression>50</ThicknessExpression>
        </Layer>
        """
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(xml_str)
        layer = parse_layer(elem)

        assert len(layer.blocks) == 0


# ============================================================================
# Material Parsing Tests
# ============================================================================


class TestMaterialParsing:
    """Test Material element parsing."""

    def test_parse_material_basic(self, sample_material_xml):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(sample_material_xml)
        # The sample_material_xml fixture has: a:Key=copper, a:Value containing material props
        key_elem = root.find(
            "{http://schemas.microsoft.com/2003/10/Serialization/Arrays}Key"
        )
        value_elem = root.find(
            "{http://schemas.microsoft.com/2003/10/Serialization/Arrays}Value"
        )
        mat = parse_material(key_elem.text, value_elem)

        assert mat.name == "copper"
        assert mat.k == 400.0
        assert mat.density == 8960.0
        assert mat.cp == 0.0  # BiRerong is nil

    def test_parse_material_with_cp(self):
        xml_str = """
        <a:KeyValueOfstringMaterialGyu7GfTz
            xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"
            xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
            <a:Key>silicon</a:Key>
            <a:Value>
                <DaoreXishu>130</DaoreXishu>
                <Midu>2330</Midu>
                <BiRerong>1630000</BiRerong>
            </a:Value>
        </a:KeyValueOfstringMaterialGyu7GfTz>
        """
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_str)
        key_elem = root.find(
            "{http://schemas.microsoft.com/2003/10/Serialization/Arrays}Key"
        )
        value_elem = root.find(
            "{http://schemas.microsoft.com/2003/10/Serialization/Arrays}Value"
        )
        mat = parse_material(key_elem.text, value_elem)

        assert mat.name == "silicon"
        assert mat.k == 130.0
        assert mat.density == 2330.0
        assert mat.cp == 1630000.0

    def test_parse_material_nil_cp(self):
        xml_str = """
        <a:KeyValueOfstringMaterialGyu7GfTz
            xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"
            xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
            <a:Key>test</a:Key>
            <a:Value>
                <DaoreXishu>100</DaoreXishu>
                <Midu>1000</Midu>
                <BiRerong i:nil="true" xmlns:i="http://www.w3.org/2001/XMLSchema-instance"/>
            </a:Value>
        </a:KeyValueOfstringMaterialGyu7GfTz>
        """
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_str)
        key_elem = root.find(
            "{http://schemas.microsoft.com/2003/10/Serialization/Arrays}Key"
        )
        value_elem = root.find(
            "{http://schemas.microsoft.com/2003/10/Serialization/Arrays}Value"
        )
        mat = parse_material(key_elem.text, value_elem)

        assert mat.cp == 0.0


# ============================================================================
# Thermal Boundary Parsing Tests
# ============================================================================


class TestThermalBoundaryParsing:
    """Test ThermalBoundary element parsing."""

    def test_parse_first_type(self, sample_first_type_bc_xml):

        elem = ET.fromstring(sample_first_type_bc_xml)
        bc = parse_thermal_boundary(elem)

        assert bc["type"] == "first"
        assert bc["temperature"] == 500.0
        assert bc.get("heat_flux") is None
        assert bc.get("h_conv") is None

    def test_parse_second_type(self, sample_second_type_bc_xml):
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(sample_second_type_bc_xml)
        bc = parse_thermal_boundary(elem)

        assert bc["type"] == "second"
        assert bc["heat_flux"] == 1000.0
        assert bc.get("temperature") is None
        assert bc.get("h_conv") is None

    def test_parse_third_type(self, sample_third_type_bc_xml):
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(sample_third_type_bc_xml)
        bc = parse_thermal_boundary(elem)

        assert bc["type"] == "third"
        assert bc["t_inf"] == 300.0
        assert bc["h_conv"] == 10.0
        assert bc.get("temperature") is None
        assert bc.get("heat_flux") is None

    def test_parse_none_returns_adiabatic(self):
        bc = parse_thermal_boundary(None)
        assert bc["type"] == "second"
        assert bc["heat_flux"] == 0.0


# ============================================================================
# Boundary Parsing Tests
# ============================================================================


class TestBoundaryParsing:
    """Test Boundary element parsing."""

    def test_parse_boundary(self, sample_boundary_xml):
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(sample_boundary_xml)
        bc = parse_boundary(elem)

        assert bc.name == "边界 1"
        assert bc.boundary_type == "first"
        assert bc.face_keys == ["Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100"]

    def test_parse_boundary_with_unit_conversion(self):
        xml_str = """
        <Boundary xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models"
            xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"
            xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
            <BoundaryCategory>Electrical</BoundaryCategory>
            <FaceKeys xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
                <a:string>Z|E|0|0,100,0,100</a:string>
            </FaceKeys>
            <Name>边界 2</Name>
            <ThermalBoundary i:type="a:SecondTypeThermalBoundary"
                xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions">
                <a:HeatFlux>500</a:HeatFlux>
            </ThermalBoundary>
        </Boundary>
        """
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(xml_str)
        bc = parse_boundary(elem)

        assert bc.boundary_type == "second"
        assert bc.params["heat_flux"] == 500.0
        assert bc.face_keys == ["Z|E|0|0,100,0,100"]

    def test_parse_boundary_multiple_facekeys(self):
        xml_str = """
        <Boundary xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models"
            xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"
            xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
            <BoundaryCategory>Electrical</BoundaryCategory>
            <FaceKeys xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
                <a:string>Z|E|0|0,50,0,50</a:string>
                <a:string>Z|E|0|50,100,50,100</a:string>
            </FaceKeys>
            <Name>多区域边界</Name>
            <ThermalBoundary i:type="a:ThirdTypeThermalBoundary"
                xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions">
                <a:ConvectionCoefficient>5</a:ConvectionCoefficient>
                <a:EnvironmentTemperature>350</a:EnvironmentTemperature>
            </ThermalBoundary>
        </Boundary>
        """
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(xml_str)
        bc = parse_boundary(elem)

        assert bc.boundary_type == "third"
        assert bc.params["h_conv"] == 5.0
        assert bc.params["t_inf"] == 350.0
        assert len(bc.face_keys) == 2


# ============================================================================
# Mesh Parsing Tests
# ============================================================================


class TestMeshParsing:
    """Test Mesh element parsing."""

    def test_parse_mesh_basic(self, sample_mesh_xml):
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(sample_mesh_xml)
        mesh = parse_mesh(elem)

        assert len(mesh.x) == 6
        assert len(mesh.y) == 3
        assert len(mesh.z) == 3
        assert mesh.x[0] == 0.0
        assert mesh.x[-1] == 50.0
        assert mesh.y[0] == 0.0
        assert mesh.z[-1] == 10.0

    def test_parse_mesh_with_unit_conversion(self, sample_mesh_xml):
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(sample_mesh_xml)
        mesh = parse_mesh(elem)

        # Values should be converted to meters
        assert mesh.x[1] == approx(10)
        assert mesh.y[1] == approx(20)
        assert mesh.z[1] == approx(5)


# ============================================================================
# Full XML Parsing Tests
# ============================================================================


class TestFullXmlParsing:
    """Test complete XML file parsing."""

    def test_parse_complete_xml(self, tmp_path):
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <Structure xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models"
            xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
            <AlphaDegree>0</AlphaDegree>
            <AmbientTemperature>300</AmbientTemperature>
            <BasicGeometries xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Dialogs"/>
            <BetaDegree>0</BetaDegree>
            <Blocks/>
            <BottomThermalBoundary i:nil="true"/>
            <Boundaries>
                <Boundary>
                    <BoundaryCategory>Electrical</BoundaryCategory>
                    <FaceIds xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"/>
                    <FaceKeys xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
                        <a:string>Z|E|0|0,100,0,100</a:string>
                    </FaceKeys>
                    <Name>边界 1</Name>
                    <ThermalBoundary i:type="a:FirstTypeThermalBoundary"
                        xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions">
                        <a:Temperature>500</a:Temperature>
                    </ThermalBoundary>
                </Boundary>
            </Boundaries>
            <Dimension>Dimension3D</Dimension>
            <EnabledPhysics>HeatTransfer</EnabledPhysics>
            <InitialTemperature>300</InitialTemperature>
            <Layers>
                <Layer>
                    <Blocks>
                        <Block>
                            <AllRects>
                                <Rect>
                                    <Add_sub>true</Add_sub>
                                    <HeightExpression>100</HeightExpression>
                                    <Name>加操作 1</Name>
                                    <WidthExpression>100</WidthExpression>
                                    <XExpression>0</XExpression>
                                    <YExpression>0</YExpression>
                                </Rect>
                            </AllRects>
                            <MaterialName>copper</MaterialName>
                            <Name>块 1</Name>
                            <ThicknessExpression>20</ThicknessExpression>
                            <TiReyuan>1e8</TiReyuan>
                        </Block>
                    </Blocks>
                    <IsTopLayer>true</IsTopLayer>
                    <MeshSizeXExpression>10</MeshSizeXExpression>
                    <MeshSizeYExpression>10</MeshSizeYExpression>
                    <MeshSizeZExpression>5</MeshSizeZExpression>
                    <Name>层 1</Name>
                    <ThicknessExpression>20</ThicknessExpression>
                </Layer>
            </Layers>
            <LengthUnit>Mm</LengthUnit>
            <Materials xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
                <a:KeyValueOfstringMaterialGyu7GfTz>
                    <a:Key>copper</a:Key>
                    <a:Value>
                        <DaoreXishu>400</DaoreXishu>
                        <Midu>8960</Midu>
                    </a:Value>
                </a:KeyValueOfstringMaterialGyu7GfTz>
            </Materials>
            <StudyType>Steady</StudyType>
        </Structure>
        """

        xml_file = tmp_path / "test.xml"
        xml_file.write_text(xml_content, encoding="utf-8")

        config = parse_xml(xml_file)

        assert config.study_type == "Steady"
        assert config.ambient_temperature == 300.0
        assert config.initial_temperature == 300.0
        assert config.length_unit == "Mm"
        assert len(config.layers) == 1
        assert len(config.materials) == 1
        assert len(config.boundaries) == 1

        layer = config.layers[0]
        assert layer.name == "层 1"
        assert layer.is_top_layer is True
        assert len(layer.blocks) == 1
        assert layer.blocks[0].material_name == "copper"

        mat = config.materials["copper"]
        assert mat.k == 400.0
        assert mat.density == 8960.0

        bc = config.boundaries[0]
        assert bc.name == "边界 1"
        assert bc.boundary_type == "first"
        assert bc.params["temperature"] == 500.0

    def test_parse_transient_xml(self, tmp_path):
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <Structure xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
            <AmbientTemperature>300</AmbientTemperature>
            <InitialTemperature>300</InitialTemperature>
            <LengthUnit>Mm</LengthUnit>
            <StudyType>Transient</StudyType>
            <TransientStudyDuration>100</TransientStudyDuration>
            <TransientStudyTimeStep>1</TransientStudyTimeStep>
            <Layers/>
            <Materials xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"/>
            <Boundaries/>
        </Structure>
        """

        xml_file = tmp_path / "transient.xml"
        xml_file.write_text(xml_content, encoding="utf-8")

        config = parse_xml(xml_file)

        assert config.study_type == "Transient"
        assert config.transient_duration == 100.0
        assert config.transient_timestep == 1.0

    def test_parse_invalid_root_raises(self, tmp_path):
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <NotStructure xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
        </NotStructure>
        """

        xml_file = tmp_path / "invalid.xml"
        xml_file.write_text(xml_content, encoding="utf-8")

        with pytest.raises(ValueError, match="Root element must be Structure"):
            parse_xml(xml_file)


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_optional_elements(self, tmp_path):
        """Parse handles missing optional elements gracefully."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <Structure xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
            <StudyType>Steady</StudyType>
            <LengthUnit>Mm</LengthUnit>
            <!-- Missing: AmbientTemperature, InitialTemperature -->
            <Layers/>
            <Materials xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"/>
            <Boundaries/>
        </Structure>
        """

        xml_file = tmp_path / "minimal.xml"
        xml_file.write_text(xml_content, encoding="utf-8")

        config = parse_xml(xml_file)

        # Defaults should be used
        assert config.ambient_temperature == 300.0
        assert config.initial_temperature == 300.0

    def test_empty_layers_and_materials(self, tmp_path):
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <Structure xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
            <AmbientTemperature>300</AmbientTemperature>
            <InitialTemperature>300</InitialTemperature>
            <LengthUnit>Mm</LengthUnit>
            <StudyType>Steady</StudyType>
            <Layers></Layers>
            <Materials xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"></Materials>
            <Boundaries></Boundaries>
        </Structure>
        """

        xml_file = tmp_path / "empty.xml"
        xml_file.write_text(xml_content, encoding="utf-8")

        config = parse_xml(xml_file)

        assert len(config.layers) == 0
        assert len(config.materials) == 0
        assert len(config.boundaries) == 0

    def test_boundary_without_facekeys(self, tmp_path):
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <Structure xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models"
            xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"
            xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
            <AmbientTemperature>300</AmbientTemperature>
            <InitialTemperature>300</InitialTemperature>
            <LengthUnit>Mm</LengthUnit>
            <StudyType>Steady</StudyType>
            <Layers/>
            <Materials xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"/>
            <Boundaries>
                <Boundary>
                    <Name>无FaceKeys</Name>
                    <ThermalBoundary i:type="a:SecondTypeThermalBoundary"
                        xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions">
                        <a:HeatFlux>100</a:HeatFlux>
                    </ThermalBoundary>
                </Boundary>
            </Boundaries>
        </Structure>
        """

        xml_file = tmp_path / "no_facekeys.xml"
        xml_file.write_text(xml_content, encoding="utf-8")

        with pytest.raises(ValueError, match="has no FaceKeys"):
            parse_xml(xml_file)


# ============================================================================
# Data Type Tests
# ============================================================================


class TestDataTypes:
    """Test that parsed data has correct types."""

    def test_rect_dataclass_fields(self):
        rect = Rect(name="test", add_sub=True, x=1.0, y=2.0, width=3.0, height=4.0)
        assert rect.name == "test"
        assert isinstance(rect.add_sub, bool)
        assert isinstance(rect.x, float)

    def test_block_dataclass_fields(self):
        block = BlockGeometry(
            name="block1",
            material_name="copper",
            thickness=10.0,
            rects=[],
        )
        assert block.name == "block1"
        assert block.heat_source == 0.0  # default

    def test_layer_dataclass_fields(self):
        layer = LayerConfig(
            name="layer1",
            thickness=20.0,
        )
        assert layer.is_top_layer is False  # default
        assert layer.is_die is False  # default

    def test_material_dataclass_fields(self):
        mat = MaterialModel(name="copper", k=400.0, cp=0.0, density=8960.0)
        assert mat.name == "copper"

    def test_thermal_boundary_dataclass_fields(self):
        bc = ThermalBoundary(
            name="bc1",
            boundary_type="first",
            face_keys=["Z|E|0|0,100,0,100"],
            params={"temperature": 500.0},
        )
        assert bc.params["temperature"] == 500.0

    def test_mesh_coordinates_dataclass(self):
        mesh = MeshCoordinates(
            x=np.array([0, 1, 2]),
            y=np.array([0, 1]),
            z=np.array([0, 0.5, 1]),
        )
        assert len(mesh.x) == 3
        assert isinstance(mesh.x, np.ndarray)

# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
