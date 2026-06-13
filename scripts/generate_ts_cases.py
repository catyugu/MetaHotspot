#!/usr/bin/env python3
"""Generate BDF2 / AdaptiveBdf test cases with a simple, deterministic structure.

Mirrors the format of cases/simple_transient_tests/case1.xml (proper
namespaces, schema-instance attributes, all the required blocks).
"""
import os
import sys

def make_case(duration, time_step, scheme, initial_dt, output_dt, min_dt=None, max_dt=None,
              abs_tol="1e-6", rel_tol="1e-3"):
    ts_block = f"""    <TimeScheme>
        <Scheme>{scheme}</Scheme>
        <InitialDt>{initial_dt}</InitialDt>"""
    if min_dt is not None:
        ts_block += f"\n        <MinDt>{min_dt}</MinDt>"
    if max_dt is not None:
        ts_block += f"\n        <MaxDt>{max_dt}</MaxDt>"
    ts_block += f"""
        <AbsTol>{abs_tol}</AbsTol>
        <RelTol>{rel_tol}</RelTol>
        <OutputDt>{output_dt}</OutputDt>
    </TimeScheme>"""

    return f"""<?xml version="1.0" encoding="utf-8"?>
<Structure xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models"
    xmlns:i="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays"
    xmlns:b="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
    <AmbientTemperature>300</AmbientTemperature>
    <Boundaries>
        <Boundary>
            <BoundaryCategory>Electrical</BoundaryCategory>
            <FaceKeys xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
                <a:string>Y|E|0|0|10|0|10</a:string>
            </FaceKeys>
            <Name>bottom</Name>
            <ThermalBoundary i:type="a:SecondTypeThermalBoundary"
                xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions">
                <a:HeatFlux>-10000</a:HeatFlux>
            </ThermalBoundary>
        </Boundary>
        <Boundary>
            <BoundaryCategory>Electrical</BoundaryCategory>
            <FaceKeys xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
                <a:string>Y|E|100|0|10|0|10</a:string>
            </FaceKeys>
            <Name>top</Name>
            <ThermalBoundary i:type="a:SecondTypeThermalBoundary"
                xmlns:a="http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions">
                <a:HeatFlux>10000</a:HeatFlux>
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
                            <HeightExpression>40</HeightExpression>
                            <Name>加操作 1</Name>
                            <WidthExpression>10</WidthExpression>
                            <XExpression>0</XExpression>
                            <XIntervalExpression>0</XIntervalExpression>
                            <XSizeExpression>10</XSizeExpression>
                            <YExpression>0</YExpression>
                            <YIntervalExpression>0</YIntervalExpression>
                            <YSizeExpression>40</YSizeExpression>
                        </Rect>
                    </AllRects>
                    <IsNormalMaterial>true</IsNormalMaterial>
                    <MaterialName>copper</MaterialName>
                    <Name>块 1</Name>
                    <TiReyuan>0</TiReyuan>
                </Block>
            </Blocks>
            <IsTopLayer>true</IsTopLayer>
            <Name>layer1</Name>
            <ThicknessExpression>10</ThicknessExpression>
        </Layer>
    </Layers>
    <LengthUnit>Mm</LengthUnit>
    <Materials>
        <a:KeyValueOfstringMaterialGyu7GfTz>
            <a:Key>copper</a:Key>
            <a:Value>
                <BiRerong>385</BiRerong>
                <DaoreXishu>400</DaoreXishu>
                <Midu>8920</Midu>
            </a:Value>
        </a:KeyValueOfstringMaterialGyu7GfTz>
    </Materials>
    <OtherThermalBondary i:type="SecondTypeThermalBoundary">
        <a:HeatFlux>0</a:HeatFlux>
    </OtherThermalBondary>
    <Results>
        <a:anyType i:type="Result3D">
            <Mesh>
                <b:XArray xmlns:b="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
                    <a:double>0</a:double>
                    <a:double>5</a:double>
                    <a:double>10</a:double>
                </b:XArray>
                <b:YArray xmlns:b="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
                    <a:double>0</a:double>
                    <a:double>25</a:double>
                    <a:double>50</a:double>
                    <a:double>75</a:double>
                    <a:double>100</a:double>
                </b:YArray>
                <b:ZArray xmlns:b="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
                    <a:double>0</a:double>
                    <a:double>5</a:double>
                    <a:double>10</a:double>
                </b:ZArray>
            </Mesh>
            <Values>
                <SizeX>3</SizeX>
                <SizeY>5</SizeY>
                <SizeZ>3</SizeZ>
                <Data><a:double>300</a:double></Data>
            </Values>
        </a:anyType>
    </Results>
    <StudyType>Transient</StudyType>
    {ts_block}
    <TransientStudyDuration>{duration}</TransientStudyDuration>
    <TransientStudyTimeStep>{time_step}</TransientStudyTimeStep>
    <TransientTimeUnit>S</TransientTimeUnit>
</Structure>"""


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    # BDF2 case
    with open(os.path.join(out_dir, "case1.xml"), "w", encoding="utf-8") as f:
        f.write(make_case(
            duration=10.0, time_step=1.0,
            scheme="Bdf2", initial_dt=1.0, output_dt=1.0))

    # Adaptive cases
    adaptive_dir = os.path.dirname(out_dir.rstrip("/")) + "/adaptive_transient_tests"
    os.makedirs(adaptive_dir, exist_ok=True)

    with open(os.path.join(adaptive_dir, "case1.xml"), "w", encoding="utf-8") as f:
        f.write(make_case(
            duration=2.0, time_step=0.5,
            scheme="AdaptiveBdf", initial_dt=0.5, output_dt=0.5,
            min_dt=0.001, max_dt=0.5))

    with open(os.path.join(adaptive_dir, "case2.xml"), "w", encoding="utf-8") as f:
        f.write(make_case(
            duration=20.0, time_step=0.5,
            scheme="AdaptiveBdf", initial_dt=0.1, output_dt=1.0,
            min_dt=0.001, max_dt=2.0))

    with open(os.path.join(adaptive_dir, "case3.xml"), "w", encoding="utf-8") as f:
        f.write(make_case(
            duration=10.0, time_step=0.3,
            scheme="AdaptiveBdf", initial_dt=0.3, output_dt=0.7,
            min_dt=0.001, max_dt=0.3))
